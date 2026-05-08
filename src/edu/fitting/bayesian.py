"""Hierarchical Bayesian fitting via NumPyro.

CLAUDE.md §5: *"Hierarchical Bayesian throughout. Use NumPyro for speed.
Specify priors in pre-registration. Report posterior distributions for all
parameters; do not collapse to point estimates. Use posterior predictive
checks for model adequacy."*

This module provides per-subject and hierarchical models for the demand,
discount, and unified-model parameterisations introduced in Phase 1-2.
The hierarchical pooling on ``alpha`` is the load-bearing piece for
CLAUDE.md H1: a shared population mean constrains individual ``alpha``
estimates, which is the partial-pooling analogue of the NLS shared-vs-
independent ΔBIC test.

Two model shapes:

* **Joint unified** (:func:`hierarchical_unified`) — the primary fitting
  model for the main study. Per-subject ``Q0_i, alpha_i, k_i`` drawn
  from population hyperpriors; ``B_i`` is anchored to measured capability
  (passed in as data). Two task likelihoods:

  - log10-Gaussian on purchase-task consumption (``Q``).
  - linear Gaussian on effort-discount subjective values (``SV``).

* **Single-task demand** (:func:`hierarchical_demand`) — Koffarnus alone,
  for sanity comparisons against the joint model. Same hierarchical
  structure on ``Q0, alpha, k``.

All models use a non-centered parameterisation on the per-subject deviates
to avoid funnel pathologies in NUTS sampling.

Conventions
-----------
* Priors live in :class:`Priors`. Defaults match ``docs/analysis_plan.md``
  and were tuned against the synthetic priors in
  :mod:`edu.simulation.generate`.
* All models accept an explicit ``rng_key`` (a JAX PRNG key). The
  fitting helpers wrap key creation from a Python ``int`` seed.
* :func:`fit_unified_hierarchical` returns ``arviz.InferenceData`` so
  callers get the full posterior + diagnostics in one object.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpy.typing import NDArray
from numpyro.infer import MCMC, NUTS

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Priors:
    """Population-level hyperpriors for the hierarchical model.

    All defaults are weakly informative, matched to the population
    distributions used in :mod:`edu.simulation.generate` (log-normal
    priors on ``alpha``, ``Q0``, ``B``; truncated normal on ``k``).

    Attributes
    ----------
    log_alpha_mean, log_alpha_mean_sd
        Mean and prior-sd of the log10-mean of ``alpha`` across subjects.
    log_alpha_scale_concentration
        Half-Cauchy scale for the population sd of log10(alpha).
    log_Q0_mean, log_Q0_mean_sd
        Same for ``Q0``.
    log_Q0_scale_concentration
        Half-Cauchy scale for the population sd of log10(Q0).
    k_mean, k_mean_sd
        Population mean and its prior-sd for ``k`` (linear scale).
    k_scale_concentration
        Half-Cauchy scale for the population sd of ``k``.
    sigma_purchase_concentration
        Half-Cauchy scale for the log10-Gaussian noise on purchase data.
    sigma_sv_concentration
        Half-Cauchy scale for the linear-Gaussian noise on SV.
    """

    log_alpha_mean: float = -2.0
    log_alpha_mean_sd: float = 0.5
    log_alpha_scale_concentration: float = 0.5

    log_Q0_mean: float = 1.0
    log_Q0_mean_sd: float = 0.5
    log_Q0_scale_concentration: float = 0.5

    k_mean: float = 3.0
    k_mean_sd: float = 0.5
    k_scale_concentration: float = 0.5

    sigma_purchase_concentration: float = 0.5
    sigma_sv_concentration: float = 1.0


DEFAULT_PRIORS = Priors()


# ---------------------------------------------------------------------------
# Model functions (NumPyro)
# ---------------------------------------------------------------------------


def _koffarnus_q(
    Q0: jnp.ndarray, alpha: jnp.ndarray, k: jnp.ndarray, P: jnp.ndarray
) -> jnp.ndarray:
    """Vectorised Koffarnus Q at a stack of per-subject params and shared price grid.

    Shapes:
        Q0, alpha, k: (n_subjects,)
        P: (n_prices,)
    Returns: (n_subjects, n_prices)
    """
    Q0_b = Q0[:, None]
    alpha_b = alpha[:, None]
    k_b = k[:, None]
    P_b = P[None, :]
    return Q0_b * jnp.power(10.0, k_b * (jnp.exp(-alpha_b * Q0_b * P_b) - 1.0))


def _unified_sv(
    A: float,
    Q0: jnp.ndarray,
    alpha: jnp.ndarray,
    k: jnp.ndarray,
    B: jnp.ndarray,
    E: jnp.ndarray,
) -> jnp.ndarray:
    """Capability-bounded SV across subjects and effort levels.

    Shapes:
        Q0, alpha, k, B: (n_subjects,)
        E: (n_subjects, n_effort) — per-subject effort grids (E = fraction * B)
    Returns: (n_subjects, n_effort)
    """
    Q0_b = Q0[:, None]
    alpha_b = alpha[:, None]
    k_b = k[:, None]
    B_b = B[:, None]
    demand_term = jnp.power(10.0, k_b * (jnp.exp(-alpha_b * Q0_b * E / A) - 1.0))
    capability_term = jnp.where(E > 0, B_b / (Q0_b * E), jnp.inf)
    return A * jnp.minimum(demand_term, capability_term)


def hierarchical_demand(
    P: jnp.ndarray,
    Q_obs: jnp.ndarray | None = None,
    *,
    priors: Priors = DEFAULT_PRIORS,
) -> None:
    """NumPyro model for hierarchical Koffarnus demand.

    Parameters
    ----------
    P
        Shared price array, shape ``(n_prices,)``.
    Q_obs
        Observed consumption, shape ``(n_subjects, n_prices)`` or ``None``
        for prior-predictive sampling.
    priors
        Population hyperpriors.
    """
    if Q_obs is None:
        msg = "Q_obs is required for posterior sampling; pass shape (n_subjects, n_prices)."
        raise ValueError(msg)

    n_subj = Q_obs.shape[0]

    # Population hyperpriors.
    mu_la = numpyro.sample(
        "mu_log_alpha", dist.Normal(priors.log_alpha_mean, priors.log_alpha_mean_sd)
    )
    sigma_la = numpyro.sample(
        "sigma_log_alpha", dist.HalfCauchy(priors.log_alpha_scale_concentration)
    )
    mu_lQ0 = numpyro.sample("mu_log_Q0", dist.Normal(priors.log_Q0_mean, priors.log_Q0_mean_sd))
    sigma_lQ0 = numpyro.sample("sigma_log_Q0", dist.HalfCauchy(priors.log_Q0_scale_concentration))
    mu_k = numpyro.sample("mu_k", dist.Normal(priors.k_mean, priors.k_mean_sd))
    sigma_k = numpyro.sample("sigma_k", dist.HalfCauchy(priors.k_scale_concentration))
    sigma_obs = numpyro.sample("sigma_obs", dist.HalfCauchy(priors.sigma_purchase_concentration))

    # Non-centered per-subject offsets.
    with numpyro.plate("subjects", n_subj):
        z_la = numpyro.sample("z_log_alpha", dist.Normal(0.0, 1.0))
        z_lQ0 = numpyro.sample("z_log_Q0", dist.Normal(0.0, 1.0))
        z_k = numpyro.sample("z_k", dist.Normal(0.0, 1.0))

    log_alpha = mu_la + sigma_la * z_la
    log_Q0 = mu_lQ0 + sigma_lQ0 * z_lQ0
    k = jnp.clip(mu_k + sigma_k * z_k, 0.5, 6.0)

    alpha = numpyro.deterministic("alpha", jnp.power(10.0, log_alpha))
    Q0 = numpyro.deterministic("Q0", jnp.power(10.0, log_Q0))
    numpyro.deterministic("k_subj", k)

    Q_pred = _koffarnus_q(Q0, alpha, k, P)
    log_q_pred = jnp.log10(jnp.maximum(Q_pred, 1e-12))
    log_q_obs = jnp.log10(jnp.maximum(Q_obs, 1e-12))
    numpyro.sample("Q_obs", dist.Normal(log_q_pred, sigma_obs), obs=log_q_obs)


def hierarchical_unified(
    P: jnp.ndarray,
    Q_obs: jnp.ndarray,
    E: jnp.ndarray,
    SV_obs: jnp.ndarray,
    B_anchor: jnp.ndarray,
    A: float = 10.0,
    *,
    priors: Priors = DEFAULT_PRIORS,
    share_k: bool = True,
    arm_index: jnp.ndarray | None = None,
) -> None:
    """NumPyro model for the hierarchical unified-model joint fit.

    Parameters
    ----------
    P
        Shared price array, shape ``(n_prices,)``.
    Q_obs
        Purchase-task consumption, shape ``(n_subjects, n_prices)``.
    E
        Per-subject effort grids (effort = fraction * B), shape
        ``(n_subjects, n_effort)``.
    SV_obs
        Subjective-value observations, shape ``(n_subjects, n_effort)``.
    B_anchor
        Per-subject measured capability, shape ``(n_subjects,)``. Anchored,
        not estimated. Pass the *measured* value (with whatever measurement
        noise the experiment introduced); the recovery sweep tests how
        sensitive estimates are to perturbing this.
    A
        Reward magnitude (design constant).
    priors
        Population hyperpriors.
    share_k
        If True (default), sample a single ``k`` across all subjects —
        Hursh's standard practice in the demand-curve literature. If False,
        ``k`` partially pools through a hierarchical normal. The shared-k
        form has one population parameter for ``k`` and no per-subject
        deviates, which is the most identifiable choice and the one
        pre-registered in ``docs/analysis_plan.md``.
    arm_index
        Optional integer array of shape ``(n_subjects,)`` with values 0
        (low-IF arm) or 1 (high-IF arm). When supplied, the model
        estimates a separate ``mu_log_alpha`` per arm; the H3 arm
        contrast is then read directly from ``mu_log_alpha_per_arm`` in
        the posterior. When ``None`` (default), one global
        ``mu_log_alpha`` is shared across all subjects, which is
        appropriate for the synthetic recovery experiments and the
        single-arm pilot but **biases the H3 contrast** in real data
        (see notebook 05). Pass the arm index whenever H3 is being
        tested.
    """
    n_subj = Q_obs.shape[0]

    if arm_index is not None:
        # Two-arm model: per-arm population means, shared population sd.
        mu_la_per_arm = numpyro.sample(
            "mu_log_alpha_per_arm",
            dist.Normal(
                priors.log_alpha_mean * jnp.ones(2),
                priors.log_alpha_mean_sd * jnp.ones(2),
            ),
        )
        mu_la = mu_la_per_arm[arm_index]  # broadcast to (n_subj,)
        # Expose the contrast directly for downstream analysis.
        # Stored on the natural-log scale; the analysis converts to log10.
        numpyro.deterministic("diff_log_alpha", mu_la_per_arm[0] - mu_la_per_arm[1])
    else:
        mu_la = numpyro.sample(
            "mu_log_alpha", dist.Normal(priors.log_alpha_mean, priors.log_alpha_mean_sd)
        )
    sigma_la = numpyro.sample(
        "sigma_log_alpha", dist.HalfCauchy(priors.log_alpha_scale_concentration)
    )
    mu_lQ0 = numpyro.sample("mu_log_Q0", dist.Normal(priors.log_Q0_mean, priors.log_Q0_mean_sd))
    sigma_lQ0 = numpyro.sample("sigma_log_Q0", dist.HalfCauchy(priors.log_Q0_scale_concentration))

    if share_k:
        k_shared = numpyro.sample("k_shared", dist.Normal(priors.k_mean, priors.k_mean_sd))
    else:
        mu_k = numpyro.sample("mu_k", dist.Normal(priors.k_mean, priors.k_mean_sd))
        sigma_k = numpyro.sample("sigma_k", dist.HalfCauchy(priors.k_scale_concentration))

    sigma_purchase = numpyro.sample(
        "sigma_purchase", dist.HalfCauchy(priors.sigma_purchase_concentration)
    )
    sigma_sv = numpyro.sample("sigma_sv", dist.HalfCauchy(priors.sigma_sv_concentration))

    with numpyro.plate("subjects", n_subj):
        z_la = numpyro.sample("z_log_alpha", dist.Normal(0.0, 1.0))
        z_lQ0 = numpyro.sample("z_log_Q0", dist.Normal(0.0, 1.0))
        if not share_k:
            z_k = numpyro.sample("z_k", dist.Normal(0.0, 1.0))

    log_alpha = mu_la + sigma_la * z_la
    log_Q0 = mu_lQ0 + sigma_lQ0 * z_lQ0
    if share_k:
        k = jnp.broadcast_to(jnp.clip(k_shared, 0.5, 6.0), (n_subj,))
    else:
        k = jnp.clip(mu_k + sigma_k * z_k, 0.5, 6.0)

    alpha = numpyro.deterministic("alpha", jnp.power(10.0, log_alpha))
    Q0 = numpyro.deterministic("Q0", jnp.power(10.0, log_Q0))
    numpyro.deterministic("k_subj", k)

    # Purchase-task likelihood (log10).
    Q_pred = _koffarnus_q(Q0, alpha, k, P)
    log_q_pred = jnp.log10(jnp.maximum(Q_pred, 1e-12))
    log_q_obs = jnp.log10(jnp.maximum(Q_obs, 1e-12))
    numpyro.sample("Q_obs", dist.Normal(log_q_pred, sigma_purchase), obs=log_q_obs)

    # Effort-discount likelihood (linear).
    sv_pred = _unified_sv(A, Q0, alpha, k, B_anchor, E)
    numpyro.sample("SV_obs", dist.Normal(sv_pred, sigma_sv), obs=SV_obs)


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------


def fit_demand_hierarchical(
    P: FloatArray,
    Q_obs: FloatArray,
    *,
    priors: Priors = DEFAULT_PRIORS,
    n_warmup: int = 1000,
    n_samples: int = 1000,
    n_chains: int = 2,
    seed: int = 2025,
    progress_bar: bool = False,
) -> az.InferenceData:
    """Run NUTS on :func:`hierarchical_demand`. Returns InferenceData."""
    P_j = jnp.asarray(P, dtype=jnp.float32)
    Q_j = jnp.asarray(Q_obs, dtype=jnp.float32)
    nuts = NUTS(hierarchical_demand)
    mcmc = MCMC(
        nuts,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        progress_bar=progress_bar,
        chain_method="sequential",
    )
    rng_key = jax.random.PRNGKey(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mcmc.run(rng_key, P=P_j, Q_obs=Q_j, priors=priors)
    return az.from_numpyro(mcmc)  # type: ignore[no-any-return,no-untyped-call,unused-ignore]


def fit_unified_hierarchical(
    P: FloatArray,
    Q_obs: FloatArray,
    E: FloatArray,
    SV_obs: FloatArray,
    B_anchor: FloatArray,
    *,
    A: float = 10.0,
    priors: Priors = DEFAULT_PRIORS,
    share_k: bool = True,
    arm_index: NDArray[np.integer[Any]] | None = None,
    n_warmup: int = 1000,
    n_samples: int = 1000,
    n_chains: int = 2,
    seed: int = 2025,
    progress_bar: bool = False,
) -> az.InferenceData:
    """Run NUTS on :func:`hierarchical_unified`. Returns InferenceData.

    Pass ``arm_index`` (per-subject 0/1 array) whenever the H3 contrast
    will be evaluated; this gives the model a per-arm
    ``mu_log_alpha_per_arm`` and an exposed ``diff_log_alpha``
    deterministic, both of which avoid the bias documented in
    ``notebooks/05_pilot_analysis.ipynb``.
    """
    P_j = jnp.asarray(P, dtype=jnp.float32)
    Q_j = jnp.asarray(Q_obs, dtype=jnp.float32)
    E_j = jnp.asarray(E, dtype=jnp.float32)
    SV_j = jnp.asarray(SV_obs, dtype=jnp.float32)
    B_j = jnp.asarray(B_anchor, dtype=jnp.float32)
    arm_j = jnp.asarray(arm_index, dtype=jnp.int32) if arm_index is not None else None
    nuts = NUTS(hierarchical_unified)
    mcmc = MCMC(
        nuts,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        progress_bar=progress_bar,
        chain_method="sequential",
    )
    rng_key = jax.random.PRNGKey(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mcmc.run(
            rng_key,
            P=P_j,
            Q_obs=Q_j,
            E=E_j,
            SV_obs=SV_j,
            B_anchor=B_j,
            A=A,
            priors=priors,
            share_k=share_k,
            arm_index=arm_j,
        )
    return az.from_numpyro(mcmc)  # type: ignore[no-any-return,no-untyped-call,unused-ignore]


__all__ = [
    "DEFAULT_PRIORS",
    "Priors",
    "fit_demand_hierarchical",
    "fit_unified_hierarchical",
    "hierarchical_demand",
    "hierarchical_unified",
]
