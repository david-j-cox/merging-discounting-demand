"""Hierarchical Bayesian fitting via NumPyro.

Two models: :func:`hierarchical_unified` (joint purchase + effort fit;
the primary model for the main study) and :func:`hierarchical_demand`
(Koffarnus alone, for sanity comparisons). Both use non-centered
parameterisation to avoid NUTS funnel pathologies. Returns
``arviz.InferenceData``.
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


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Priors:
    """Weakly-informative population hyperpriors.

    Defaults are matched to the synthetic priors in
    :mod:`edu.simulation.generate`: log-normal on ``alpha`` and ``Q0``,
    truncated-normal on ``k``, half-Cauchy on the population sds and
    likelihood noise. ``*_mean`` is a Normal prior mean, ``*_mean_sd``
    its prior sd; ``*_scale_concentration`` is the half-Cauchy scale
    for the corresponding population sd.
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
    """Hierarchical Koffarnus demand for ``Q_obs`` of shape ``(n_subj, n_prices)``."""
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
    arm_index: jnp.ndarray | None = None,
) -> None:
    """Hierarchical unified-model joint fit (purchase + effort discount).

    ``k`` is shared across subjects per Hursh's standard practice (see
    ``docs/analysis_plan.md``); fitting per-subject ``k`` produces an
    alpha-k identifiability ridge that recovery cannot resolve.

    ``B_anchor`` is the per-subject measured capability — anchored, not
    estimated. The recovery sweep in :mod:`edu.simulation.recovery`
    quantifies sensitivity to perturbing it.

    ``arm_index`` (per-subject 0/1) gives a separate ``mu_log_alpha``
    per arm and exposes ``diff_log_alpha`` as a deterministic. Pass it
    whenever H3 is being tested; the single-population fit followed by
    post-hoc subject slicing biases the H3 contrast (see notebook 05).
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

    k_shared = numpyro.sample("k_shared", dist.Normal(priors.k_mean, priors.k_mean_sd))

    sigma_purchase = numpyro.sample(
        "sigma_purchase", dist.HalfCauchy(priors.sigma_purchase_concentration)
    )
    sigma_sv = numpyro.sample("sigma_sv", dist.HalfCauchy(priors.sigma_sv_concentration))

    with numpyro.plate("subjects", n_subj):
        z_la = numpyro.sample("z_log_alpha", dist.Normal(0.0, 1.0))
        z_lQ0 = numpyro.sample("z_log_Q0", dist.Normal(0.0, 1.0))

    log_alpha = mu_la + sigma_la * z_la
    log_Q0 = mu_lQ0 + sigma_lQ0 * z_lQ0
    k = jnp.broadcast_to(jnp.clip(k_shared, 0.5, 6.0), (n_subj,))

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
    """Run NUTS on :func:`hierarchical_demand`."""
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
    return az.from_numpyro(mcmc)  # type: ignore[no-any-return,no-untyped-call]


def fit_unified_hierarchical(
    P: FloatArray,
    Q_obs: FloatArray,
    E: FloatArray,
    SV_obs: FloatArray,
    B_anchor: FloatArray,
    *,
    A: float = 10.0,
    priors: Priors = DEFAULT_PRIORS,
    arm_index: NDArray[np.integer[Any]] | None = None,
    n_warmup: int = 1000,
    n_samples: int = 1000,
    n_chains: int = 2,
    seed: int = 2025,
    progress_bar: bool = False,
) -> az.InferenceData:
    """Run NUTS on :func:`hierarchical_unified`.

    Pass ``arm_index`` (per-subject 0/1 array) whenever the H3 contrast
    will be evaluated; the deterministic ``diff_log_alpha`` it exposes
    avoids the bias documented in ``notebooks/05_pilot_analysis.ipynb``.
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
            arm_index=arm_j,
        )
    return az.from_numpyro(mcmc)  # type: ignore[no-any-return,no-untyped-call]


__all__ = [
    "DEFAULT_PRIORS",
    "Priors",
    "fit_demand_hierarchical",
    "fit_unified_hierarchical",
    "hierarchical_demand",
    "hierarchical_unified",
]
