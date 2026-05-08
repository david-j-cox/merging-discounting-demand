"""Group-level analyses for the pre-registered hypotheses.

Three primary outputs:

- :func:`linkage_correlation` — H1-stronger. Posterior correlation between
  per-subject ``alpha`` (from the hierarchical fit) and an externally
  derived effort-discount steepness (typically a per-subject NLS fit of
  one of the discount forms to Task 2 alone). Propagates posterior
  uncertainty in ``alpha`` through to the correlation.

- :func:`arm_contrast` — H3. Posterior of the difference in population-
  level ``alpha`` between substitutability arms. Reads the per-subject
  arm assignment plus the alpha posterior; reports the difference of
  arm-conditional means and the posterior probability that the
  difference is in the predicted direction.

- :func:`posterior_predictive_purchase` and
  :func:`posterior_predictive_sv` — PPCs. Generate replicate purchase
  and SV data from posterior draws so callers can compare against
  observations.

All functions accept the InferenceData object from
:func:`edu.fitting.bayesian.fit_unified_hierarchical`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import arviz as az
import numpy as np
from numpy.typing import NDArray

from edu.analysis.individual import _stack_samples

FloatArray = NDArray[np.floating[Any]]


# ---------------------------------------------------------------------------
# H1-stronger: linkage correlation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinkageCorrelation:
    """Posterior summary of the per-subject correlation.

    ``alpha`` (from the joint Bayesian fit) is correlated against
    ``lambda_external`` — typically a per-subject NLS estimate of the
    steepness of effort discounting from Task 2 in isolation.

    Attributes
    ----------
    correlation_mean : float
        Mean of the posterior on Pearson r.
    correlation_hdi : (float, float)
        95% HDI on the correlation.
    p_positive : float
        Posterior probability that the correlation is positive (the
        H1-stronger predicted direction).
    n_subjects : int
        How many subjects contributed to the correlation.
    n_draws : int
        Number of posterior draws used.
    """

    correlation_mean: float
    correlation_hdi: tuple[float, float]
    p_positive: float
    n_subjects: int
    n_draws: int


def linkage_correlation(
    idata: az.InferenceData,
    lambda_external: FloatArray,
    *,
    transform: Literal["log", "linear"] = "log",
) -> LinkageCorrelation:
    """Posterior correlation between subject-level ``alpha`` and an external steepness.

    Each posterior draw of ``alpha[i]`` (vector across subjects) is
    correlated against the fixed ``lambda_external``; the distribution
    of correlations across draws is the posterior of Pearson r.

    Parameters
    ----------
    idata
        Hierarchical posterior; must have ``alpha`` of shape
        ``(chain, draw, n_subjects)``.
    lambda_external
        Per-subject externally-derived steepness, length ``n_subjects``.
        Treated as fixed (its measurement uncertainty is not propagated
        in this function; see :func:`linkage_correlation_propagated` for
        the version that does).
    transform
        ``"log"`` correlates ``log(alpha)`` against ``log(lambda_external)``
        (matches the model-derived linkage on a multiplicative scale);
        ``"linear"`` does not transform.
    """
    alpha = _stack_samples(idata, "alpha")  # (n_draws, n_subj)
    n_draws, n_subj = alpha.shape
    if lambda_external.shape != (n_subj,):
        msg = f"lambda_external must have length n_subj={n_subj}; got shape {lambda_external.shape}"
        raise ValueError(msg)

    if transform == "log":
        x_per_draw = np.log(np.maximum(alpha, 1e-12))
        y = np.log(np.maximum(lambda_external, 1e-12))
    else:
        x_per_draw = alpha
        y = lambda_external

    y_centered = y - np.mean(y)
    y_norm = float(np.sqrt(np.sum(y_centered**2)))

    correlations = np.empty(n_draws, dtype=float)
    for d in range(n_draws):
        x = x_per_draw[d]
        x_centered = x - np.mean(x)
        x_norm = float(np.sqrt(np.sum(x_centered**2)))
        if x_norm == 0 or y_norm == 0:
            correlations[d] = float("nan")
        else:
            correlations[d] = float(np.dot(x_centered, y_centered) / (x_norm * y_norm))

    valid = correlations[np.isfinite(correlations)]
    if valid.size == 0:
        return LinkageCorrelation(
            correlation_mean=float("nan"),
            correlation_hdi=(float("nan"), float("nan")),
            p_positive=float("nan"),
            n_subjects=n_subj,
            n_draws=n_draws,
        )
    bounds = np.asarray(az.hdi(valid, hdi_prob=0.95)).flatten()  # type: ignore[no-untyped-call,unused-ignore]
    return LinkageCorrelation(
        correlation_mean=float(np.mean(valid)),
        correlation_hdi=(float(bounds[0]), float(bounds[1])),
        p_positive=float(np.mean(valid > 0)),
        n_subjects=n_subj,
        n_draws=n_draws,
    )


# ---------------------------------------------------------------------------
# H3: arm contrast
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmContrast:
    """Posterior of the population-level alpha difference between arms.

    Attributes
    ----------
    diff_log_alpha_mean : float
        Posterior mean of ``mean(log10 alpha | low) - mean(log10 alpha | high)``.
        H3 predicts a positive value (low-IF arm has steeper discounting).
    diff_log_alpha_hdi : (float, float)
        95% HDI on the difference.
    p_predicted_direction : float
        Posterior probability that the difference is positive.
    n_low, n_high : int
        Subject counts per arm.
    n_draws : int
        Number of posterior draws used.
    """

    diff_log_alpha_mean: float
    diff_log_alpha_hdi: tuple[float, float]
    p_predicted_direction: float
    n_low: int
    n_high: int
    n_draws: int


def arm_contrast(
    idata: az.InferenceData,
    arm: NDArray[np.str_] | list[str],
) -> ArmContrast:
    """Compute the posterior of the per-arm alpha difference.

    Two paths:

    - **Preferred** (used when the fit was run with ``arm_index=...``):
      read the deterministic ``diff_log_alpha`` directly from the posterior.
      Equivalent to ``log10`` of the ratio of the per-arm population means
      and is unbiased under the synthetic test in
      ``notebooks/05_pilot_analysis.ipynb``.

    - **Fallback** (single-population fit): construct the contrast
      ``mean(log10 alpha_i | low) - mean(log10 alpha_i | high)`` from the
      per-subject ``alpha`` posterior. This path is biased upward by
      ~0.05 log units when arms genuinely differ — fine for sensitivity
      checks but not for the primary H3 result. A warning is emitted
      when this path is taken.

    Parameters
    ----------
    idata
        Hierarchical posterior. Must contain ``alpha`` of shape
        (chain, draw, n_subj). If it also contains ``diff_log_alpha``
        (i.e. the model was fit with ``arm_index=...``) the preferred
        path is used.
    arm
        Per-subject arm assignment, length n_subj. Each entry must be
        ``"low"`` or ``"high"``.
    """
    arm_arr = np.asarray(arm)

    # Preferred path: the model exposed diff_log_alpha as a deterministic.
    # Stored on the natural-log scale by NumPyro; convert to log10 for
    # consistency with the docstring and analysis plan.
    if "diff_log_alpha" in idata.posterior:  # type: ignore[attr-defined,unused-ignore]
        diffs_natural = _stack_samples(idata, "diff_log_alpha")  # (n_draws,)
        diffs = diffs_natural / np.log(10.0)  # convert ln to log10
        n_draws = diffs.shape[0]
        n_low = int((arm_arr == "low").sum())
        n_high = int((arm_arr == "high").sum())
        bounds = np.asarray(az.hdi(diffs, hdi_prob=0.95)).flatten()  # type: ignore[no-untyped-call,unused-ignore]
        return ArmContrast(
            diff_log_alpha_mean=float(np.mean(diffs)),
            diff_log_alpha_hdi=(float(bounds[0]), float(bounds[1])),
            p_predicted_direction=float(np.mean(diffs > 0)),
            n_low=n_low,
            n_high=n_high,
            n_draws=n_draws,
        )

    # Fallback path. Warn callers; this is biased.
    import warnings as _warnings

    _warnings.warn(
        "arm_contrast was called on a single-population fit; the contrast "
        "is biased. Pass arm_index= to fit_unified_hierarchical for the "
        "unbiased contrast.",
        stacklevel=2,
    )
    alpha = _stack_samples(idata, "alpha")  # (n_draws, n_subj)
    n_draws, n_subj = alpha.shape
    if arm_arr.shape != (n_subj,):
        msg = f"arm must have length n_subj={n_subj}; got shape {arm_arr.shape}"
        raise ValueError(msg)
    is_low = arm_arr == "low"
    is_high = arm_arr == "high"
    n_low = int(is_low.sum())
    n_high = int(is_high.sum())
    if n_low == 0 or n_high == 0:
        msg = "Both arms must have at least one subject."
        raise ValueError(msg)

    log_alpha = np.log10(np.maximum(alpha, 1e-12))
    diffs = log_alpha[:, is_low].mean(axis=1) - log_alpha[:, is_high].mean(axis=1)
    bounds = np.asarray(az.hdi(diffs, hdi_prob=0.95)).flatten()  # type: ignore[no-untyped-call,unused-ignore]
    return ArmContrast(
        diff_log_alpha_mean=float(np.mean(diffs)),
        diff_log_alpha_hdi=(float(bounds[0]), float(bounds[1])),
        p_predicted_direction=float(np.mean(diffs > 0)),
        n_low=n_low,
        n_high=n_high,
        n_draws=n_draws,
    )


# ---------------------------------------------------------------------------
# Posterior predictive checks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PosteriorPredictive:
    """Replicate datasets drawn from the posterior.

    Attributes
    ----------
    replicates
        Shape ``(n_draws, n_subjects, n_points)``. Each replicate is a
        sampled value from the posterior predictive at the given price /
        effort grid point.
    observed
        Shape ``(n_subjects, n_points)``. The observed data for direct
        comparison.
    grid
        Shape ``(n_points,)``. The price or effort grid points.
    """

    replicates: FloatArray
    observed: FloatArray
    grid: FloatArray


def posterior_predictive_purchase(
    idata: az.InferenceData,
    P: FloatArray,
    Q_obs: FloatArray,
    *,
    n_draws: int | None = None,
    rng: np.random.Generator | None = None,
) -> PosteriorPredictive:
    """Sample replicate purchase-task Q values from the posterior.

    Uses the model's mean prediction at each posterior draw plus the
    posterior ``sigma_purchase`` to draw log10-Gaussian noise.
    """
    alpha = _stack_samples(idata, "alpha")  # (D, n_subj)
    Q0 = _stack_samples(idata, "Q0")
    if "k_shared" in idata.posterior:  # type: ignore[attr-defined,unused-ignore]
        k_full = _stack_samples(idata, "k_shared")  # (D,)
        k_full = np.broadcast_to(k_full[:, None], alpha.shape)
    else:
        k_full = _stack_samples(idata, "k_subj")
    sigma_purchase = _stack_samples(idata, "sigma_purchase")  # (D,)

    D, n_subj = alpha.shape
    n_p = P.shape[0]
    if n_draws is not None and n_draws < D:
        idx = (rng or np.random.default_rng(0)).choice(D, n_draws, replace=False)
        alpha = alpha[idx]
        Q0 = Q0[idx]
        k_full = k_full[idx]
        sigma_purchase = sigma_purchase[idx]
        D = n_draws

    rng = rng or np.random.default_rng(0)
    Q0_b = Q0[:, :, None]  # (D, n_subj, 1)
    alpha_b = alpha[:, :, None]
    k_b = k_full[:, :, None]
    P_b = P[None, None, :]
    Q_pred = Q0_b * np.power(10.0, k_b * (np.exp(-alpha_b * Q0_b * P_b) - 1.0))
    log_q_pred = np.log10(np.maximum(Q_pred, 1e-12))
    noise = rng.normal(0.0, sigma_purchase[:, None, None], size=(D, n_subj, n_p))
    log_q_repl = log_q_pred + noise
    replicates = np.power(10.0, log_q_repl)
    return PosteriorPredictive(
        replicates=replicates,
        observed=Q_obs,
        grid=P,
    )


def posterior_predictive_sv(
    idata: az.InferenceData,
    E: FloatArray,
    SV_obs: FloatArray,
    B_anchor: FloatArray,
    *,
    A: float = 10.0,
    n_draws: int | None = None,
    rng: np.random.Generator | None = None,
) -> PosteriorPredictive:
    """Sample replicate SV values from the unified-model posterior.

    Each draw evaluates the capability-bounded SV at the per-subject
    effort grid plus posterior ``sigma_sv`` Gaussian noise.
    """
    alpha = _stack_samples(idata, "alpha")
    Q0 = _stack_samples(idata, "Q0")
    if "k_shared" in idata.posterior:  # type: ignore[attr-defined,unused-ignore]
        k_full = _stack_samples(idata, "k_shared")
        k_full = np.broadcast_to(k_full[:, None], alpha.shape)
    else:
        k_full = _stack_samples(idata, "k_subj")
    sigma_sv = _stack_samples(idata, "sigma_sv")

    D, n_subj = alpha.shape
    if n_draws is not None and n_draws < D:
        idx = (rng or np.random.default_rng(0)).choice(D, n_draws, replace=False)
        alpha = alpha[idx]
        Q0 = Q0[idx]
        k_full = k_full[idx]
        sigma_sv = sigma_sv[idx]
        D = n_draws
    rng = rng or np.random.default_rng(0)

    n_e = E.shape[1]
    Q0_b = Q0[:, :, None]
    alpha_b = alpha[:, :, None]
    k_b = k_full[:, :, None]
    B_b = B_anchor[None, :, None]
    E_b = E[None, :, :]
    demand_term = np.power(10.0, k_b * (np.exp(-alpha_b * Q0_b * E_b / A) - 1.0))
    capability_term = np.where(E_b > 0, B_b / (Q0_b * E_b), np.inf)
    sv_pred = A * np.minimum(demand_term, capability_term)
    noise = rng.normal(0.0, sigma_sv[:, None, None], size=(D, n_subj, n_e))
    replicates = np.maximum(sv_pred + noise, 0.0)
    return PosteriorPredictive(
        replicates=replicates,
        observed=SV_obs,
        grid=np.asarray(E[0], dtype=float),  # representative grid (per-subj scaled)
    )


__all__ = [
    "ArmContrast",
    "LinkageCorrelation",
    "PosteriorPredictive",
    "arm_contrast",
    "linkage_correlation",
    "posterior_predictive_purchase",
    "posterior_predictive_sv",
]
