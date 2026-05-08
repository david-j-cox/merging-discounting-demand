"""Group-level analyses for the pre-registered hypotheses.

All functions consume the ``InferenceData`` from
:func:`edu.fitting.bayesian.fit_unified_hierarchical`:

- :func:`linkage_correlation` — H1-stronger.
- :func:`arm_contrast` — H3.
- :func:`posterior_predictive_purchase`, :func:`posterior_predictive_sv`
  — model-adequacy PPCs.
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
    """Posterior summary of Pearson r between per-subject alpha and an external steepness.

    ``p_positive`` is the posterior probability that ``r > 0`` (the
    H1-stronger predicted direction).
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
    """Posterior of Pearson r between per-subject ``alpha`` and ``lambda_external``.

    For each posterior draw, correlate the vector of per-subject
    ``alpha`` values against ``lambda_external``; the distribution of
    those correlations is the posterior r. The default ``log`` transform
    matches the model-derived multiplicative linkage; ``linear`` skips
    the transform.

    ``lambda_external`` is treated as fixed — its own measurement
    uncertainty is not propagated. (Substituting posterior samples for
    both quantities is the natural extension when needed.)
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
    bounds = np.asarray(az.hdi(valid, hdi_prob=0.95)).flatten()  # type: ignore[no-untyped-call]
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
    """Posterior of ``log10(alpha | low) - log10(alpha | high)`` (H3).

    H3 predicts a positive value (low-IF arm has steeper discounting).
    ``p_predicted_direction`` is the posterior probability the
    difference is positive.
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
    """Posterior of the H3 alpha contrast between low-IF and high-IF arms.

    Preferred path: when ``idata`` has ``diff_log_alpha`` (the fit was
    run with ``arm_index=...``), read the deterministic directly.

    Fallback: contrast per-subject ``alpha`` posteriors by arm. Biased
    upward by ~0.05 log units when arms genuinely differ; a warning is
    emitted. See ``notebooks/05_pilot_analysis.ipynb`` for the
    validation that drove this design.
    """
    arm_arr = np.asarray(arm)

    # Preferred path: the model exposed diff_log_alpha as a deterministic.
    # Stored on the natural-log scale by NumPyro; convert to log10 for
    # consistency with the docstring and analysis plan.
    if "diff_log_alpha" in idata.posterior:  # type: ignore[attr-defined]
        diffs_natural = _stack_samples(idata, "diff_log_alpha")  # (n_draws,)
        diffs = diffs_natural / np.log(10.0)  # convert ln to log10
        n_draws = diffs.shape[0]
        n_low = int((arm_arr == "low").sum())
        n_high = int((arm_arr == "high").sum())
        bounds = np.asarray(az.hdi(diffs, hdi_prob=0.95)).flatten()  # type: ignore[no-untyped-call]
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
    bounds = np.asarray(az.hdi(diffs, hdi_prob=0.95)).flatten()  # type: ignore[no-untyped-call]
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

    Shapes: ``replicates`` ``(n_draws, n_subjects, n_points)``,
    ``observed`` ``(n_subjects, n_points)``, ``grid`` ``(n_points,)``.
    """

    replicates: FloatArray
    observed: FloatArray
    grid: FloatArray


def _draw_subject_params(
    idata: az.InferenceData,
    extra_var: str,
    *,
    n_draws: int | None,
    rng: np.random.Generator | None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, np.random.Generator, int]:
    """Pull (alpha, Q0, k_full, sigma) posterior arrays, optionally thinning to ``n_draws``.

    Returns the four arrays plus a Generator and the (post-thinning)
    draw count, ready for the broadcasted likelihood evaluation.
    """
    alpha = _stack_samples(idata, "alpha")
    Q0 = _stack_samples(idata, "Q0")
    k_full = np.broadcast_to(_stack_samples(idata, "k_shared")[:, None], alpha.shape)
    sigma = _stack_samples(idata, extra_var)

    rng = rng or np.random.default_rng(0)
    n_total = alpha.shape[0]
    if n_draws is not None and n_draws < n_total:
        idx = rng.choice(n_total, n_draws, replace=False)
        alpha, Q0, k_full, sigma = alpha[idx], Q0[idx], k_full[idx], sigma[idx]
        n_total = n_draws
    return alpha, Q0, np.asarray(k_full), sigma, rng, n_total


def posterior_predictive_purchase(
    idata: az.InferenceData,
    P: FloatArray,
    Q_obs: FloatArray,
    *,
    n_draws: int | None = None,
    rng: np.random.Generator | None = None,
) -> PosteriorPredictive:
    """Replicate purchase-task Q values from the posterior (log10-Gaussian noise)."""
    alpha, Q0, k_full, sigma, rng, D = _draw_subject_params(
        idata, "sigma_purchase", n_draws=n_draws, rng=rng
    )
    n_subj = alpha.shape[1]
    n_p = P.shape[0]

    Q0_b = Q0[:, :, None]
    alpha_b = alpha[:, :, None]
    k_b = k_full[:, :, None]
    P_b = P[None, None, :]
    Q_pred = Q0_b * np.power(10.0, k_b * (np.exp(-alpha_b * Q0_b * P_b) - 1.0))
    log_q_pred = np.log10(np.maximum(Q_pred, 1e-12))
    noise = rng.normal(0.0, sigma[:, None, None], size=(D, n_subj, n_p))
    replicates = np.power(10.0, log_q_pred + noise)
    return PosteriorPredictive(replicates=replicates, observed=Q_obs, grid=P)


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
    """Replicate SV values from the unified-model posterior (linear Gaussian noise, clipped at 0)."""
    alpha, Q0, k_full, sigma, rng, D = _draw_subject_params(
        idata, "sigma_sv", n_draws=n_draws, rng=rng
    )
    n_subj, n_e = alpha.shape[1], E.shape[1]

    Q0_b = Q0[:, :, None]
    alpha_b = alpha[:, :, None]
    k_b = k_full[:, :, None]
    B_b = B_anchor[None, :, None]
    E_b = E[None, :, :]
    demand_term = np.power(10.0, k_b * (np.exp(-alpha_b * Q0_b * E_b / A) - 1.0))
    capability_term = np.where(E_b > 0, B_b / (Q0_b * E_b), np.inf)
    sv_pred = A * np.minimum(demand_term, capability_term)
    noise = rng.normal(0.0, sigma[:, None, None], size=(D, n_subj, n_e))
    replicates = np.maximum(sv_pred + noise, 0.0)
    return PosteriorPredictive(
        replicates=replicates,
        observed=SV_obs,
        grid=np.asarray(E[0], dtype=float),
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
