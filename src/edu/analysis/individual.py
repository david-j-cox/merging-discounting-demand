"""Per-subject summaries from a hierarchical Bayesian fit.

Collapses ``alpha``, ``Q0``, ``k`` posterior samples into per-subject
mean/HDI, the model-implied unconstrained-regime steepness
``lambda = alpha * Q0 * k * ln 10 / A``, and the unified-model
crossover effort ``E*``. Feeds :mod:`edu.analysis.group`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np
from numpy.typing import NDArray

from edu.models.unified import UnifiedCapabilityBounded

FloatArray = NDArray[np.floating[Any]]
_LN10 = float(np.log(10.0))


@dataclass(frozen=True)
class SubjectSummary:
    """Per-subject collapse of a hierarchical posterior.

    ``k_value`` is the population-level ``k`` (shared across subjects).
    ``crossover_mean`` is averaged over draws and is ``nan`` when the
    capability bound never activates.
    """

    alpha_mean: float
    alpha_hdi: tuple[float, float]
    Q0_mean: float
    Q0_hdi: tuple[float, float]
    k_value: float
    lambda_unconstrained_mean: float
    crossover_mean: float
    n_draws: int


def _stack_samples(idata: az.InferenceData, var: str) -> FloatArray:
    """Flatten chain x draw into a single sample dim. Returns shape (n_draws, ...)."""
    da = idata.posterior[var]  # type: ignore[attr-defined]
    n_chains = da.sizes["chain"]
    n_draws = da.sizes["draw"]
    return np.asarray(da.values.reshape(n_chains * n_draws, *da.shape[2:]), dtype=float)


def _hdi(samples: FloatArray, prob: float = 0.95) -> tuple[float, float]:
    """Highest density interval for 1D samples via arviz."""
    arr = np.asarray(samples, dtype=float)
    bounds = az.hdi(arr, hdi_prob=prob)  # type: ignore[no-untyped-call]
    # arviz returns either (2,) or a 1D ndarray-like; coerce both ways.
    bounds_arr = np.asarray(bounds).flatten()
    return float(bounds_arr[0]), float(bounds_arr[1])


def summarise_subjects(
    idata: az.InferenceData,
    *,
    A: float = 10.0,
    B_anchor: FloatArray | None = None,
) -> list[SubjectSummary]:
    """One ``SubjectSummary`` per subject from a hierarchical posterior.

    ``B_anchor`` (per-subject capability) is required to compute the
    crossover ``E*``; without it ``crossover_mean`` is ``nan``.
    """
    alpha_samples = _stack_samples(idata, "alpha")  # (n_draws, n_subj)
    Q0_samples = _stack_samples(idata, "Q0")
    k_samples = _stack_samples(idata, "k_shared")  # (n_draws,)
    k_per_subj = np.broadcast_to(k_samples[:, None], alpha_samples.shape)

    n_draws, n_subj = alpha_samples.shape
    summaries: list[SubjectSummary] = []
    model = UnifiedCapabilityBounded()

    for i in range(n_subj):
        alpha_i = alpha_samples[:, i]
        Q0_i = Q0_samples[:, i]
        k_i = k_per_subj[:, i]
        lambda_i = alpha_i * Q0_i * k_i * _LN10 / A

        cross_mean = float("nan")
        if B_anchor is not None:
            B_i = float(B_anchor[i])
            cross_vals = np.empty(n_draws, dtype=float)
            for d in range(n_draws):
                cross_vals[d] = model.crossover_effort(
                    {
                        "A": A,
                        "Q0": float(Q0_i[d]),
                        "alpha": float(alpha_i[d]),
                        "k": float(k_i[d]),
                        "B": B_i,
                    }
                )
            cross_mean = float(np.nanmean(cross_vals))

        summaries.append(
            SubjectSummary(
                alpha_mean=float(np.mean(alpha_i)),
                alpha_hdi=_hdi(alpha_i),
                Q0_mean=float(np.mean(Q0_i)),
                Q0_hdi=_hdi(Q0_i),
                k_value=float(np.mean(k_i)),
                lambda_unconstrained_mean=float(np.mean(lambda_i)),
                crossover_mean=cross_mean,
                n_draws=n_draws,
            )
        )

    return summaries


__all__ = [
    "SubjectSummary",
    "summarise_subjects",
]
