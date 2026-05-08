"""Per-subject summaries from a hierarchical Bayesian fit.

The hierarchical fit returns posterior samples for ``alpha``, ``Q0``,
``k_subj`` (or ``k_shared``) per subject. This module collapses those to
the summary statistics secondary analyses need:

- Posterior mean and 95% HDI for ``alpha`` per subject.
- The model-implied effort-discount steepness ``lambda`` per subject,
  computed from the per-subject demand parameters via the linearised
  unconstrained-regime approximation.
- Per-subject crossover effort ``E*`` from the unified-model
  capability bound.

These outputs feed directly into :mod:`edu.analysis.group` for H1-
stronger correlation and H3 contrast tests.
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

    Attributes
    ----------
    alpha_mean, alpha_hdi
        Posterior mean and 95% HDI for ``alpha``.
    Q0_mean, Q0_hdi
        Posterior mean and 95% HDI for ``Q0``.
    k_value
        ``k`` value used by the fit. With ``share_k=True`` (the
        pre-registered default), this is one population-level value
        repeated; with ``share_k=False``, it is the per-subject posterior
        mean of ``k_subj``.
    lambda_unconstrained_mean
        Model-implied unconstrained-regime steepness
        ``alpha * Q0 * k * ln(10) / A``. Posterior mean.
    crossover_mean
        Posterior mean of the unified-model crossover effort ``E*``,
        computed by evaluating the closed-form crossover at each
        posterior draw and averaging. ``nan`` if the capability bound
        never activates for a given draw and that's the modal case.
    n_draws
        Number of posterior draws used in the summary.
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
    da = idata.posterior[var]  # type: ignore[attr-defined,unused-ignore]
    n_chains = da.sizes["chain"]
    n_draws = da.sizes["draw"]
    return np.asarray(da.values.reshape(n_chains * n_draws, *da.shape[2:]), dtype=float)


def _hdi(samples: FloatArray, prob: float = 0.95) -> tuple[float, float]:
    """Highest density interval for 1D samples via arviz."""
    arr = np.asarray(samples, dtype=float)
    bounds = az.hdi(arr, hdi_prob=prob)  # type: ignore[no-untyped-call,unused-ignore]
    # arviz returns either (2,) or a 1D ndarray-like; coerce both ways.
    bounds_arr = np.asarray(bounds).flatten()
    return float(bounds_arr[0]), float(bounds_arr[1])


def summarise_subjects(
    idata: az.InferenceData,
    *,
    A: float = 10.0,
    B_anchor: FloatArray | None = None,
) -> list[SubjectSummary]:
    """Collapse the hierarchical posterior into a list of per-subject summaries.

    Parameters
    ----------
    idata
        Output of :func:`edu.fitting.bayesian.fit_unified_hierarchical`.
    A
        Reward magnitude used in the fit (matches the design constant).
    B_anchor
        Per-subject capability values, shape ``(n_subjects,)``. Required
        to compute the crossover ``E*``. If ``None``, ``crossover_mean``
        is set to ``nan`` for every subject.
    """
    alpha_samples = _stack_samples(idata, "alpha")  # (n_draws, n_subj)
    Q0_samples = _stack_samples(idata, "Q0")
    if "k_shared" in idata.posterior:  # type: ignore[attr-defined,unused-ignore]
        k_samples = _stack_samples(idata, "k_shared")  # (n_draws,)
        k_per_subj = np.broadcast_to(k_samples[:, None], alpha_samples.shape)
    else:
        k_per_subj = _stack_samples(idata, "k_subj")  # (n_draws, n_subj)

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
