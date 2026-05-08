"""Cross-task linkage simulation (CLAUDE.md H1).

Tests whether shared-alpha beats independent-alpha by BIC when data are
generated from a single per-subject alpha. The shared model has one
fewer parameter, so it should win by ``~log(n_obs)`` per subject.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from edu.models.demand import Koffarnus
from edu.simulation.generate import (
    SubjectParams,
    simulate_effort_purchase_task,
    simulate_purchase_task,
)

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class LinkageResult:
    """Population summary of the shared-vs-independent ΔBIC test.

    ``delta_bic`` is per-subject ``BIC_indep - BIC_shared``. Positive
    favours shared (the unification claim).
    """

    delta_bic: FloatArray
    pct_favoring_shared: float
    median_delta: float


def _bic(rss: float, n: int, k: int) -> float:
    if rss <= 0 or n <= 0:
        return float("nan")
    ll = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)
    return float(k * np.log(n) - 2 * ll)


def _fit_shared_alpha(
    P1: FloatArray,
    Q1: FloatArray,
    P3: FloatArray,
    Q3: FloatArray,
    *,
    fix_k: float,
) -> tuple[float, int, dict[str, float]] | None:
    """Fit shared ``alpha`` plus per-task ``Q0`` (k held fixed). Returns ``(rss, n_obs, params)``."""
    ko = Koffarnus()

    def residuals(x: FloatArray) -> FloatArray:
        Q0_1, Q0_3, alpha = x
        # purchase task in log10
        Q1_pred = ko.value({"Q0": Q0_1, "alpha": alpha, "k": fix_k}, P1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r1 = np.where(
                (Q1_pred > 0) & (Q1 > 0),
                np.log10(np.maximum(Q1_pred, 1e-12)) - np.log10(np.maximum(Q1, 1e-12)),
                0.0,
            )
        # effort-purchase task likewise
        Q3_pred = ko.value({"Q0": Q0_3, "alpha": alpha, "k": fix_k}, P3)
        with np.errstate(divide="ignore", invalid="ignore"):
            r3 = np.where(
                (Q3_pred > 0) & (Q3 > 0),
                np.log10(np.maximum(Q3_pred, 1e-12)) - np.log10(np.maximum(Q3, 1e-12)),
                0.0,
            )
        return np.concatenate([r1, r3])

    Q0_init = float(np.max(np.concatenate([Q1, Q3])))
    x0 = np.array([Q0_init, Q0_init, 0.01])
    lb = np.array([1e-6, 1e-6, 1e-9])
    ub = np.array([1e4, 1e4, 5.0])
    try:
        result = least_squares(residuals, x0, bounds=(lb, ub), method="trf", max_nfev=200)
    except (ValueError, RuntimeError):
        return None
    if not result.success:
        return None
    rss = float(np.sum(residuals(result.x) ** 2))
    n = P1.size + P3.size
    return (
        rss,
        n,
        {
            "Q0_1": float(result.x[0]),
            "Q0_3": float(result.x[1]),
            "alpha": float(result.x[2]),
        },
    )


def _fit_independent_alpha(
    P1: FloatArray,
    Q1: FloatArray,
    P3: FloatArray,
    Q3: FloatArray,
    *,
    fix_k: float,
) -> tuple[float, int, dict[str, float]] | None:
    """Fit two ``alpha`` parameters, one per task. Same Q0 freedom."""
    ko = Koffarnus()
    try:
        f1 = ko.fit(P1, Q1, fixed={"k": fix_k})
        f3 = ko.fit(P3, Q3, fixed={"k": fix_k})
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None
    if not (f1.success and f3.success):
        return None
    # Sum residuals from both fits (they are on log10 scale by default).
    return (
        float(f1.rss + f3.rss),
        P1.size + P3.size,
        {
            "Q0_1": f1.params["Q0"],
            "Q0_3": f3.params["Q0"],
            "alpha_1": f1.params["alpha"],
            "alpha_3": f3.params["alpha"],
        },
    )


def linkage_test(
    subjects: list[SubjectParams],
    *,
    rng: np.random.Generator,
    log_noise_sd: float = 0.10,
    fix_k: float | None = None,
) -> LinkageResult:
    """Per-population shared-vs-independent ΔBIC test on Tasks 1 and 3."""
    if fix_k is None:
        fix_k = float(np.median([s.k for s in subjects]))

    deltas: list[float] = []
    for s in subjects:
        P1, Q1 = simulate_purchase_task(s, rng=rng, log_noise_sd=log_noise_sd)
        P3, Q3 = simulate_effort_purchase_task(s, rng=rng, log_noise_sd=log_noise_sd)
        shared = _fit_shared_alpha(P1, Q1, P3, Q3, fix_k=fix_k)
        indep = _fit_independent_alpha(P1, Q1, P3, Q3, fix_k=fix_k)
        if shared is None or indep is None:
            continue
        bic_shared = _bic(shared[0], shared[1], k=3)  # Q0_1, Q0_3, alpha
        bic_indep = _bic(indep[0], indep[1], k=4)  # Q0_1, Q0_3, alpha_1, alpha_3
        deltas.append(bic_indep - bic_shared)

    arr = np.array(deltas, dtype=float)
    return LinkageResult(
        delta_bic=arr,
        pct_favoring_shared=float((arr > 0).mean()) if arr.size else float("nan"),
        median_delta=float(np.median(arr)) if arr.size else float("nan"),
    )


__all__ = ["LinkageResult", "linkage_test"]
