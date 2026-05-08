"""Identifiability checks via profile likelihoods.

For each parameter pair, fix on a 2D grid and reoptimise the rest;
pathological identifiability shows as a flat ridge.

Implemented pairs:

- ``(alpha, k)`` — flagged by the recovery sweep (57% RMSE on alpha
  with free k).
- ``(alpha, B)`` — both scale SV in the unified model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from edu.models.demand import Koffarnus
from edu.models.unified import UnifiedCapabilityBounded
from edu.simulation.generate import (
    SubjectParams,
    simulate_effort_discounting,
    simulate_purchase_task,
)

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class ProfileResult:
    """2D profile-likelihood surface."""

    p1_name: str
    p2_name: str
    p1_grid: FloatArray
    p2_grid: FloatArray
    nll: FloatArray  # shape (len(p1_grid), len(p2_grid)); nan where fit failed
    truth: tuple[float, float]


# ---------------------------------------------------------------------------
# alpha-k profile (single-task, Koffarnus)
# ---------------------------------------------------------------------------


def profile_alpha_k_demand(
    subject: SubjectParams,
    *,
    rng: np.random.Generator,
    alpha_grid: FloatArray | None = None,
    k_grid: FloatArray | None = None,
    log_noise_sd: float = 0.10,
) -> ProfileResult:
    """Profile RSS over (alpha, k) for one subject's purchase-task data.

    Uses RSS as a proxy for NLL (Gaussian, profiled variance); the
    ridge-vs-no-ridge structure is preserved either way.
    """
    P, Q = simulate_purchase_task(subject, rng=rng, log_noise_sd=log_noise_sd)
    ko = Koffarnus()

    if alpha_grid is None:
        alpha_grid = np.geomspace(subject.alpha / 8.0, subject.alpha * 8.0, 25)
    if k_grid is None:
        k_grid = np.linspace(max(subject.k - 2.0, 0.5), subject.k + 2.0, 25)

    nll = np.full((len(alpha_grid), len(k_grid)), np.nan)
    for i, alpha in enumerate(alpha_grid):
        for j, k in enumerate(k_grid):
            try:
                fit = ko.fit(P, Q, fixed={"alpha": float(alpha), "k": float(k)})
                nll[i, j] = float(fit.rss)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue

    return ProfileResult(
        p1_name="alpha",
        p2_name="k",
        p1_grid=np.asarray(alpha_grid, dtype=float),
        p2_grid=np.asarray(k_grid, dtype=float),
        nll=nll,
        truth=(subject.alpha, subject.k),
    )


# ---------------------------------------------------------------------------
# alpha-B profile (unified joint task)
# ---------------------------------------------------------------------------


def profile_alpha_B_unified(
    subject: SubjectParams,
    *,
    rng: np.random.Generator,
    A: float = 10.0,
    alpha_grid: FloatArray | None = None,
    B_grid: FloatArray | None = None,
    log_noise_sd: float = 0.10,
    sv_noise_sd: float = 0.5,
) -> ProfileResult:
    """Profile NLL over (alpha, B) for the unified joint-task fit.

    Fixes ``alpha, B`` and refits ``Q0, k`` against the joint
    purchase + effort-discount data per grid point.
    """
    purchase = simulate_purchase_task(subject, rng=rng, log_noise_sd=log_noise_sd)
    effort = simulate_effort_discounting(subject, rng=rng, A=A, sv_noise_sd=sv_noise_sd)
    P_obs, Q_obs = purchase
    E_obs, SV_obs = effort

    m = UnifiedCapabilityBounded()
    ko = Koffarnus()

    if alpha_grid is None:
        alpha_grid = np.geomspace(subject.alpha / 4.0, subject.alpha * 4.0, 25)
    if B_grid is None:
        B_grid = np.geomspace(subject.B / 4.0, subject.B * 4.0, 25)

    nll = np.full((len(alpha_grid), len(B_grid)), np.nan)
    for i, alpha in enumerate(alpha_grid):
        for j, B in enumerate(B_grid):
            x0 = np.array([subject.Q0, subject.k], dtype=float)
            lb = np.array([1e-6, 0.5])
            ub = np.array([1e4, 6.0])

            def residuals(
                x: FloatArray, alpha: float = float(alpha), B: float = float(B)
            ) -> FloatArray:
                Q0, k = x
                Q_pred = ko.value({"Q0": Q0, "alpha": alpha, "k": k}, P_obs)
                with np.errstate(divide="ignore", invalid="ignore"):
                    r_demand = np.where(
                        (Q_pred > 0) & (Q_obs > 0),
                        np.log10(np.maximum(Q_pred, 1e-12)) - np.log10(np.maximum(Q_obs, 1e-12)),
                        0.0,
                    )
                sv_pred = m.value({"A": A, "Q0": Q0, "alpha": alpha, "k": k, "B": B}, E_obs)
                r_sv = sv_pred - SV_obs
                return np.concatenate([r_demand, r_sv])

            try:
                result = least_squares(residuals, x0, bounds=(lb, ub), method="trf", max_nfev=200)
                if result.success:
                    nll[i, j] = float(np.sum(residuals(result.x) ** 2))
            except (ValueError, RuntimeError):
                continue

    return ProfileResult(
        p1_name="alpha",
        p2_name="B",
        p1_grid=np.asarray(alpha_grid, dtype=float),
        p2_grid=np.asarray(B_grid, dtype=float),
        nll=nll,
        truth=(subject.alpha, subject.B),
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def ridge_score(profile: ProfileResult, *, threshold_pct: float = 1.0) -> float:
    """Fraction of grid points within ``threshold_pct`` of the minimum NLL.

    Small for well-identified pairs (minimum at one cell); near 1 for
    pathological ridges that fill the grid.
    """
    valid = np.isfinite(profile.nll)
    if not valid.any():
        return float("nan")
    nll_min = float(np.nanmin(profile.nll))
    threshold = nll_min * (1.0 + threshold_pct / 100.0) if nll_min > 0 else nll_min + threshold_pct
    near_min = (profile.nll <= threshold) & valid
    return float(near_min.sum() / valid.sum())


__all__ = [
    "ProfileResult",
    "profile_alpha_B_unified",
    "profile_alpha_k_demand",
    "ridge_score",
]
