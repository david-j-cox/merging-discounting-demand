"""Parameter-recovery experiments for the unified model.

For each requested sample size, simulate ``n`` subjects from a known
population, fit them back, and report bias / RMSE / 95% CI width per
parameter. CLAUDE.md §4 Phase 3 step 1 specifies n ∈ {30, 60, 120, 240}
and a target of <5% bias plus interpretable CI widths at the planned
sample size.

Two recovery designs:

* **Single-task** — fit the Koffarnus demand form to each subject's
  purchase-task data, recovering ``(Q0, alpha, k)``. Establishes a
  baseline; if the basic demand model can't be recovered there is no
  hope for the unified model.
* **Joint-task (unified)** — for each subject, fit ``UnifiedCapabilityBounded``
  to their *combined* purchase-task and effort-discount data, recovering
  ``(Q0, alpha, k, B)`` jointly. Tests CLAUDE.md H1's parameter-linkage
  prediction in the synthetic-data limit.

Both designs anchor ``B`` to its true value in the joint-task case to
mimic the planned experimental practice of fixing ``B`` from measured
MVC / N-back (CLAUDE.md §9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from edu.models.demand import Koffarnus
from edu.models.unified import UnifiedCapabilityBounded
from edu.simulation.generate import (
    SubjectParams,
    sample_population,
    simulate_effort_discounting,
    simulate_purchase_task,
)

FloatArray = NDArray[np.floating[Any]]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterRecovery:
    """Recovery summary for one parameter across a population of fits.

    Attributes
    ----------
    name : str
        Parameter name.
    true : ndarray
        True values, one per subject.
    fit : ndarray
        Fitted values, one per subject. ``nan`` for failed fits.
    bias_rel : float
        Median relative bias ``(fit - true) / true`` across successful fits.
    rmse_rel : float
        Root mean squared relative error across successful fits.
    success_rate : float
        Fraction of fits that converged.
    """

    name: str
    true: FloatArray
    fit: FloatArray
    bias_rel: float
    rmse_rel: float
    success_rate: float


@dataclass(frozen=True)
class RecoveryReport:
    """Per-sample-size recovery report (one ``RecoveryReport`` per ``n``)."""

    n: int
    per_param: dict[str, ParameterRecovery]
    extra: dict[str, Any] = field(default_factory=dict)

    def summary_row(self) -> str:
        lines = [f"n = {self.n}"]
        for name, p in self.per_param.items():
            lines.append(
                f"  {name:6s}  bias={p.bias_rel:+.2%}  rmse={p.rmse_rel:.2%}  "
                f"success={p.success_rate:.0%}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single-task recovery (Koffarnus demand)
# ---------------------------------------------------------------------------


def _summarise(name: str, true: FloatArray, fit: FloatArray) -> ParameterRecovery:
    ok = np.isfinite(fit)
    if not ok.any():
        return ParameterRecovery(
            name=name,
            true=true,
            fit=fit,
            bias_rel=float("nan"),
            rmse_rel=float("nan"),
            success_rate=0.0,
        )
    rel = (fit[ok] - true[ok]) / true[ok]
    return ParameterRecovery(
        name=name,
        true=true,
        fit=fit,
        bias_rel=float(np.median(rel)),
        rmse_rel=float(np.sqrt(np.mean(rel**2))),
        success_rate=float(ok.mean()),
    )


def recover_demand(
    subjects: list[SubjectParams],
    *,
    rng: np.random.Generator,
    log_noise_sd: float = 0.10,
    fix_k: float | None = None,
) -> RecoveryReport:
    """Fit Koffarnus to each subject's purchase-task data; report recovery.

    Parameters
    ----------
    fix_k
        If given, hold ``k`` constant at this value (Hursh's standard
        practice — see ``docs/derivation.md`` and the simulation suite
        finding that free ``k`` produces an α-k identifiability ridge).
    """
    ko = Koffarnus()
    n = len(subjects)
    true = {name: np.array([getattr(s, name) for s in subjects]) for name in ("Q0", "alpha", "k")}
    fit_arr = {name: np.full(n, np.nan) for name in ("Q0", "alpha", "k")}

    fixed = {"k": fix_k} if fix_k is not None else None

    for i, s in enumerate(subjects):
        prices, q = simulate_purchase_task(s, rng=rng, log_noise_sd=log_noise_sd)
        try:
            result = ko.fit(prices, q, fixed=fixed)
            if result.success:
                for name in fit_arr:
                    fit_arr[name][i] = result.params[name]
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue

    per_param = {name: _summarise(name, true[name], fit_arr[name]) for name in fit_arr}
    return RecoveryReport(n=n, per_param=per_param, extra={"fix_k": fix_k})


# ---------------------------------------------------------------------------
# Joint-task recovery (unified model)
# ---------------------------------------------------------------------------


def _fit_unified_joint(
    subject: SubjectParams,
    purchase_data: tuple[FloatArray, FloatArray],
    effort_data: tuple[FloatArray, FloatArray],
    *,
    A: float = 10.0,
    fix_B: bool = True,
) -> dict[str, float] | None:
    """Fit UnifiedCapabilityBounded jointly to one subject's two task datasets.

    Residuals are computed in two scales (log10 for purchase task, linear
    for SV) and concatenated. ``B`` is held fixed at the true value when
    ``fix_B`` is true, mirroring the experimental practice of anchoring B
    to measured capability.
    """
    P_obs, Q_obs = purchase_data
    E_obs, SV_obs = effort_data
    m = UnifiedCapabilityBounded()
    ko = Koffarnus()

    # Free parameters: Q0, alpha, k. (B fixed; A fixed at design value.)
    # Initial guess from the data magnitudes.
    Q0_init = float(np.max(Q_obs[Q_obs > 0])) if (Q_obs > 0).any() else 1.0
    x0 = np.array([Q0_init, 0.01, 3.0], dtype=float)
    lb = np.array([1e-6, 1e-9, 0.5])
    ub = np.array([1e4, 5.0, 6.0])

    def residuals(x: FloatArray) -> FloatArray:
        Q0, alpha, k = x
        # purchase-task residuals in log10
        Q_pred = ko.value({"Q0": Q0, "alpha": alpha, "k": k}, P_obs)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_demand = np.where(
                (Q_pred > 0) & (Q_obs > 0),
                np.log10(np.maximum(Q_pred, 1e-12)) - np.log10(np.maximum(Q_obs, 1e-12)),
                0.0,
            )
        # SV residuals linear
        sv_pred = m.value({"A": A, "Q0": Q0, "alpha": alpha, "k": k, "B": subject.B}, E_obs)
        r_sv = sv_pred - SV_obs
        return np.concatenate([r_demand, r_sv])

    if not fix_B:
        msg = "Joint fit with free B is not yet implemented; pass fix_B=True."
        raise NotImplementedError(msg)

    try:
        result = least_squares(residuals, x0, bounds=(lb, ub), method="trf", max_nfev=200)
    except (ValueError, RuntimeError):
        return None
    if not result.success:
        return None
    return {"Q0": float(result.x[0]), "alpha": float(result.x[1]), "k": float(result.x[2])}


def recover_unified_joint(
    subjects: list[SubjectParams],
    *,
    rng: np.random.Generator,
    log_noise_sd: float = 0.10,
    sv_noise_sd: float = 0.5,
    A: float = 10.0,
) -> RecoveryReport:
    """Fit the unified model jointly to purchase + effort-discount data per subject."""
    n = len(subjects)
    names = ("Q0", "alpha", "k")
    true = {name: np.array([getattr(s, name) for s in subjects]) for name in names}
    fit_arr = {name: np.full(n, np.nan) for name in names}

    for i, s in enumerate(subjects):
        purchase = simulate_purchase_task(s, rng=rng, log_noise_sd=log_noise_sd)
        effort = simulate_effort_discounting(s, rng=rng, A=A, sv_noise_sd=sv_noise_sd)
        result = _fit_unified_joint(s, purchase, effort, A=A, fix_B=True)
        if result is not None:
            for name in names:
                fit_arr[name][i] = result[name]

    per_param = {name: _summarise(name, true[name], fit_arr[name]) for name in names}
    return RecoveryReport(n=n, per_param=per_param)


# ---------------------------------------------------------------------------
# Sample-size sweep
# ---------------------------------------------------------------------------


def sample_size_sweep(
    sample_sizes: list[int],
    *,
    seed: int = 2025,
    mode: str = "joint",
    log_noise_sd: float = 0.10,
    sv_noise_sd: float = 0.5,
    fix_k: float | None = None,
) -> list[RecoveryReport]:
    """Run a recovery experiment at each requested sample size.

    Parameters
    ----------
    mode
        One of:

        * ``"single_free_k"`` — Koffarnus to purchase task only, all params free.
        * ``"single_fixed_k"`` — Koffarnus to purchase task only, ``k`` fixed
          at ``fix_k`` (population median by default).
        * ``"joint"`` — UnifiedCapabilityBounded jointly to purchase + effort
          discounting, ``B`` fixed at the true subject-specific value.

    Each run uses an independent RNG seeded from ``seed`` plus the size.
    """
    valid = {"single_free_k", "single_fixed_k", "joint"}
    if mode not in valid:
        msg = f"mode must be one of {valid}, got {mode!r}"
        raise ValueError(msg)

    reports: list[RecoveryReport] = []
    for n in sample_sizes:
        rng = np.random.default_rng(seed + n)
        subjects = sample_population(n, rng=rng)
        if mode == "joint":
            report = recover_unified_joint(
                subjects, rng=rng, log_noise_sd=log_noise_sd, sv_noise_sd=sv_noise_sd
            )
        elif mode == "single_free_k":
            report = recover_demand(subjects, rng=rng, log_noise_sd=log_noise_sd)
        else:  # single_fixed_k
            k_used = fix_k if fix_k is not None else float(np.median([s.k for s in subjects]))
            report = recover_demand(subjects, rng=rng, log_noise_sd=log_noise_sd, fix_k=k_used)
        reports.append(report)
    return reports


__all__ = [
    "ParameterRecovery",
    "RecoveryReport",
    "recover_demand",
    "recover_unified_joint",
    "sample_size_sweep",
]
