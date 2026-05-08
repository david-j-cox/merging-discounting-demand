"""Parameter-recovery experiments for the unified model.

CLAUDE.md §4 Phase 3 step 1: at each ``n`` ∈ {30, 60, 120, 240},
simulate, fit, report bias / RMSE / success rate per parameter.

- :func:`recover_demand` — Koffarnus on the purchase task alone.
- :func:`recover_unified_joint` — UnifiedCapabilityBounded on the joint
  purchase + effort-discount data with ``B`` anchored to the true
  value (mirrors the experimental practice of fixing B from MVC).
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

    ``bias_rel`` and ``rmse_rel`` are ``(fit - true) / true`` summaries
    over successful fits; failed fits are ``nan`` in ``fit``.
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
    fix_k: float | None = None,
) -> dict[str, float] | None:
    """Fit UnifiedCapabilityBounded jointly to one subject's purchase + effort data.

    Residuals are concatenated across the two scales (log10 for Q,
    linear for SV). ``fix_B`` anchors ``B`` to the subject's true
    value; ``fix_k`` anchors ``k`` (the pre-registered NLS commitment).
    """
    P_obs, Q_obs = purchase_data
    E_obs, SV_obs = effort_data
    m = UnifiedCapabilityBounded()
    ko = Koffarnus()

    if not fix_B:
        msg = "Joint fit with free B is not yet implemented; pass fix_B=True."
        raise NotImplementedError(msg)

    Q0_init = float(np.max(Q_obs[Q_obs > 0])) if (Q_obs > 0).any() else 1.0
    if fix_k is not None:
        # Two parameters free: Q0, alpha. k is held constant.
        x0 = np.array([Q0_init, 0.01], dtype=float)
        lb = np.array([1e-6, 1e-9])
        ub = np.array([1e4, 5.0])

        def residuals(x: FloatArray) -> FloatArray:
            Q0, alpha = x
            k = float(fix_k)
            Q_pred = ko.value({"Q0": Q0, "alpha": alpha, "k": k}, P_obs)
            with np.errstate(divide="ignore", invalid="ignore"):
                r_demand = np.where(
                    (Q_pred > 0) & (Q_obs > 0),
                    np.log10(np.maximum(Q_pred, 1e-12)) - np.log10(np.maximum(Q_obs, 1e-12)),
                    0.0,
                )
            sv_pred = m.value({"A": A, "Q0": Q0, "alpha": alpha, "k": k, "B": subject.B}, E_obs)
            return np.concatenate([r_demand, sv_pred - SV_obs])

        try:
            result = least_squares(residuals, x0, bounds=(lb, ub), method="trf", max_nfev=200)
        except (ValueError, RuntimeError):
            return None
        if not result.success:
            return None
        return {
            "Q0": float(result.x[0]),
            "alpha": float(result.x[1]),
            "k": float(fix_k),
        }

    # Per-subject k (pre-2026-01 default; kept for sensitivity comparison).
    x0 = np.array([Q0_init, 0.01, 3.0], dtype=float)
    lb = np.array([1e-6, 1e-9, 0.5])
    ub = np.array([1e4, 5.0, 6.0])

    def residuals_free_k(x: FloatArray) -> FloatArray:
        Q0, alpha, k = x
        Q_pred = ko.value({"Q0": Q0, "alpha": alpha, "k": k}, P_obs)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_demand = np.where(
                (Q_pred > 0) & (Q_obs > 0),
                np.log10(np.maximum(Q_pred, 1e-12)) - np.log10(np.maximum(Q_obs, 1e-12)),
                0.0,
            )
        sv_pred = m.value({"A": A, "Q0": Q0, "alpha": alpha, "k": k, "B": subject.B}, E_obs)
        return np.concatenate([r_demand, sv_pred - SV_obs])

    try:
        result = least_squares(residuals_free_k, x0, bounds=(lb, ub), method="trf", max_nfev=200)
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
    b_noise_pct: float = 0.0,
    share_k: bool = True,
) -> RecoveryReport:
    """Joint per-subject unified-model fit across a population.

    ``b_noise_pct`` injects log-normal noise on the ``B`` value seen by
    the fitter (data still generated from true ``B``) — quantifies
    sensitivity to MVC / N-back measurement error.

    ``share_k=True`` (the default, pre-registered) anchors ``k`` at the
    population median; ``share_k=False`` fits ``k`` per subject as a
    sensitivity check.
    """
    n = len(subjects)
    names = ("Q0", "alpha", "k")
    true = {name: np.array([getattr(s, name) for s in subjects]) for name in names}
    fit_arr = {name: np.full(n, np.nan) for name in names}

    log_noise = b_noise_pct / 100.0
    fix_k_value: float | None = float(np.median([s.k for s in subjects])) if share_k else None

    for i, s in enumerate(subjects):
        purchase = simulate_purchase_task(s, rng=rng, log_noise_sd=log_noise_sd)
        effort = simulate_effort_discounting(s, rng=rng, A=A, sv_noise_sd=sv_noise_sd)
        if log_noise > 0:
            B_used = float(s.B * np.exp(rng.normal(0.0, log_noise)))
            s_for_fit = SubjectParams(alpha=s.alpha, Q0=s.Q0, k=s.k, B=B_used)
        else:
            s_for_fit = s
        result = _fit_unified_joint(s_for_fit, purchase, effort, A=A, fix_B=True, fix_k=fix_k_value)
        if result is not None:
            for name in names:
                fit_arr[name][i] = result[name]

    per_param = {name: _summarise(name, true[name], fit_arr[name]) for name in names}
    return RecoveryReport(
        n=n,
        per_param=per_param,
        extra={"b_noise_pct": b_noise_pct, "share_k": share_k, "fix_k_value": fix_k_value},
    )


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
    b_noise_pct: float = 0.0,
) -> list[RecoveryReport]:
    """Run recovery at each ``n``. ``mode`` selects the fitting strategy.

    Modes:

    - ``single_free_k`` — Koffarnus on purchase task only, all params free.
    - ``single_fixed_k`` — Koffarnus on purchase task only, ``k`` fixed.
    - ``joint`` — UnifiedCapabilityBounded jointly on purchase + effort,
      ``B`` fixed at the true subject value.
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
                subjects,
                rng=rng,
                log_noise_sd=log_noise_sd,
                sv_noise_sd=sv_noise_sd,
                b_noise_pct=b_noise_pct,
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
