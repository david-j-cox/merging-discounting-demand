"""Behavioural-economic demand-curve models.

Two parameterisations are implemented:

:class:`HurshSilberberg`
    The original Hursh & Silberberg (2008) exponential model in log-Q space.
    Cannot represent zero consumption directly because ``log10(0) = -inf``.

:class:`Koffarnus`
    The Koffarnus, Franck, Stein & Bickel (2015) "exponentiated" form. Maps
    naturally onto Q in linear units and handles zero consumption. Default
    choice for new fits per CLAUDE.md §3.2.

Both share the parameters ``Q0``, ``alpha``, ``k`` and the closed-form
:meth:`Pmax`/:meth:`Omax` quantities derived in Gilroy, Kaplan, Reed, Hantula
& Hursh (2019).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.special import lambertw

from edu.models._base import FitResult, FloatArray, ParametricModel


@dataclass(frozen=True)
class DemandDerived:
    """Closed-form derived quantities from a fitted demand curve.

    Attributes
    ----------
    Pmax : float
        Price at unit elasticity. Where consumption transitions from inelastic
        to elastic. Computed via the Gilroy et al. (2019) Lambert-W solution.
    Omax : float
        Maximum total expenditure ``Pmax * Q(Pmax)``.
    essential_value : float
        Hursh's :math:`EV \\propto 1 / (\\alpha \\cdot k^{1.5})`. Reported in
        relative units (proportionality constant set to 1).
    """

    Pmax: float
    Omax: float
    essential_value: float


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_LN10 = float(np.log(10.0))


def _pmax_lambert(alpha: float, Q0: float, k: float) -> float:
    """Closed-form Pmax from Gilroy, Kaplan, Reed, Hantula & Hursh (2019).

    At unit elasticity ``alpha * Q0 * P * k * ln(10) * exp(-alpha * Q0 * P) = 1``.
    Letting ``u = alpha * Q0 * P`` rearranges to ``-u * exp(-u) = -1 / (k * ln 10)``,
    whose solution is ``u = -W_{-1}(-1 / (k * ln 10))``.

    The lower branch of the Lambert-W function is required; the principal
    branch ``W_0`` gives the trivial solution at the inelastic side of the
    curve.
    """
    arg = -1.0 / (k * _LN10)
    # W_{-1} is real for arg in [-1/e, 0); k * ln10 > 1/e is satisfied by any
    # k > 0.16, comfortably true for behavioural data where k is typically 2-4.
    if arg <= -1.0 / np.e:
        return float("nan")
    u = -lambertw(arg, k=-1).real
    return float(u / (alpha * Q0))


# ---------------------------------------------------------------------------
# Hursh-Silberberg
# ---------------------------------------------------------------------------


class HurshSilberberg(ParametricModel):
    """Hursh & Silberberg (2008) exponential demand.

    .. math::

        \\log_{10} Q = \\log_{10} Q_0 + k \\bigl(\\exp(-\\alpha Q_0 P) - 1\\bigr)

    Parameters
    ----------
    Q0 : float
        Demand at zero price. Must be positive.
    alpha : float
        Rate-of-change-in-elasticity parameter. Smaller values indicate more
        inelastic (essential) commodities. Must be positive.
    k : float
        Span of the consumption data in :math:`\\log_{10}` units. Often fixed
        across subjects in published work; here it is a free parameter
        unless held constant by passing a fixed value to :meth:`fit`.

    Notes
    -----
    Returns ``Q``, not ``log10(Q)``. Callers fitting in log-space should
    transform externally; the test suite covers both modes.
    """

    param_names: ClassVar[tuple[str, ...]] = ("Q0", "alpha", "k")

    def _value(self, params: FloatArray, x: FloatArray) -> FloatArray:
        Q0, alpha, k = params
        log10_q = np.log10(Q0) + k * (np.exp(-alpha * Q0 * x) - 1.0)
        return np.asarray(np.power(10.0, log10_q), dtype=float)

    def derivative(self, params: dict[str, float] | FloatArray, x: ArrayLike) -> FloatArray:
        """Analytic ``dQ/dP``.

        :math:`\\frac{dQ}{dP} = -Q \\cdot k \\ln 10 \\cdot \\alpha Q_0 \\exp(-\\alpha Q_0 P)`.
        """
        p = self._coerce_params(params)
        Q0, alpha, _k = p
        x_arr = np.asarray(x, dtype=float)
        Q = self._value(p, x_arr)
        return np.asarray(
            -Q * p[2] * _LN10 * alpha * Q0 * np.exp(-alpha * Q0 * x_arr),
            dtype=float,
        )

    def derived(self, params: dict[str, float] | FloatArray) -> DemandDerived:
        """Closed-form Pmax, Omax, and essential value."""
        p = self._coerce_params(params)
        Q0, alpha, k = p
        pmax = _pmax_lambert(alpha, Q0, k)
        qmax = float(self._value(p, np.array([pmax]))[0])
        omax = pmax * qmax
        ev = 1.0 / (alpha * k**1.5)
        return DemandDerived(Pmax=pmax, Omax=omax, essential_value=ev)

    # ---- fitting ------------------------------------------------------------

    def fit(
        self,
        prices: ArrayLike,
        consumption: ArrayLike,
        *,
        log_space: bool = True,
        x0: dict[str, float] | None = None,
        bounds: tuple[dict[str, float], dict[str, float]] | None = None,
        fixed: dict[str, float] | None = None,
    ) -> FitResult:
        """Fit by nonlinear least squares.

        Parameters
        ----------
        prices, consumption
            Observed price array and corresponding consumption.
        log_space
            When ``True`` (default) fit residuals on ``log10(Q)``, which is
            the convention in the Hursh literature and stabilises the fit
            when consumption spans orders of magnitude.
        x0
            Starting values keyed by parameter name. Any omitted parameters
            are auto-initialised from the data.
        bounds
            ``(lower, upper)`` dicts. Parameters not listed get sensible
            defaults (positivity, reasonable ranges).
        fixed
            Parameters to hold constant during the fit.
        """
        return _fit_demand(
            self,
            prices,
            consumption,
            log_space=log_space,
            x0=x0,
            bounds=bounds,
            fixed=fixed,
        )


# ---------------------------------------------------------------------------
# Koffarnus exponentiated
# ---------------------------------------------------------------------------


class Koffarnus(ParametricModel):
    """Koffarnus, Franck, Stein & Bickel (2015) exponentiated demand.

    .. math::

        Q = Q_0 \\cdot 10^{k(\\exp(-\\alpha Q_0 P) - 1)}

    Mathematically equivalent to :class:`HurshSilberberg` for ``Q > 0`` but
    handles zero consumption naturally because the exponentiation never
    diverges. CLAUDE.md §3.2 specifies this as the default for new fits.
    """

    param_names: ClassVar[tuple[str, ...]] = ("Q0", "alpha", "k")

    def _value(self, params: FloatArray, x: FloatArray) -> FloatArray:
        Q0, alpha, k = params
        return np.asarray(Q0 * np.power(10.0, k * (np.exp(-alpha * Q0 * x) - 1.0)), dtype=float)

    def derivative(self, params: dict[str, float] | FloatArray, x: ArrayLike) -> FloatArray:
        """Analytic ``dQ/dP``. Same shape as Hursh-Silberberg by construction."""
        p = self._coerce_params(params)
        Q0, alpha, k = p
        x_arr = np.asarray(x, dtype=float)
        Q = self._value(p, x_arr)
        return np.asarray(-Q * k * _LN10 * alpha * Q0 * np.exp(-alpha * Q0 * x_arr), dtype=float)

    def derived(self, params: dict[str, float] | FloatArray) -> DemandDerived:
        """Closed-form Pmax, Omax, essential value (identical formulae to HS)."""
        p = self._coerce_params(params)
        Q0, alpha, k = p
        pmax = _pmax_lambert(alpha, Q0, k)
        qmax = float(self._value(p, np.array([pmax]))[0])
        omax = pmax * qmax
        ev = 1.0 / (alpha * k**1.5)
        return DemandDerived(Pmax=pmax, Omax=omax, essential_value=ev)

    def fit(
        self,
        prices: ArrayLike,
        consumption: ArrayLike,
        *,
        log_space: bool = False,
        x0: dict[str, float] | None = None,
        bounds: tuple[dict[str, float], dict[str, float]] | None = None,
        fixed: dict[str, float] | None = None,
    ) -> FitResult:
        """Fit by nonlinear least squares.

        Defaults to linear-space residuals so that zero-consumption
        observations can participate in the fit; pass ``log_space=True`` to
        match the Hursh-Silberberg convention (in which case all observations
        must be strictly positive).
        """
        return _fit_demand(
            self,
            prices,
            consumption,
            log_space=log_space,
            x0=x0,
            bounds=bounds,
            fixed=fixed,
        )


# ---------------------------------------------------------------------------
# shared NLS implementation
# ---------------------------------------------------------------------------


def _initial_guess(prices: FloatArray, consumption: FloatArray) -> dict[str, float]:
    """Reasonable starting values from data magnitudes."""
    pos = consumption > 0
    if not np.any(pos):
        return {"Q0": 1.0, "alpha": 0.01, "k": 2.0}
    Q0_guess = float(np.max(consumption[pos]))
    q_min = float(np.min(consumption[pos]))
    span = float(np.log10(Q0_guess / max(q_min, 1e-9)))
    k_guess = max(span, 1.5)
    # alpha rough guess: where Q halves vs P=0 -> alpha*Q0*P_half ~= ln(2)/k
    p_mid = float(np.median(prices[prices > 0])) if np.any(prices > 0) else 1.0
    alpha_guess = float(max(np.log(2) / (k_guess * Q0_guess * max(p_mid, 1e-9)), 1e-6))
    return {"Q0": Q0_guess, "alpha": alpha_guess, "k": k_guess}


def _fit_demand(
    model: ParametricModel,
    prices: ArrayLike,
    consumption: ArrayLike,
    *,
    log_space: bool,
    x0: dict[str, float] | None,
    bounds: tuple[dict[str, float], dict[str, float]] | None,
    fixed: dict[str, float] | None,
) -> FitResult:
    p_arr = np.asarray(prices, dtype=float)
    q_arr = np.asarray(consumption, dtype=float)
    if p_arr.shape != q_arr.shape:
        msg = f"prices and consumption must share shape, got {p_arr.shape} vs {q_arr.shape}"
        raise ValueError(msg)
    if log_space and np.any(q_arr <= 0):
        msg = "log-space fit requires strictly positive consumption; pass log_space=False or filter zeros."
        raise ValueError(msg)

    fixed = fixed or {}
    free_names = [n for n in model.param_names if n not in fixed]
    if not free_names:
        msg = "All parameters fixed; nothing to fit."
        raise ValueError(msg)

    guess = _initial_guess(p_arr, q_arr)
    if x0:
        guess.update(x0)

    default_lower = {"Q0": 1e-9, "alpha": 1e-9, "k": 0.1}
    default_upper = {"Q0": np.inf, "alpha": np.inf, "k": 10.0}
    if bounds:
        default_lower.update(bounds[0])
        default_upper.update(bounds[1])

    x0_vec = np.array([guess[n] for n in free_names], dtype=float)
    lb = np.array([default_lower[n] for n in free_names], dtype=float)
    ub = np.array([default_upper[n] for n in free_names], dtype=float)

    def assemble(free_vec: FloatArray) -> FloatArray:
        full = np.empty(len(model.param_names), dtype=float)
        for i, name in enumerate(model.param_names):
            if name in fixed:
                full[i] = fixed[name]
            else:
                full[i] = free_vec[free_names.index(name)]
        return full

    def residuals(free_vec: FloatArray) -> FloatArray:
        pred = model._value(assemble(free_vec), p_arr)
        if log_space:
            return np.asarray(np.log10(pred) - np.log10(q_arr), dtype=float)
        return np.asarray(pred - q_arr, dtype=float)

    result = least_squares(residuals, x0_vec, bounds=(lb, ub), method="trf")

    full_params = assemble(result.x)
    final_resid = residuals(result.x)
    rss = float(np.sum(final_resid**2))
    n = q_arr.size
    k_free = len(free_names)
    aic, bic = ParametricModel._information_criteria(rss, n, k_free)

    # R^2 reported on the same scale as the fit
    y = np.log10(q_arr) if log_space else q_arr
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")

    # standard errors via Gauss-Newton covariance approximation
    stderr = dict.fromkeys(free_names, float("nan"))
    if result.jac.size and n > k_free:
        try:
            jtj = result.jac.T @ result.jac
            cov = np.linalg.inv(jtj) * (rss / max(n - k_free, 1))
            for i, name in enumerate(free_names):
                stderr[name] = float(np.sqrt(max(cov[i, i], 0.0)))
        except np.linalg.LinAlgError:
            pass

    params: dict[str, float] = {
        name: float(full_params[i]) for i, name in enumerate(model.param_names)
    }
    extra: dict[str, Any] = {"scipy_result": result, "log_space": log_space, "fixed": dict(fixed)}

    return FitResult(
        params=params,
        stderr=stderr,
        aic=aic,
        bic=bic,
        rss=rss,
        r_squared=float(r2),
        n_obs=n,
        success=bool(result.success),
        message=str(result.message),
        extra=extra,
    )


__all__ = [
    "DemandDerived",
    "HurshSilberberg",
    "Koffarnus",
]
