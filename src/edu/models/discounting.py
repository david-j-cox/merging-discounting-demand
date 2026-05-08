"""Effort-discount models with a shared interface.

Six discount functions from the effort-discounting literature, all expressed
as :math:`SV(E)` where ``SV`` is subjective value, ``E`` is effort (in
participant-specific normalised units, typically fraction of MVC or N-back
threshold), and ``A`` is the unconstrained reward magnitude.

Functional forms (CLAUDE.md §3.3):

================  ================================  ============================
class             :math:`SV(E)`                     reference
================  ================================  ============================
Hyperbolic        :math:`A / (1 + l E)`             Mazur 1987 / Mitchell tradn.
Exponential       :math:`A \\exp(-l E)`              Treadway et al. 2009
Parabolic         :math:`A - l E^2`                 Hartmann et al. 2013
Power             :math:`A - l E^s`                 Bialaszek et al. 2017
Sigmoidal         :math:`A / (1 + \\exp(s(E - E_0)))` Klein-Flügge et al. 2015
Hyperboloid       :math:`A / (1 + l E)^s`           Rachlin / Myerson-Green
================  ================================  ============================

The parabolic and power forms are *subtractive*: ``SV`` can drop below zero
mathematically. The :meth:`value` methods clamp the output at zero because
subjective value below zero has no behavioural meaning (a participant cannot
be willing to pay to *avoid* a reward they would otherwise accept).

Conventions:

* ``A`` is the maximum reward (always free, positive).
* ``l`` (the "lambda" steepness) is positive across all multiplicative forms;
  larger values mean steeper discounting.
* ``s`` differs by form: a power exponent for Power and Hyperboloid, a slope
  parameter for Sigmoidal. Each docstring is explicit.
* ``E0`` (Sigmoidal only) is the inflection point.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from edu.models._base import FitResult, FloatArray, ParametricModel


def _safe_clip(arr: FloatArray) -> FloatArray:
    """Clip subjective value at zero from below (no negative SV)."""
    return np.asarray(np.maximum(arr, 0.0), dtype=float)


# ---------------------------------------------------------------------------
# Hyperbolic
# ---------------------------------------------------------------------------


class Hyperbolic(ParametricModel):
    """Hyperbolic discount: :math:`SV = A / (1 + l E)`.

    The default form in delay discounting (Mazur 1987) and a common starting
    point in effort discounting. Asymptotes to zero; never negative.
    """

    param_names: ClassVar[tuple[str, ...]] = ("A", "l")

    def _value(self, params: FloatArray, x: FloatArray) -> FloatArray:
        A, l_ = params
        return np.asarray(A / (1.0 + l_ * x), dtype=float)


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------


class Exponential(ParametricModel):
    """Exponential discount: :math:`SV = A \\exp(-l E)`.

    The economically rational form (constant rate per unit effort). Treadway's
    EEfRT analyses use this when separating effort cost from probability.
    """

    param_names: ClassVar[tuple[str, ...]] = ("A", "l")

    def _value(self, params: FloatArray, x: FloatArray) -> FloatArray:
        A, l_ = params
        return np.asarray(A * np.exp(-l_ * x), dtype=float)


# ---------------------------------------------------------------------------
# Parabolic
# ---------------------------------------------------------------------------


class Parabolic(ParametricModel):
    """Parabolic discount: :math:`SV = \\max(A - l E^2,\\ 0)`.

    Hartmann, Hager, Tobler & Kaiser (2013) introduced this form for physical
    effort. Subtractive: SV crosses zero at finite ``E = sqrt(A / l)``. We
    clamp at zero rather than letting it go negative.
    """

    param_names: ClassVar[tuple[str, ...]] = ("A", "l")

    def _value(self, params: FloatArray, x: FloatArray) -> FloatArray:
        A, l_ = params
        return _safe_clip(A - l_ * x**2)


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


class Power(ParametricModel):
    """Power discount: :math:`SV = \\max(A - l E^s,\\ 0)`.

    Bialaszek, Marcowski & Ostaszewski (2017) reported this form best across
    physical and cognitive effort. Generalises the parabolic case (``s = 2``).
    """

    param_names: ClassVar[tuple[str, ...]] = ("A", "l", "s")

    def _value(self, params: FloatArray, x: FloatArray) -> FloatArray:
        A, l_, s = params
        return _safe_clip(A - l_ * np.power(x, s))


# ---------------------------------------------------------------------------
# Sigmoidal (Klein-Flügge)
# ---------------------------------------------------------------------------


class Sigmoidal(ParametricModel):
    """Sigmoidal discount: :math:`SV = A / (1 + \\exp(s(E - E_0)))`.

    Klein-Flügge, Kennerley, Friston & Bestmann (2015) used a sigmoidal cost
    function to model neural data; here it is parameterised so that ``E0`` is
    the inflection point and ``s`` is the slope at that point. Note that
    ``s`` here is *not* a power exponent (cf. :class:`Power`,
    :class:`Hyperboloid`).
    """

    param_names: ClassVar[tuple[str, ...]] = ("A", "s", "E0")

    def _value(self, params: FloatArray, x: FloatArray) -> FloatArray:
        A, s, E0 = params
        return np.asarray(A / (1.0 + np.exp(s * (x - E0))), dtype=float)


# ---------------------------------------------------------------------------
# Hyperboloid (Rachlin / Myerson-Green)
# ---------------------------------------------------------------------------


class Hyperboloid(ParametricModel):
    """Hyperboloid discount: :math:`SV = A / (1 + l E)^s`.

    Rachlin's generalisation of the hyperbolic. Exponent ``s`` modulates how
    sharply value drops at high effort. With ``s = 1`` reduces to
    :class:`Hyperbolic`.
    """

    param_names: ClassVar[tuple[str, ...]] = ("A", "l", "s")

    def _value(self, params: FloatArray, x: FloatArray) -> FloatArray:
        A, l_, s = params
        return np.asarray(A * np.power(1.0 + l_ * x, -s), dtype=float)


# ---------------------------------------------------------------------------
# Shared NLS fitting (parameterised by model)
# ---------------------------------------------------------------------------


def _initial_guess_discount(model: ParametricModel, e: FloatArray, sv: FloatArray) -> FloatArray:
    """Reasonable starting values per model class."""
    A_guess = float(np.max(sv)) if sv.size else 1.0
    # rough steepness: where SV drops to half of A
    half = A_guess / 2.0
    e_half_idx = int(np.argmin(np.abs(sv - half))) if sv.size else 0
    e_half = float(e[e_half_idx]) if e.size and e[e_half_idx] > 0 else 1.0
    e_max_sq = max(float(np.max(e)) ** 2, 1e-3) if e.size else 1.0

    name = type(model).__name__
    if name == "Hyperbolic":
        return np.array([A_guess, 1.0 / max(e_half, 1e-3)], dtype=float)
    if name == "Exponential":
        return np.array([A_guess, np.log(2) / max(e_half, 1e-3)], dtype=float)
    if name == "Parabolic":
        return np.array([A_guess, A_guess / e_max_sq], dtype=float)
    if name == "Power":
        return np.array([A_guess, A_guess / e_max_sq, 2.0], dtype=float)
    if name == "Sigmoidal":
        return np.array([A_guess, 5.0, max(e_half, 0.5)], dtype=float)
    if name == "Hyperboloid":
        return np.array([A_guess, 1.0 / max(e_half, 1e-3), 1.0], dtype=float)
    msg = f"No default guess for model {name}"
    raise ValueError(msg)


_DEFAULT_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "Hyperbolic": {"A": (1e-9, np.inf), "l": (1e-9, np.inf)},
    "Exponential": {"A": (1e-9, np.inf), "l": (1e-9, np.inf)},
    "Parabolic": {"A": (1e-9, np.inf), "l": (1e-9, np.inf)},
    "Power": {"A": (1e-9, np.inf), "l": (1e-9, np.inf), "s": (0.1, 10.0)},
    "Sigmoidal": {"A": (1e-9, np.inf), "s": (1e-3, 1e3), "E0": (0.0, np.inf)},
    "Hyperboloid": {"A": (1e-9, np.inf), "l": (1e-9, np.inf), "s": (0.1, 10.0)},
}


def fit_discount(
    model: ParametricModel,
    effort: ArrayLike,
    sv: ArrayLike,
    *,
    x0: dict[str, float] | None = None,
    bounds: tuple[dict[str, float], dict[str, float]] | None = None,
    fixed: dict[str, float] | None = None,
) -> FitResult:
    """Fit a discount model by nonlinear least squares.

    The helper is exposed at module level (rather than as a method) because
    the six classes share a common fitting loop with class-specific defaults
    and there is no per-class state worth carrying.
    """
    e_arr = np.asarray(effort, dtype=float)
    sv_arr = np.asarray(sv, dtype=float)
    if e_arr.shape != sv_arr.shape:
        msg = f"effort and sv must share shape, got {e_arr.shape} vs {sv_arr.shape}"
        raise ValueError(msg)

    fixed = fixed or {}
    free_names = [n for n in model.param_names if n not in fixed]
    if not free_names:
        msg = "All parameters fixed; nothing to fit."
        raise ValueError(msg)

    x0_full = _initial_guess_discount(model, e_arr, sv_arr)
    guess = dict(zip(model.param_names, x0_full, strict=True))
    if x0:
        guess.update(x0)

    cls_defaults = _DEFAULT_BOUNDS[type(model).__name__]
    lower = {n: cls_defaults[n][0] for n in model.param_names}
    upper = {n: cls_defaults[n][1] for n in model.param_names}
    if bounds:
        lower.update(bounds[0])
        upper.update(bounds[1])

    x0_vec = np.array([guess[n] for n in free_names], dtype=float)
    lb = np.array([lower[n] for n in free_names], dtype=float)
    ub = np.array([upper[n] for n in free_names], dtype=float)

    def assemble(free_vec: FloatArray) -> FloatArray:
        full = np.empty(len(model.param_names), dtype=float)
        for i, name in enumerate(model.param_names):
            full[i] = fixed[name] if name in fixed else free_vec[free_names.index(name)]
        return full

    def residuals(free_vec: FloatArray) -> FloatArray:
        pred = model._value(assemble(free_vec), e_arr)
        return np.asarray(pred - sv_arr, dtype=float)

    result = least_squares(residuals, x0_vec, bounds=(lb, ub), method="trf")

    full_params = assemble(result.x)
    final_resid = residuals(result.x)
    rss = float(np.sum(final_resid**2))
    n = sv_arr.size
    k_free = len(free_names)
    aic, bic = ParametricModel._information_criteria(rss, n, k_free)
    tss = float(np.sum((sv_arr - np.mean(sv_arr)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")

    stderr = dict.fromkeys(free_names, float("nan"))
    if result.jac.size and n > k_free:
        try:
            jtj = result.jac.T @ result.jac
            cov = np.linalg.inv(jtj) * (rss / max(n - k_free, 1))
            for i, name in enumerate(free_names):
                stderr[name] = float(np.sqrt(max(cov[i, i], 0.0)))
        except np.linalg.LinAlgError:
            pass

    params = {name: float(full_params[i]) for i, name in enumerate(model.param_names)}
    extra: dict[str, Any] = {"scipy_result": result, "fixed": dict(fixed)}

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


def compare_models(
    models: list[ParametricModel],
    effort: ArrayLike,
    sv: ArrayLike,
) -> dict[str, FitResult]:
    """Fit every model in ``models`` and return results keyed by class name.

    Convenience for the kind of model comparison Bialaszek et al. (2017) ran:
    fit each candidate function to the same per-subject data and rank by
    AIC / BIC.
    """
    return {type(m).__name__: fit_discount(m, effort, sv) for m in models}


__all__ = [
    "Exponential",
    "Hyperbolic",
    "Hyperboloid",
    "Parabolic",
    "Power",
    "Sigmoidal",
    "compare_models",
    "fit_discount",
]
