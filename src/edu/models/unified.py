"""Unified effort-demand model.

Three forms following ``docs/derivation.md``:

:class:`UnifiedNaive`
    Step 4 of the derivation. ``SV/A = 10^(k(exp(-alpha*Q0*E/A) - 1))``.
    Asymptotes to ``A * 10^(-k)``; never reaches zero. Useful as a baseline
    and to demonstrate why the capability bound is needed.

:class:`UnifiedCapabilityBounded`
    Step 5, **Option B** — preferred. Adds ``Q2(E) <= B/E`` as a physical
    feasibility constraint. ``SV/A = min[demand-term, B/(Q0 E)]``. Reaches
    zero at large ``E``. ``value()`` also returns a regime flag.

:class:`UnifiedNonlinearCost`
    Step 5, **Option A**. Replaces ``E`` with ``psi(E) = E^s`` inside the
    Hursh-Silberberg exponent. No capability parameter ``B``; does not link
    to measured MVC / N-back. Implemented for comparison only.

:func:`with_substitutability`
    Wraps any unified model with the Hursh interaction factor ``IF``
    (``§3.5`` in CLAUDE.md, ``Step 6`` in the derivation):
    ``Q2_effective = Q2 * (1 - IF * Q1_consumed)``.

All forms share the parameters ``A`` (reward magnitude), ``Q0``
(unconstrained demand at zero effort), ``alpha`` (demand decay rate), and
``k`` (log-span). Option B adds ``B`` (effort capability bound). Option A
adds ``s`` (effort-cost exponent).

Conventions:

* Effort ``E`` is measured in participant-specific normalised units
  (typically fraction of MVC or N-back threshold), so ``E ~ 0.1`` to
  ``1.0`` is the experimental range.
* The naive form is rotated into ``SV(E)`` space — output is subjective
  value, not consumption rate. The relationship ``SV/A = Q2/Q0`` is the
  derivation's A4 mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from edu.models._base import FloatArray, ParametricModel

_LN10 = float(np.log(10.0))


# ---------------------------------------------------------------------------
# Naive form (Step 4)
# ---------------------------------------------------------------------------


class UnifiedNaive(ParametricModel):
    """Step 4 — naive unification of demand and discounting.

    .. math::

        \\frac{SV(E)}{A} = 10^{k\\left(\\exp\\left(
            -\\alpha Q_0 E / A
        \\right) - 1\\right)}

    Equivalent to substituting ``P = E / A`` into the Koffarnus demand form
    and reading the result as ``SV/A`` rather than ``Q/Q0``. Asymptotes
    to ``A * 10^(-k)`` as ``E -> infinity``; never crosses zero.
    """

    param_names: ClassVar[tuple[str, ...]] = ("A", "Q0", "alpha", "k")

    def _value(self, params: FloatArray, x: FloatArray) -> FloatArray:
        A, Q0, alpha, k = params
        # P_effort = E / A  ->  Q/Q0 = 10^(k(exp(-alpha*Q0*E/A) - 1))  ->  SV/A
        ratio = np.power(10.0, k * (np.exp(-alpha * Q0 * x / A) - 1.0))
        return np.asarray(A * ratio, dtype=float)

    def asymptote(self, params: dict[str, float] | FloatArray) -> float:
        """``SV(E -> infinity) = A * 10^(-k)``. Used in tests and the notebook."""
        p = self._coerce_params(params)
        A, _Q0, _alpha, k = p
        return float(A * np.power(10.0, -k))


# ---------------------------------------------------------------------------
# Option B — capability-bounded (Step 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityBoundedResult:
    """Output of :meth:`UnifiedCapabilityBounded.value_with_regime`.

    Attributes
    ----------
    sv : ndarray
        Subjective value at each requested effort.
    regime : ndarray of int
        Regime indicator: 0 = unconstrained (demand-curve term binds),
        1 = truncated (capability term ``B/E`` binds).
    crossover : float
        The effort ``E*`` at which the two regimes meet, found numerically.
        ``nan`` if no crossover exists in the parameter regime (i.e. the
        capability is so high it never binds).
    """

    sv: FloatArray
    regime: np.ndarray  # int dtype; np.ndarray rather than FloatArray
    crossover: float


class UnifiedCapabilityBounded(ParametricModel):
    """Step 5, Option B — preferred unified model.

    .. math::

        \\frac{SV(E)}{A} = \\min\\left[
            10^{k(\\exp(-\\alpha Q_0 E / A) - 1)},
            \\frac{B}{Q_0 E}
        \\right]

    The ``B`` parameter is the participant's effort capability — measurable
    via MVC for physical effort or N-back threshold for cognitive effort.
    When ``B`` is anchored to that measurement (CLAUDE.md §9), the model
    becomes identifiable even when ``alpha`` and ``B`` would trade off
    freely.
    """

    param_names: ClassVar[tuple[str, ...]] = ("A", "Q0", "alpha", "k", "B")

    def _value(self, params: FloatArray, x: FloatArray) -> FloatArray:
        A, Q0, alpha, k, B = params
        demand_term = np.power(10.0, k * (np.exp(-alpha * Q0 * x / A) - 1.0))
        # Capability term: B / (Q0 * E). Guard the divide-by-zero at E = 0
        # by returning the demand term there (the capability never binds at
        # E = 0 because Q0 < B/E for any finite B as E -> 0+).
        with np.errstate(divide="ignore", invalid="ignore"):
            capability_term = np.where(x > 0, B / (Q0 * x), np.inf)
        ratio = np.minimum(demand_term, capability_term)
        return np.asarray(A * ratio, dtype=float)

    def value_with_regime(
        self, params: dict[str, float] | FloatArray, x: ArrayLike
    ) -> CapabilityBoundedResult:
        """Like :meth:`value`, but also returns the active regime per point."""
        p = self._coerce_params(params)
        A, Q0, alpha, k, B = p
        x_arr = np.asarray(x, dtype=float)

        demand_term = np.power(10.0, k * (np.exp(-alpha * Q0 * x_arr / A) - 1.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            capability_term = np.where(x_arr > 0, B / (Q0 * x_arr), np.inf)
        sv = A * np.minimum(demand_term, capability_term)
        regime = (capability_term < demand_term).astype(int)

        return CapabilityBoundedResult(
            sv=np.asarray(sv, dtype=float),
            regime=regime,
            crossover=self.crossover_effort(p),
        )

    def crossover_effort(
        self, params: dict[str, float] | FloatArray, *, e_max: float = 1e6
    ) -> float:
        """The first effort ``E* > 0`` at which the capability bound activates.

        Solves ``Q0 * 10^(k(exp(-alpha*Q0*E/A) - 1)) = B / E`` numerically.
        Returns ``nan`` if the capability bound never activates on
        ``(0, e_max]``.

        The two terms can intersect more than once: the capability term
        ``B/E`` falls monotonically, but the demand term flattens to a
        non-zero asymptote ``Q0 * 10^(-k)``. If that asymptote lies above
        ``B/e_max`` the curves re-cross at large ``E`` (capability re-binds
        once demand has saturated). The *experimentally relevant* crossover
        is the first sign change as ``E`` increases from zero — that is the
        one a participant traverses as effort grows. We return that one.
        """
        p = self._coerce_params(params)
        A, Q0, alpha, k, B = p

        def diff(E: float) -> float:
            demand = Q0 * np.power(10.0, k * (np.exp(-alpha * Q0 * E / A) - 1.0))
            capability = B / E
            return float(demand - capability)

        # Scan for the first sign change with a logarithmic grid; refine
        # within the bracketing interval via Brent.
        grid = np.geomspace(1e-9, e_max, 200)
        try:
            vals = np.array([diff(float(e)) for e in grid])
        except (FloatingPointError, ValueError):
            return float("nan")
        sign_change = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
        if sign_change.size == 0:
            return float("nan")
        i = int(sign_change[0])
        try:
            return float(brentq(diff, float(grid[i]), float(grid[i + 1]), xtol=1e-9, maxiter=200))
        except (ValueError, RuntimeError):
            return float("nan")


# ---------------------------------------------------------------------------
# Option A — nonlinear cost
# ---------------------------------------------------------------------------


class UnifiedNonlinearCost(ParametricModel):
    """Step 5, Option A — psi(E) = E^s substituted into the demand exponent.

    .. math::

        \\frac{SV(E)}{A} = 10^{k(\\exp(-\\alpha Q_0 E^s / A) - 1)}

    No capability parameter; the concavity of ``SV(E)`` is governed entirely
    by ``s``:

    * ``s = 1`` recovers :class:`UnifiedNaive`.
    * ``s > 1`` produces concave SV(E) (parabolic-like).
    * ``s < 1`` produces convex SV(E) (sigmoidal-like at low E).

    Implemented for cross-form comparison; Option B is preferred theoretically
    because ``B`` links to measured capability while ``s`` is a free shape
    parameter with no direct behavioural-physical anchor.
    """

    param_names: ClassVar[tuple[str, ...]] = ("A", "Q0", "alpha", "k", "s")

    def _value(self, params: FloatArray, x: FloatArray) -> FloatArray:
        A, Q0, alpha, k, s = params
        # E^s for E >= 0; clip negatives (shouldn't happen in practice).
        x_safe = np.maximum(x, 0.0)
        psi = np.power(x_safe, s)
        ratio = np.power(10.0, k * (np.exp(-alpha * Q0 * psi / A) - 1.0))
        return np.asarray(A * ratio, dtype=float)


# ---------------------------------------------------------------------------
# Substitutability wrapper (Step 6 / CLAUDE.md §3.5)
# ---------------------------------------------------------------------------


def with_substitutability(
    base: ParametricModel,
    params: dict[str, float] | FloatArray,
    effort: ArrayLike,
    *,
    interaction_factor: float,
    q1_consumed: ArrayLike | float,
) -> FloatArray:
    """Apply Hursh's interaction-factor scaling to a unified-model output.

    ``Q2_effective(E) = Q2(E) * (1 - IF * Q1_consumed)``.

    The wrapper is exposed as a function rather than a class because ``IF``
    and ``Q1_consumed`` are *experimental* quantities (assigned by arm or
    measured per trial) rather than free model parameters. Letting the
    fitter touch ``IF`` would conflate the manipulation we're trying to
    test with the model that interprets it.

    Parameters
    ----------
    base
        Any :class:`ParametricModel` (typically a Unified* class).
    params
        Parameter dict / vector for ``base``.
    effort
        Effort levels at which to evaluate.
    interaction_factor
        ``IF`` ∈ ``[0, 1]``. ``0`` is independence, ``1`` is perfect
        substitutes. Held fixed by the experimenter.
    q1_consumed
        Consumption of the effortless commodity. Scalar (constant across
        trials) or array (per trial). Should be normalised to lie in
        ``[0, 1]`` so the bracketed factor stays in ``[0, 1]``.
    """
    if not 0.0 <= interaction_factor <= 1.0:
        msg = f"IF must lie in [0, 1], got {interaction_factor}"
        raise ValueError(msg)
    sv = base.value(params, effort)
    q1 = np.asarray(q1_consumed, dtype=float)
    factor = np.clip(1.0 - interaction_factor * q1, 0.0, 1.0)
    return np.asarray(sv * factor, dtype=float)


__all__ = [
    "CapabilityBoundedResult",
    "UnifiedCapabilityBounded",
    "UnifiedNaive",
    "UnifiedNonlinearCost",
    "with_substitutability",
]
