"""Unified effort-demand model. See ``docs/derivation.md``.

- :class:`UnifiedNaive` (Step 4) — ``SV/A = 10^(k(exp(-alpha*Q0*E/A) - 1))``.
  Asymptotes to ``A * 10^(-k)``; never reaches zero. Demonstrates why
  the capability bound is needed.
- :class:`UnifiedCapabilityBounded` (Step 5, Option B, **preferred**) —
  adds ``Q2(E) <= B/E``; reaches zero at finite ``E``. ``B`` anchors
  to measured MVC / N-back capability.
- :class:`UnifiedNonlinearCost` (Step 5, Option A) — substitutes
  ``psi(E) = E^s`` for the effort axis. No ``B``; for comparison only.
- :func:`with_substitutability` — applies Hursh's ``IF`` factor
  (Step 6): ``Q2_effective = Q2 * (1 - IF * Q1_consumed)``.

Effort ``E`` is in participant-normalised units (E/B fractions). The
SV-from-demand mapping ``SV/A = Q2/Q0`` is derivation assumption A4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from edu.models._base import FloatArray, ParametricModel

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

    ``B`` (effort capability) anchors to measured MVC / N-back; this is
    what makes ``alpha`` and ``B`` jointly identifiable.
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
        """First ``E* > 0`` where the capability bound activates.

        Solves ``Q0 * 10^(k(exp(-alpha*Q0*E/A) - 1)) = B/E`` numerically;
        returns ``nan`` if it never activates on ``(0, e_max]``.

        The two terms can re-cross at very large ``E`` because the demand
        term flattens to a non-zero asymptote ``Q0 * 10^(-k)``. We return
        the *first* sign change, which is the one a participant traverses
        as effort grows; the second has no behavioural meaning.
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
    """Apply Hursh's interaction-factor scaling: ``SV * (1 - IF * Q1_consumed)``.

    ``IF`` (in ``[0, 1]``) is held fixed by the experimenter — it's the
    arm-assigned manipulation we're testing, not a free model parameter
    — so this is a function over ``base`` rather than a class with state.
    ``q1_consumed`` should be normalised to ``[0, 1]``.
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
