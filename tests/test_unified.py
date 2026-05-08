"""Tests for :mod:`edu.models.unified`.

Mirrors the structure of ``docs/derivation.md`` — each Step in the math has
a corresponding test class. The Phase 3 simulation suite will exercise the
model at scale; these tests verify per-equation correctness.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from edu.models._base import ParametricModel
from edu.models.discounting import Exponential, Hyperbolic, Power
from edu.models.unified import (
    UnifiedCapabilityBounded,
    UnifiedNaive,
    UnifiedNonlinearCost,
    with_substitutability,
)

# Effort grid matching CLAUDE.md §4 Phase 4 design (10-85% of capability).
EFFORT = np.array([0.10, 0.25, 0.40, 0.55, 0.70, 0.85])


# ---------------------------------------------------------------------------
# Step 4 — naive form
# ---------------------------------------------------------------------------


class TestUnifiedNaive:
    """SV/A = 10^(k(exp(-alpha*Q0*E/A) - 1))."""

    PARAMS: ClassVar[dict[str, float]] = {"A": 10.0, "Q0": 5.0, "alpha": 0.05, "k": 3.0}

    def test_sv_at_zero_effort_equals_A(self) -> None:
        m = UnifiedNaive()
        sv0 = float(m.value(self.PARAMS, np.array([0.0]))[0])
        assert sv0 == pytest.approx(self.PARAMS["A"], rel=1e-12)

    def test_monotonically_decreases(self) -> None:
        m = UnifiedNaive()
        E = np.linspace(0.0, 100.0, 500)
        sv = m.value(self.PARAMS, E)
        assert np.all(np.diff(sv) <= 1e-12)

    def test_asymptote_approached_at_large_E(self) -> None:
        """Naive form's defining failure: it stops at A * 10^(-k), not zero."""
        m = UnifiedNaive()
        sv_far = float(m.value(self.PARAMS, np.array([1e6]))[0])
        expected = m.asymptote(self.PARAMS)
        assert expected == pytest.approx(self.PARAMS["A"] * 10 ** (-self.PARAMS["k"]), rel=1e-12)
        assert sv_far == pytest.approx(expected, rel=1e-9)
        # And the asymptote is non-zero — this is exactly why Option B exists.
        assert sv_far > 0.0

    def test_vectorised_matches_loop(self) -> None:
        m = UnifiedNaive()
        vec = m.value(self.PARAMS, EFFORT)
        elem = np.array([m.value(self.PARAMS, np.array([e]))[0] for e in EFFORT])
        np.testing.assert_allclose(vec, elem, rtol=1e-14)


# ---------------------------------------------------------------------------
# Step 5, Option B — capability-bounded
# ---------------------------------------------------------------------------


class TestUnifiedCapabilityBounded:
    """SV/A = min[demand-term, B/(Q0 E)]."""

    # Parameters tuned so the crossover lies inside the experimental range.
    PARAMS: ClassVar[dict[str, float]] = {
        "A": 10.0,
        "Q0": 5.0,
        "alpha": 0.05,
        "k": 3.0,
        "B": 0.5,
    }

    def test_sv_at_zero_effort_equals_A(self) -> None:
        m = UnifiedCapabilityBounded()
        sv0 = float(m.value(self.PARAMS, np.array([0.0]))[0])
        assert sv0 == pytest.approx(self.PARAMS["A"], rel=1e-12)

    def test_reaches_arbitrarily_small_sv_at_large_E(self) -> None:
        """The truncation regime drives SV asymptotically to zero (cf. naive)."""
        m = UnifiedCapabilityBounded()
        sv_far = float(m.value(self.PARAMS, np.array([1e5]))[0])
        assert sv_far < 1e-3
        assert sv_far > 0  # not exactly zero in floating point

    def test_monotonically_decreases(self) -> None:
        m = UnifiedCapabilityBounded()
        E = np.linspace(0.0, 100.0, 500)
        sv = m.value(self.PARAMS, E)
        assert np.all(np.diff(sv) <= 1e-12)

    def test_two_regimes_present_in_realistic_range(self) -> None:
        """At least one effort point is unconstrained, at least one is truncated."""
        m = UnifiedCapabilityBounded()
        out = m.value_with_regime(self.PARAMS, EFFORT)
        assert (out.regime == 0).any(), "no point in the unconstrained regime"
        assert (out.regime == 1).any(), "no point in the truncated regime"

    def test_crossover_is_inside_regime_transition(self) -> None:
        """E* should lie between the last unconstrained point and the first truncated."""
        m = UnifiedCapabilityBounded()
        E_dense = np.linspace(1e-4, 5.0, 1000)
        out = m.value_with_regime(self.PARAMS, E_dense)
        assert np.isfinite(out.crossover)
        # All E < crossover should be unconstrained (regime 0); all E > crossover truncated.
        before = E_dense < out.crossover
        after = E_dense > out.crossover
        assert (out.regime[before] == 0).all()
        assert (out.regime[after] == 1).all()

    def test_crossover_returns_nan_when_capability_never_binds(self) -> None:
        m = UnifiedCapabilityBounded()
        params_huge_B = {**self.PARAMS, "B": 1e9}
        assert np.isnan(m.crossover_effort(params_huge_B))

    def test_capability_term_dominates_at_high_E(self) -> None:
        """In the truncated regime, SV ~ A * B / (Q0 * E)."""
        m = UnifiedCapabilityBounded()
        E_high = np.array([10.0, 50.0, 100.0])
        sv = m.value(self.PARAMS, E_high)
        expected = self.PARAMS["A"] * self.PARAMS["B"] / (self.PARAMS["Q0"] * E_high)
        np.testing.assert_allclose(sv, expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# Step 5, Option A — nonlinear cost
# ---------------------------------------------------------------------------


class TestUnifiedNonlinearCost:
    """SV/A = 10^(k(exp(-alpha*Q0*E^s/A) - 1))."""

    BASE: ClassVar[dict[str, float]] = {"A": 10.0, "Q0": 5.0, "alpha": 0.05, "k": 3.0}

    def test_s_equals_one_recovers_naive(self) -> None:
        a = UnifiedNonlinearCost()
        n = UnifiedNaive()
        params_a = {**self.BASE, "s": 1.0}
        E = np.linspace(0.0, 5.0, 50)
        np.testing.assert_allclose(a.value(params_a, E), n.value(self.BASE, E), rtol=1e-12)

    def test_sv_at_zero_effort_equals_A(self) -> None:
        m = UnifiedNonlinearCost()
        params = {**self.BASE, "s": 2.0}
        sv0 = float(m.value(params, np.array([0.0]))[0])
        assert sv0 == pytest.approx(self.BASE["A"], rel=1e-12)

    def test_monotonically_decreases_for_s_gt_zero(self) -> None:
        m = UnifiedNonlinearCost()
        for s in [0.5, 1.0, 1.5, 2.0, 3.0]:
            params = {**self.BASE, "s": s}
            E = np.linspace(0.0, 50.0, 500)
            sv = m.value(params, E)
            assert np.all(np.diff(sv) <= 1e-12), f"non-monotone for s={s}"


# ---------------------------------------------------------------------------
# Step 6 — substitutability
# ---------------------------------------------------------------------------


class TestSubstitutability:
    PARAMS: ClassVar[dict[str, float]] = {
        "A": 10.0,
        "Q0": 5.0,
        "alpha": 0.05,
        "k": 3.0,
        "B": 0.5,
    }

    def test_if_zero_returns_unchanged_sv(self) -> None:
        m = UnifiedCapabilityBounded()
        sv_base = m.value(self.PARAMS, EFFORT)
        sv_wrap = with_substitutability(
            m, self.PARAMS, EFFORT, interaction_factor=0.0, q1_consumed=1.0
        )
        np.testing.assert_allclose(sv_wrap, sv_base, rtol=1e-12)

    def test_if_one_with_q1_one_zeroes_sv(self) -> None:
        m = UnifiedCapabilityBounded()
        sv_wrap = with_substitutability(
            m, self.PARAMS, EFFORT, interaction_factor=1.0, q1_consumed=1.0
        )
        np.testing.assert_allclose(sv_wrap, 0.0, atol=1e-12)

    def test_partial_substitution_lies_between(self) -> None:
        m = UnifiedCapabilityBounded()
        sv0 = with_substitutability(m, self.PARAMS, EFFORT, interaction_factor=0.0, q1_consumed=1.0)
        sv_half = with_substitutability(
            m, self.PARAMS, EFFORT, interaction_factor=0.5, q1_consumed=1.0
        )
        sv1 = with_substitutability(m, self.PARAMS, EFFORT, interaction_factor=1.0, q1_consumed=1.0)
        # sv1 <= sv_half <= sv0 elementwise
        assert (sv1 <= sv_half + 1e-12).all()
        assert (sv_half <= sv0 + 1e-12).all()

    def test_rejects_invalid_interaction_factor(self) -> None:
        m = UnifiedCapabilityBounded()
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            with_substitutability(m, self.PARAMS, EFFORT, interaction_factor=1.5, q1_consumed=1.0)


# ---------------------------------------------------------------------------
# Limiting-case correspondence to single-form discount functions
# ---------------------------------------------------------------------------


class TestLimitingCases:
    """The unified model must reproduce the standard single-form discount
    functions in the appropriate parameter regimes. CLAUDE.md §4 Phase 2 step 7
    requires this; the exact correspondence is approximate (different
    functional forms with shared limiting behaviour) so we test by fitting
    each candidate to data simulated from the unified model and checking
    R²-of-fit, not parameter equivalence.
    """

    @staticmethod
    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def test_unconstrained_regime_fits_well_with_exponential(self) -> None:
        """In the unconstrained regime, the naive form is exponential-like.

        Fit Exponential to data simulated from UnifiedCapabilityBounded with
        a parameter regime where every point lies in regime 0 (capability
        never activates across the sample range, even though it could
        activate at very large E far outside the sample).
        """
        m = UnifiedCapabilityBounded()
        params = {"A": 10.0, "Q0": 5.0, "alpha": 0.5, "k": 1.5, "B": 100.0}
        E_dense = np.linspace(0.05, 0.85, 30)
        out = m.value_with_regime(params, E_dense)
        assert (out.regime == 0).all(), "test setup: capability should not bind here"

        from edu.models.discounting import fit_discount

        fit = fit_discount(Exponential(), E_dense, out.sv)
        assert fit.success
        assert (
            fit.r_squared > 0.99
        ), f"unconstrained regime should fit Exponential well, got R²={fit.r_squared:.4f}"

    def test_capability_truncated_regime_fits_well_with_hyperbolic(self) -> None:
        """In the fully truncated regime SV = A*B/(Q0*E), which is hyperbolic.

        Hyperbolic is SV = A / (1 + l*E). For E >> 1/l this approaches A/(l*E),
        an exact 1/E form. So data dominated by the truncation regime should
        fit Hyperbolic well.
        """
        m = UnifiedCapabilityBounded()
        # B small so truncation activates immediately past E ~ 0.05.
        params = {"A": 10.0, "Q0": 5.0, "alpha": 0.05, "k": 3.0, "B": 0.05}
        E_dense = np.linspace(0.10, 1.5, 30)  # all in truncated regime
        sv = m.value(params, E_dense)

        from edu.models.discounting import fit_discount

        fit = fit_discount(Hyperbolic(), E_dense, sv)
        assert fit.success
        assert fit.r_squared > 0.99

    def test_mixed_regime_data_better_fit_by_power_than_exponential(self) -> None:
        """When the capability bound activates mid-range (so the data span
        both unconstrained and truncated regions), the Power form should
        beat Exponential — the Bialaszek (2017) finding that motivates the
        unification.
        """
        m = UnifiedCapabilityBounded()
        # B=1.5 with these other params yields crossover ~0.39, with 5
        # unconstrained and 7 truncated points across [0.10, 0.85].
        params = {"A": 10.0, "Q0": 5.0, "alpha": 0.3, "k": 2.0, "B": 1.5}
        E = np.linspace(0.10, 0.85, 12)
        out = m.value_with_regime(params, E)
        assert (out.regime == 0).any() and (
            out.regime == 1
        ).any(), "test setup: data must span both regimes"

        from edu.models.discounting import fit_discount

        fit_exp = fit_discount(Exponential(), E, out.sv)
        fit_pow = fit_discount(Power(), E, out.sv)
        assert fit_pow.aic < fit_exp.aic, (
            f"Power should beat Exponential on mixed-regime data; "
            f"got AIC_pow={fit_pow.aic:.2f}, AIC_exp={fit_exp.aic:.2f}"
        )


# ---------------------------------------------------------------------------
# Interface contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [UnifiedNaive, UnifiedCapabilityBounded, UnifiedNonlinearCost],
)
def test_value_accepts_dict_or_array(cls: type[ParametricModel]) -> None:
    m = cls()
    if cls is UnifiedNaive:
        d = {"A": 10.0, "Q0": 5.0, "alpha": 0.05, "k": 3.0}
    elif cls is UnifiedCapabilityBounded:
        d = {"A": 10.0, "Q0": 5.0, "alpha": 0.05, "k": 3.0, "B": 0.5}
    else:
        d = {"A": 10.0, "Q0": 5.0, "alpha": 0.05, "k": 3.0, "s": 1.5}
    arr = np.array([d[n] for n in m.param_names])
    np.testing.assert_allclose(m.value(d, EFFORT), m.value(arr, EFFORT), rtol=1e-15)
