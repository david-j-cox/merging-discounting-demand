"""Tests for the secondary-analysis pipeline.

Two layers of validation:

- Fast tests for pure functions (loader, summary, contrast logic).
- Slow tests that run a short Bayesian fit and exercise the full pipeline.

The full ground-truth validation (CIs contain the truth, etc.) lives in
``notebooks/05_pilot_analysis.ipynb`` so we don't pay a multi-minute
sampling cost in pytest.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from edu.analysis.load_pilot import (
    InvalidPayload,
    assemble_cohort,
    parse_session,
)

from .conftest import Cohort, fit_cohort


def _good_payload(sid: str = "abc", arm: str = "low", p_max: float = 7.0) -> dict[str, Any]:
    return {
        "study": "edu-pilot",
        "version": "0.1.0",
        "sessionId": sid,
        "timestampIso": "2026-05-08T12:00:00Z",
        "testMode": True,
        "prolificId": None,
        "randomization": {"arm": arm, "taskOrder": ["a", "b", "c"], "seed": 1},
        "calibration": {"pMax": p_max, "ok": True, "trials": []},
        "task1": {
            "commodity": "snack item",
            "trials": [
                {"price": 0.01, "quantity": 20},
                {"price": 1.0, "quantity": 10},
                {"price": 100.0, "quantity": 1},
            ],
        },
        "task2": {
            "perFraction": [
                {
                    "fraction": 0.10,
                    "titration": {"indifference": 8.0, "steps": [], "rewardMax": 10.0},
                },
                {
                    "fraction": 0.40,
                    "titration": {"indifference": 5.0, "steps": [], "rewardMax": 10.0},
                },
                {
                    "fraction": 0.85,
                    "titration": {"indifference": 1.5, "steps": [], "rewardMax": 10.0},
                },
            ],
            "allChoices": [],
            "pMaxUsed": p_max,
        },
        "task3": {"trials": [], "pMaxUsed": p_max, "feasibilityCapsByPrice": []},
        "rawTrials": [],
    }


# ---------------------------------------------------------------------------
# Loader: pure-function tests
# ---------------------------------------------------------------------------


class TestParseSession:
    def test_round_trips_valid_payload(self) -> None:
        s = parse_session(_good_payload())
        assert s.session_id == "abc"
        assert s.arm == "low"
        assert s.p_max == 7.0
        assert s.P.shape == (3,)
        assert s.Q_obs.shape == (3,)
        assert s.E.shape == (3,)
        assert s.SV_obs.shape == (3,)

    def test_E_scaled_by_p_max(self) -> None:
        s = parse_session(_good_payload(p_max=4.0))
        # Effort fractions in the payload: 0.10, 0.40, 0.85.
        np.testing.assert_allclose(s.E, [0.4, 1.6, 3.4])

    def test_rejects_missing_pMax(self) -> None:
        bad = _good_payload()
        del bad["calibration"]["pMax"]
        with pytest.raises(InvalidPayload, match="pMax"):
            parse_session(bad)

    def test_rejects_negative_pMax(self) -> None:
        bad = _good_payload()
        bad["calibration"]["pMax"] = -1.0
        with pytest.raises(InvalidPayload, match="positive"):
            parse_session(bad)

    def test_rejects_unknown_arm(self) -> None:
        bad = _good_payload()
        bad["randomization"]["arm"] = "medium"
        with pytest.raises(InvalidPayload, match="arm"):
            parse_session(bad)

    def test_rejects_empty_trials(self) -> None:
        bad = _good_payload()
        bad["task1"]["trials"] = []
        with pytest.raises(InvalidPayload, match="non-empty"):
            parse_session(bad)


class TestAssembleCohort:
    def test_stacks_consistent_sessions(self) -> None:
        a = parse_session(_good_payload(sid="a", arm="low", p_max=7.0))
        b = parse_session(_good_payload(sid="b", arm="high", p_max=5.5))
        cohort = assemble_cohort([a, b])
        assert cohort.Q_obs.shape == (2, 3)
        assert cohort.E.shape == (2, 3)
        assert list(cohort.arm) == ["low", "high"]
        np.testing.assert_allclose(cohort.B_anchor, [7.0, 5.5])

    def test_rejects_mismatched_price_arrays(self) -> None:
        a = parse_session(_good_payload(sid="a"))
        bad_payload = _good_payload(sid="b")
        bad_payload["task1"]["trials"][0]["price"] = 999.0
        b = parse_session(bad_payload)
        with pytest.raises(InvalidPayload, match="price array"):
            assemble_cohort([a, b])

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(InvalidPayload, match="No sessions"):
            assemble_cohort([])


# ---------------------------------------------------------------------------
# Slow: integration with the Bayesian fit
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSecondaryAnalyses:
    """Smoke-level integration tests: run a tiny fit, check return shapes."""

    def test_summarise_subjects_returns_one_per_subject(self, small_cohort: Cohort) -> None:
        from edu.analysis.individual import summarise_subjects

        idata = fit_cohort(small_cohort)
        summaries = summarise_subjects(idata, A=10.0, B_anchor=small_cohort.B)
        assert len(summaries) == small_cohort.B.size
        for s in summaries:
            assert np.isfinite(s.alpha_mean)
            assert s.alpha_hdi[0] <= s.alpha_mean <= s.alpha_hdi[1]
            assert s.lambda_unconstrained_mean > 0

    def test_linkage_correlation_runs_and_returns_finite(self, small_cohort: Cohort) -> None:
        from edu.analysis.group import linkage_correlation

        idata = fit_cohort(small_cohort)
        true_Q0 = np.array([s.Q0 for s in small_cohort.subjects])
        external = small_cohort.true_alpha * true_Q0 * 3.0 * np.log(10) / 10.0
        res = linkage_correlation(idata, external)
        assert np.isfinite(res.correlation_mean)
        assert -1.0 <= res.correlation_mean <= 1.0
        assert res.n_subjects == small_cohort.B.size
        assert 0.0 <= res.p_positive <= 1.0

    def test_arm_contrast_handles_balanced_arms(self, small_cohort: Cohort) -> None:
        """Preferred path: fit with arm_index, read diff_log_alpha directly."""
        from edu.analysis.group import arm_contrast

        n = small_cohort.B.size
        half = n // 2
        arm_index = np.array([0] * half + [1] * (n - half), dtype=int)
        idata = fit_cohort(small_cohort, arm_index=arm_index)
        arm = np.array(["low"] * half + ["high"] * (n - half))
        res = arm_contrast(idata, arm)
        assert res.n_low == half
        assert res.n_high == n - half
        assert np.isfinite(res.diff_log_alpha_mean)

    def test_arm_contrast_fallback_warns_on_single_population_fit(
        self, small_cohort: Cohort
    ) -> None:
        """Fallback path: single-population fit triggers the bias warning."""
        import warnings

        from edu.analysis.group import arm_contrast

        idata = fit_cohort(small_cohort)  # fit without arm_index
        n = small_cohort.B.size
        half = n // 2
        arm = np.array(["low"] * half + ["high"] * (n - half))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = arm_contrast(idata, arm)
        assert any("biased" in str(warning.message) for warning in w)
        assert np.isfinite(res.diff_log_alpha_mean)

    def test_arm_contrast_rejects_empty_arm_in_fallback(self, small_cohort: Cohort) -> None:
        from edu.analysis.group import arm_contrast

        idata = fit_cohort(small_cohort)
        all_low = np.array(["low"] * small_cohort.B.size)
        with pytest.raises(ValueError, match="at least one"):
            arm_contrast(idata, all_low)

    def test_posterior_predictive_purchase_shapes(self, small_cohort: Cohort) -> None:
        from edu.analysis.group import posterior_predictive_purchase

        idata = fit_cohort(small_cohort)
        n = small_cohort.B.size
        ppc = posterior_predictive_purchase(idata, small_cohort.P, small_cohort.Q_obs, n_draws=50)
        assert ppc.replicates.shape == (50, n, len(small_cohort.P))
        assert ppc.observed.shape == (n, len(small_cohort.P))

    def test_posterior_predictive_sv_shapes(self, small_cohort: Cohort) -> None:
        from edu.analysis.group import posterior_predictive_sv

        idata = fit_cohort(small_cohort)
        n = small_cohort.B.size
        ppc = posterior_predictive_sv(
            idata, small_cohort.E, small_cohort.SV_obs, small_cohort.B, n_draws=50
        )
        assert ppc.replicates.shape == (50, n, 6)
        assert ppc.observed.shape == (n, 6)
