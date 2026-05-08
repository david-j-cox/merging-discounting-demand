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

    def _fit(self) -> dict[str, Any]:
        # Build a tiny synthetic cohort and fit.
        from edu.fitting.bayesian import fit_unified_hierarchical
        from edu.simulation.generate import (
            sample_population,
            simulate_effort_discounting,
            simulate_purchase_task,
        )

        rng = np.random.default_rng(0)
        subjects = sample_population(6, rng=rng)
        P = np.array(
            [0.01, 0.05, 0.13, 0.25, 0.5, 1, 2, 5, 13, 25, 50, 100, 200, 350, 500, 800, 1120]
        )
        Q_obs = np.zeros((6, len(P)))
        E = np.zeros((6, 6))
        SV_obs = np.zeros((6, 6))
        B = np.zeros(6)
        for i, s in enumerate(subjects):
            _, q = simulate_purchase_task(s, rng=rng)
            e, sv = simulate_effort_discounting(s, rng=rng)
            Q_obs[i] = q
            E[i] = e
            SV_obs[i] = sv
            B[i] = s.B
        idata = fit_unified_hierarchical(
            P,
            Q_obs,
            E,
            SV_obs,
            B,
            n_warmup=150,
            n_samples=150,
            n_chains=2,
            seed=1,
        )
        return {
            "idata": idata,
            "P": P,
            "Q_obs": Q_obs,
            "E": E,
            "SV_obs": SV_obs,
            "B": B,
            "subjects": subjects,
        }

    def test_summarise_subjects_returns_one_per_subject(self) -> None:
        from edu.analysis.individual import summarise_subjects

        out = self._fit()
        summaries = summarise_subjects(out["idata"], A=10.0, B_anchor=out["B"])
        assert len(summaries) == 6
        for s in summaries:
            assert np.isfinite(s.alpha_mean)
            assert s.alpha_hdi[0] <= s.alpha_mean <= s.alpha_hdi[1]
            assert s.lambda_unconstrained_mean > 0

    def test_linkage_correlation_runs_and_returns_finite(self) -> None:
        from edu.analysis.group import linkage_correlation

        out = self._fit()
        # Use the true alpha-derived lambda as the external steepness.
        true_alpha = np.array([s.alpha for s in out["subjects"]])
        true_Q0 = np.array([s.Q0 for s in out["subjects"]])
        external = true_alpha * true_Q0 * 3.0 * np.log(10) / 10.0
        res = linkage_correlation(out["idata"], external)
        assert np.isfinite(res.correlation_mean)
        assert -1.0 <= res.correlation_mean <= 1.0
        assert res.n_subjects == 6
        assert 0.0 <= res.p_positive <= 1.0

    def test_arm_contrast_handles_balanced_arms(self) -> None:
        """Preferred path: fit with arm_index, read diff_log_alpha directly."""
        from edu.analysis.group import arm_contrast
        from edu.fitting.bayesian import fit_unified_hierarchical

        # Build a tiny cohort and fit with arm_index.
        from edu.simulation.generate import (
            sample_population,
            simulate_effort_discounting,
            simulate_purchase_task,
        )

        rng = np.random.default_rng(0)
        subjects = sample_population(6, rng=rng)
        P = np.array(
            [0.01, 0.05, 0.13, 0.25, 0.5, 1, 2, 5, 13, 25, 50, 100, 200, 350, 500, 800, 1120]
        )
        Q_obs = np.zeros((6, len(P)))
        E = np.zeros((6, 6))
        SV_obs = np.zeros((6, 6))
        B = np.zeros(6)
        for i, s in enumerate(subjects):
            _, q = simulate_purchase_task(s, rng=rng)
            e, sv = simulate_effort_discounting(s, rng=rng)
            Q_obs[i] = q
            E[i] = e
            SV_obs[i] = sv
            B[i] = s.B
        arm_index = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        idata = fit_unified_hierarchical(
            P,
            Q_obs,
            E,
            SV_obs,
            B,
            arm_index=arm_index,
            n_warmup=150,
            n_samples=150,
            n_chains=2,
            seed=1,
        )
        # arm_contrast() should detect diff_log_alpha and use it.
        arm = np.array(["low", "low", "low", "high", "high", "high"])
        res = arm_contrast(idata, arm)
        assert res.n_low == 3
        assert res.n_high == 3
        assert np.isfinite(res.diff_log_alpha_mean)

    def test_arm_contrast_fallback_warns_on_single_population_fit(self) -> None:
        """Fallback path: single-population fit triggers the bias warning."""
        import warnings

        from edu.analysis.group import arm_contrast

        out = self._fit()  # fit without arm_index
        arm = np.array(["low", "low", "low", "high", "high", "high"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = arm_contrast(out["idata"], arm)
        assert any("biased" in str(warning.message) for warning in w)
        assert np.isfinite(res.diff_log_alpha_mean)

    def test_arm_contrast_rejects_empty_arm_in_fallback(self) -> None:
        from edu.analysis.group import arm_contrast

        out = self._fit()
        all_low = np.array(["low"] * 6)
        # Fallback path raises on missing arm.
        with pytest.raises(ValueError, match="at least one"):
            arm_contrast(out["idata"], all_low)

    def test_posterior_predictive_purchase_shapes(self) -> None:
        from edu.analysis.group import posterior_predictive_purchase

        out = self._fit()
        ppc = posterior_predictive_purchase(out["idata"], out["P"], out["Q_obs"], n_draws=50)
        assert ppc.replicates.shape == (50, 6, len(out["P"]))
        assert ppc.observed.shape == (6, len(out["P"]))

    def test_posterior_predictive_sv_shapes(self) -> None:
        from edu.analysis.group import posterior_predictive_sv

        out = self._fit()
        ppc = posterior_predictive_sv(out["idata"], out["E"], out["SV_obs"], out["B"], n_draws=50)
        assert ppc.replicates.shape == (50, 6, 6)
        assert ppc.observed.shape == (6, 6)
