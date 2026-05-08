"""Smoke tests for the simulation suite.

The suite IS the verification — running it is the test. These tests
ensure that the orchestration code runs without error at small n and
returns finite numbers, so future code changes don't silently break the
suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from edu.simulation.crossover import run_narrow_vs_wide
from edu.simulation.generate import (
    sample_population,
    simulate_effort_discounting,
    simulate_effort_purchase_task,
    simulate_purchase_task,
)
from edu.simulation.identifiability import (
    profile_alpha_B_unified,
    profile_alpha_k_demand,
    ridge_score,
)
from edu.simulation.linkage import linkage_test
from edu.simulation.recovery import sample_size_sweep


def test_sample_population_reproducible() -> None:
    rng1 = np.random.default_rng(2025)
    rng2 = np.random.default_rng(2025)
    s1 = sample_population(5, rng=rng1)
    s2 = sample_population(5, rng=rng2)
    assert all(a.alpha == b.alpha for a, b in zip(s1, s2, strict=True))
    assert all(a.B == b.B for a, b in zip(s1, s2, strict=True))


def test_simulate_purchase_task_shape() -> None:
    rng = np.random.default_rng(0)
    subjects = sample_population(1, rng=rng)
    P, Q = simulate_purchase_task(subjects[0], rng=rng)
    assert P.shape == Q.shape
    assert (Q >= 0).all()


def test_simulate_effort_discounting_shape() -> None:
    rng = np.random.default_rng(0)
    subjects = sample_population(1, rng=rng)
    E, SV = simulate_effort_discounting(subjects[0], rng=rng)
    assert E.shape == SV.shape
    assert (SV >= 0).all(), "SV is clipped non-negative"


def test_simulate_effort_purchase_task_shape() -> None:
    rng = np.random.default_rng(0)
    subjects = sample_population(1, rng=rng)
    P, Q = simulate_effort_purchase_task(subjects[0], rng=rng)
    assert P.shape == Q.shape
    assert (Q >= 0).all()


def test_recovery_returns_finite_summary_at_small_n() -> None:
    reports = sample_size_sweep([10, 20], mode="joint")
    assert len(reports) == 2
    for r in reports:
        assert r.n in (10, 20)
        for name, p in r.per_param.items():
            assert np.isfinite(p.bias_rel), f"{name} bias non-finite at n={r.n}"
            assert np.isfinite(p.rmse_rel), f"{name} rmse non-finite"
            assert 0.0 <= p.success_rate <= 1.0


def test_recovery_modes_run() -> None:
    """All three recovery modes should run without error at small n."""
    for mode in ["single_free_k", "single_fixed_k", "joint"]:
        reports = sample_size_sweep([10], mode=mode)
        assert reports[0].n == 10
        assert reports[0].per_param  # non-empty


def test_recovery_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        sample_size_sweep([10], mode="bogus")


def test_identifiability_profile_alpha_k_runs() -> None:
    rng = np.random.default_rng(0)
    subject = sample_population(1, rng=rng)[0]
    prof = profile_alpha_k_demand(
        subject,
        rng=rng,
        alpha_grid=np.geomspace(subject.alpha / 2, subject.alpha * 2, 5),
        k_grid=np.linspace(max(subject.k - 0.5, 0.5), subject.k + 0.5, 5),
    )
    assert prof.nll.shape == (5, 5)
    assert np.isfinite(prof.nll).any()
    assert 0.0 <= ridge_score(prof) <= 1.0


def test_identifiability_profile_alpha_B_runs() -> None:
    rng = np.random.default_rng(0)
    subject = sample_population(1, rng=rng)[0]
    prof = profile_alpha_B_unified(
        subject,
        rng=rng,
        alpha_grid=np.geomspace(subject.alpha / 2, subject.alpha * 2, 5),
        B_grid=np.geomspace(subject.B / 2, subject.B * 2, 5),
    )
    assert prof.nll.shape == (5, 5)


def test_linkage_runs_at_small_n() -> None:
    rng = np.random.default_rng(0)
    subjects = sample_population(5, rng=rng)
    res = linkage_test(subjects, rng=rng)
    assert np.isfinite(res.median_delta) or np.isnan(res.median_delta)
    assert 0.0 <= res.pct_favoring_shared <= 1.0 or np.isnan(res.pct_favoring_shared)


def test_crossover_runs_at_small_n() -> None:
    narrow, wide = run_narrow_vs_wide(n=10, seed=42)
    assert narrow.n_subjects > 0
    assert wide.n_subjects > 0
    assert 0.0 <= narrow.fraction_concave() <= 1.0
    assert 0.0 <= wide.fraction_concave() <= 1.0
