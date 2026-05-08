"""Tests for :mod:`edu.models.discounting`.

Verification (no published data needed at this phase):

1. Limiting cases: ``SV(0) == A`` and monotonic decrease in E.
2. Vectorisation across effort arrays.
3. Subtractive forms (Parabolic, Power) clamp at zero.
4. Round-trip recovery from clean and noisy synthetic data.
5. Model comparison: when data is simulated from model X, ``compare_models``
   ranks X first by AIC/BIC.
"""

from __future__ import annotations

import numpy as np
import pytest

from edu.models._base import ParametricModel
from edu.models.discounting import (
    Exponential,
    Hyperbolic,
    Hyperboloid,
    Parabolic,
    Power,
    Sigmoidal,
    compare_models,
    fit_discount,
)

# Effort grid covering CLAUDE.md §4 Phase 4 design (10-85% of capability).
EFFORT = np.array([0.10, 0.25, 0.40, 0.55, 0.70, 0.85])


# Each entry: (class, true params). All produce strictly positive SV on EFFORT.
TRUTH_TABLE = [
    pytest.param(Hyperbolic, {"A": 10.0, "l": 1.5}, id="hyperbolic"),
    pytest.param(Exponential, {"A": 10.0, "l": 1.0}, id="exponential"),
    pytest.param(Parabolic, {"A": 10.0, "l": 8.0}, id="parabolic"),
    pytest.param(Power, {"A": 10.0, "l": 8.0, "s": 1.7}, id="power"),
    pytest.param(Sigmoidal, {"A": 10.0, "s": 5.0, "E0": 0.5}, id="sigmoidal"),
    pytest.param(Hyperboloid, {"A": 10.0, "l": 1.5, "s": 1.2}, id="hyperboloid"),
]


# ---------------------------------------------------------------------------
# limiting cases and shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("cls", "params"), TRUTH_TABLE)
def test_sv_at_zero_effort(cls: type[ParametricModel], params: dict[str, float]) -> None:
    """SV(0) must equal A for every form except Sigmoidal (where it's near A)."""
    m = cls()
    sv0 = float(m.value(params, np.array([0.0]))[0])
    if cls is Sigmoidal:
        # SV(0) = A / (1 + exp(-s * E0)); approaches A for large s*E0.
        assert sv0 < params["A"]
        assert sv0 > 0.5 * params["A"]
    else:
        assert sv0 == pytest.approx(params["A"], rel=1e-12)


@pytest.mark.parametrize(("cls", "params"), TRUTH_TABLE)
def test_sv_monotonically_decreases(cls: type[ParametricModel], params: dict[str, float]) -> None:
    m = cls()
    E = np.linspace(0.0, 5.0, 100)
    sv = m.value(params, E)
    diffs = np.diff(sv)
    assert np.all(diffs <= 1e-12), f"{cls.__name__} must be non-increasing in E"


@pytest.mark.parametrize(("cls", "params"), TRUTH_TABLE)
def test_sv_vectorised_matches_loop(cls: type[ParametricModel], params: dict[str, float]) -> None:
    m = cls()
    vec = m.value(params, EFFORT)
    elem = np.array([m.value(params, np.array([e]))[0] for e in EFFORT])
    np.testing.assert_allclose(vec, elem, rtol=1e-14)


# ---------------------------------------------------------------------------
# subtractive-form clamping
# ---------------------------------------------------------------------------


def test_parabolic_clamped_at_zero() -> None:
    """Parabolic: A - l*E^2 < 0 at large E; value() must return 0."""
    m = Parabolic()
    # A - l*E^2 = 0 at E = sqrt(A/l) = sqrt(10/10) = 1; check just past.
    sv = m.value({"A": 10.0, "l": 10.0}, np.array([0.5, 1.0, 1.5, 5.0]))
    assert sv[0] > 0
    assert sv[1] == pytest.approx(0.0, abs=1e-12)
    assert sv[2] == 0.0
    assert sv[3] == 0.0


def test_power_clamped_at_zero() -> None:
    m = Power()
    sv = m.value({"A": 10.0, "l": 10.0, "s": 2.5}, np.array([0.5, 2.0, 5.0]))
    assert sv[0] > 0
    assert sv[1] == 0.0
    assert sv[2] == 0.0


# ---------------------------------------------------------------------------
# parameter recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("cls", "true"), TRUTH_TABLE)
def test_recovers_clean_parameters(cls: type[ParametricModel], true: dict[str, float]) -> None:
    """Noiseless data must recover the true parameters."""
    m = cls()
    sv = m.value(true, EFFORT)
    fit = fit_discount(m, EFFORT, sv)
    assert fit.success
    for name in m.param_names:
        assert fit.params[name] == pytest.approx(
            true[name], rel=1e-3, abs=1e-6
        ), f"{cls.__name__} clean recovery failed for {name}"


@pytest.mark.parametrize(("cls", "true"), TRUTH_TABLE)
def test_recovers_noisy_parameters(cls: type[ParametricModel], true: dict[str, float]) -> None:
    """Recover within ~15% of the truth at noise σ=0.3 on A=10."""
    m = cls()
    sv = m.value(true, EFFORT)
    rng = np.random.default_rng(0)
    sv_noisy = sv + rng.normal(0, 0.3, size=EFFORT.shape)
    fit = fit_discount(m, EFFORT, sv_noisy)
    assert fit.success
    # A should always recover well; steepness/shape parameters can be looser.
    assert fit.params["A"] == pytest.approx(true["A"], rel=0.15)


def test_fixed_parameter_constraint_respected() -> None:
    """fixed= must hold values constant during the fit."""
    m = Power()
    true = {"A": 10.0, "l": 8.0, "s": 2.0}  # parabolic-like
    sv = m.value(true, EFFORT)
    fit = fit_discount(m, EFFORT, sv, fixed={"s": 2.0})
    assert fit.params["s"] == 2.0
    assert fit.params["A"] == pytest.approx(10.0, rel=1e-3)
    assert fit.params["l"] == pytest.approx(8.0, rel=1e-3)


# ---------------------------------------------------------------------------
# model comparison
# ---------------------------------------------------------------------------


def test_compare_models_returns_all_results() -> None:
    sv = Power().value({"A": 10.0, "l": 8.0, "s": 1.7}, EFFORT)
    rng = np.random.default_rng(0)
    sv_noisy = sv + rng.normal(0, 0.3, size=EFFORT.shape)
    results = compare_models(
        [Hyperbolic(), Exponential(), Power(), Hyperboloid()], EFFORT, sv_noisy
    )
    assert set(results) == {"Hyperbolic", "Exponential", "Power", "Hyperboloid"}
    for r in results.values():
        assert r.success


@pytest.mark.parametrize(
    ("source_cls", "source_params"),
    [
        # Restrict to the larger-difference comparisons. With six effort
        # points and modest noise the AIC differences between e.g.
        # hyperboloid and hyperbolic can be smaller than the comparison's
        # discrimination ability — that's a known small-N limitation, and
        # the simulation suite in Phase 3 will quantify it precisely.
        pytest.param(Power, {"A": 10.0, "l": 8.0, "s": 1.7}, id="from-power"),
        pytest.param(Exponential, {"A": 10.0, "l": 2.5}, id="from-exponential"),
    ],
)
def test_aic_ranks_source_model_first(
    source_cls: type[ParametricModel], source_params: dict[str, float]
) -> None:
    """When data are generated from class X, AIC should rank X best."""
    src = source_cls()
    sv = src.value(source_params, EFFORT)
    rng = np.random.default_rng(0)
    sv_noisy = sv + rng.normal(0, 0.2, size=EFFORT.shape)

    candidates: list[ParametricModel] = [
        Hyperbolic(),
        Exponential(),
        Power(),
        Hyperboloid(),
    ]
    # add the source class if not already in the candidates
    if source_cls.__name__ not in {type(c).__name__ for c in candidates}:
        candidates.append(src)

    results = compare_models(candidates, EFFORT, sv_noisy)
    ranked = sorted(results.items(), key=lambda kv: kv[1].aic)
    assert ranked[0][0] == source_cls.__name__


# ---------------------------------------------------------------------------
# interface contracts (mirrored from test_demand for symmetry)
# ---------------------------------------------------------------------------


def test_value_accepts_dict_or_array() -> None:
    m = Hyperbolic()
    as_dict = m.value({"A": 10.0, "l": 2.0}, EFFORT)
    as_arr = m.value(np.array([10.0, 2.0]), EFFORT)
    np.testing.assert_allclose(as_dict, as_arr, rtol=1e-15)


def test_log_likelihood_increases_at_truth() -> None:
    m = Power()
    true = {"A": 10.0, "l": 8.0, "s": 1.7}
    rng = np.random.default_rng(0)
    sv_obs = m.value(true, EFFORT) + rng.normal(0, 0.3, size=EFFORT.shape)
    ll_true = m.log_likelihood(true, EFFORT, sv_obs)
    ll_off = m.log_likelihood({"A": 5.0, "l": 1.0, "s": 1.0}, EFFORT, sv_obs)
    assert ll_true > ll_off
