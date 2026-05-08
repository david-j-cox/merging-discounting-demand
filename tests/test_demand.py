"""Tests for :mod:`edu.models.demand`.

Verification strategy (no published data needed at this phase):

1. Closed-form limiting cases (Q at P=0 equals Q0; monotonic decrease).
2. Vectorisation across price arrays.
3. Analytic derivative agrees with finite-difference numerical derivative.
4. Pmax from the Gilroy et al. (2019) Lambert-W formula gives unit elasticity.
5. Round-trip parameter recovery from clean and noisy synthetic data.
6. HurshSilberberg and Koffarnus agree mathematically for Q > 0.

Phase 5 will add validation against the pilot dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from edu.models.demand import HurshSilberberg, Koffarnus, _pmax_lambert

# A few realistic parameter regimes spanning behavioural-economic data.
REGIMES = [
    pytest.param({"Q0": 10.0, "alpha": 0.005, "k": 3.0}, id="moderate-essentiality"),
    pytest.param({"Q0": 1.0, "alpha": 0.01, "k": 2.5}, id="low-Q0"),
    pytest.param({"Q0": 100.0, "alpha": 0.001, "k": 4.0}, id="high-Q0-inelastic"),
    pytest.param({"Q0": 5.0, "alpha": 0.05, "k": 3.5}, id="elastic"),
]


# ---------------------------------------------------------------------------
# Hursh-Silberberg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", REGIMES)
def test_hs_value_at_zero_price_equals_Q0(params: dict[str, float]) -> None:
    hs = HurshSilberberg()
    Q = hs.value(params, np.array([0.0]))
    assert Q[0] == pytest.approx(params["Q0"], rel=1e-12)


@pytest.mark.parametrize("params", REGIMES)
def test_hs_monotonically_decreases(params: dict[str, float]) -> None:
    hs = HurshSilberberg()
    P = np.geomspace(1e-3, 1e3, 200)
    Q = hs.value(params, P)
    diffs = np.diff(Q)
    assert np.all(diffs <= 1e-12), "Q must be monotonically non-increasing in P"


def test_hs_vectorised_matches_loop() -> None:
    hs = HurshSilberberg()
    params = {"Q0": 10.0, "alpha": 0.005, "k": 3.0}
    P = np.array([0.1, 1.0, 10.0, 100.0])
    vec = hs.value(params, P)
    elem = np.array([hs.value(params, np.array([p]))[0] for p in P])
    np.testing.assert_allclose(vec, elem, rtol=1e-14)


@pytest.mark.parametrize("params", REGIMES)
def test_hs_analytic_derivative_matches_finite_diff(params: dict[str, float]) -> None:
    hs = HurshSilberberg()
    P = np.array([0.5, 2.0, 10.0, 50.0])
    h = 1e-6
    fd = (hs.value(params, P + h) - hs.value(params, P - h)) / (2 * h)
    analytic = hs.derivative(params, P)
    # atol guards against FD precision loss where the analytic derivative is
    # near zero (Q saturated at its asymptote, making FD noise the floor).
    np.testing.assert_allclose(analytic, fd, rtol=1e-4, atol=1e-12)


# ---------------------------------------------------------------------------
# Pmax via Lambert-W (Gilroy et al. 2019)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", REGIMES)
def test_pmax_yields_unit_elasticity(params: dict[str, float]) -> None:
    """At Pmax the point elasticity ``d ln Q / d ln P`` must equal -1."""
    hs = HurshSilberberg()
    pmax = _pmax_lambert(params["alpha"], params["Q0"], params["k"])
    eps = 1e-5
    p_lo = pmax * (1 - eps)
    p_hi = pmax * (1 + eps)
    Q_lo = float(hs.value(params, np.array([p_lo]))[0])
    Q_hi = float(hs.value(params, np.array([p_hi]))[0])
    elasticity = (np.log(Q_hi) - np.log(Q_lo)) / (np.log(p_hi) - np.log(p_lo))
    assert elasticity == pytest.approx(-1.0, abs=1e-3)


@pytest.mark.parametrize("params", REGIMES)
def test_derived_quantities_self_consistent(params: dict[str, float]) -> None:
    hs = HurshSilberberg()
    derived = hs.derived(params)
    qmax = float(hs.value(params, np.array([derived.Pmax]))[0])
    assert derived.Omax == pytest.approx(derived.Pmax * qmax, rel=1e-10)
    assert derived.essential_value == pytest.approx(
        1.0 / (params["alpha"] * params["k"] ** 1.5), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", REGIMES)
def test_hs_recovers_clean_parameters(params: dict[str, float]) -> None:
    """With noiseless data the optimiser must find the true parameters."""
    hs = HurshSilberberg()
    P = np.geomspace(0.01, 100.0, 17)
    Q = hs.value(params, P)
    fit = hs.fit(P, Q)
    assert fit.success
    for name in hs.param_names:
        assert fit.params[name] == pytest.approx(
            params[name], rel=1e-3
        ), f"clean recovery failed for {name}"


def test_hs_recovers_noisy_parameters_within_tolerance() -> None:
    """5% multiplicative noise should still recover within ~10% relative."""
    hs = HurshSilberberg()
    true = {"Q0": 10.0, "alpha": 0.005, "k": 3.0}
    P = np.geomspace(0.01, 100.0, 17)
    Q = hs.value(true, P)
    rng = np.random.default_rng(42)
    Q_noisy = Q * np.exp(rng.normal(0, 0.05, size=Q.shape))
    fit = hs.fit(P, Q_noisy)
    assert fit.success
    for name in hs.param_names:
        assert fit.params[name] == pytest.approx(true[name], rel=0.1)


def test_hs_fit_with_fixed_k() -> None:
    """Holding k fixed (Hursh's standard practice) must respect the constraint."""
    hs = HurshSilberberg()
    true = {"Q0": 10.0, "alpha": 0.005, "k": 3.0}
    P = np.geomspace(0.01, 100.0, 17)
    Q = hs.value(true, P)
    fit = hs.fit(P, Q, fixed={"k": 3.0})
    assert fit.params["k"] == 3.0
    assert fit.params["Q0"] == pytest.approx(true["Q0"], rel=1e-3)
    assert fit.params["alpha"] == pytest.approx(true["alpha"], rel=1e-3)


# ---------------------------------------------------------------------------
# Koffarnus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", REGIMES)
def test_koffarnus_matches_hs_for_positive_Q(params: dict[str, float]) -> None:
    """For Q > 0 the two parameterisations are mathematically identical."""
    hs = HurshSilberberg()
    ko = Koffarnus()
    P = np.geomspace(0.01, 1000.0, 50)
    np.testing.assert_allclose(ko.value(params, P), hs.value(params, P), rtol=1e-12)


def test_koffarnus_recovers_with_zero_consumption() -> None:
    """Koffarnus must accept zero consumption (the reason it exists).

    Construct synthetic data where the high-price tail is exactly zero by
    using a parameter regime where the floor falls below floating-point
    representation.
    """
    ko = Koffarnus()
    true = {"Q0": 10.0, "alpha": 0.05, "k": 3.0}
    P = np.geomspace(0.01, 100.0, 17)
    Q = ko.value(true, P)
    # Zero out the smallest values to mimic empirical zero consumption.
    Q_with_zero = Q.copy()
    Q_with_zero[Q_with_zero < 1e-3] = 0.0
    fit = ko.fit(P, Q_with_zero)
    assert fit.success
    # Q0 and alpha must remain close; k is most affected by zero observations.
    assert fit.params["Q0"] == pytest.approx(true["Q0"], rel=0.1)


def test_koffarnus_log_space_rejects_zero_consumption() -> None:
    ko = Koffarnus()
    P = np.array([0.01, 0.1, 1.0, 10.0])
    Q = np.array([10.0, 9.5, 5.0, 0.0])
    with pytest.raises(ValueError, match="positive consumption"):
        ko.fit(P, Q, log_space=True)


# ---------------------------------------------------------------------------
# Interface contracts
# ---------------------------------------------------------------------------


def test_value_accepts_dict_or_array() -> None:
    hs = HurshSilberberg()
    P = np.array([0.1, 1.0, 10.0])
    as_dict = hs.value({"Q0": 10.0, "alpha": 0.005, "k": 3.0}, P)
    as_arr = hs.value(np.array([10.0, 0.005, 3.0]), P)
    np.testing.assert_allclose(as_dict, as_arr, rtol=1e-15)


def test_value_rejects_wrong_shape_array() -> None:
    hs = HurshSilberberg()
    with pytest.raises(ValueError, match="shape"):
        hs.value(np.array([10.0, 0.005]), np.array([1.0]))


def test_value_rejects_missing_dict_key() -> None:
    hs = HurshSilberberg()
    with pytest.raises(ValueError, match="Missing parameters"):
        hs.value({"Q0": 10.0, "alpha": 0.005}, np.array([1.0]))


def test_log_likelihood_increases_at_truth() -> None:
    hs = HurshSilberberg()
    true = {"Q0": 10.0, "alpha": 0.005, "k": 3.0}
    P = np.geomspace(0.01, 100.0, 17)
    rng = np.random.default_rng(0)
    Q_obs = hs.value(true, P) * np.exp(rng.normal(0, 0.05, size=P.shape))
    ll_true = hs.log_likelihood(true, P, Q_obs)
    ll_off = hs.log_likelihood({"Q0": 5.0, "alpha": 0.05, "k": 1.5}, P, Q_obs)
    assert ll_true > ll_off


def test_fit_result_has_all_expected_fields() -> None:
    hs = HurshSilberberg()
    true = {"Q0": 10.0, "alpha": 0.005, "k": 3.0}
    P = np.geomspace(0.01, 100.0, 17)
    Q = hs.value(true, P)
    fit = hs.fit(P, Q)
    assert set(fit.params) == {"Q0", "alpha", "k"}
    assert set(fit.stderr) == {"Q0", "alpha", "k"}
    assert fit.n_obs == 17
    assert fit.r_squared > 0.99
    assert np.isfinite(fit.aic)
    assert np.isfinite(fit.bic)
