"""Smoke tests for the hierarchical Bayesian fitting module.

The full validation is in ``notebooks/04_simulation_recovery.ipynb`` — the
posteriors converge with R-hat < 1.05 and recovery RMSE < 10% at n>=60
with 1000 warmup / 1000 samples per chain. These tests use very short
chains and small n purely to verify the orchestration runs end-to-end and
returns the expected shapes.

Marked ``slow`` because even a short chain takes several seconds; opt out
with ``pytest -m 'not slow'``.
"""

from __future__ import annotations

import numpy as np
import pytest

from edu.fitting.bayesian import (
    DEFAULT_PRIORS,
    Priors,
    fit_demand_hierarchical,
    fit_unified_hierarchical,
)
from edu.simulation.generate import (
    sample_population,
    simulate_effort_discounting,
    simulate_purchase_task,
)

PRICES = np.array([0.01, 0.05, 0.13, 0.25, 0.5, 1, 2, 5, 13, 25, 50, 100, 200, 350, 500, 800, 1120])


def _build_data(
    n_subj: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (P, Q_obs, E, SV_obs, B_anchor, true_alpha) for n_subj subjects."""
    rng = np.random.default_rng(seed)
    subjects = sample_population(n_subj, rng=rng)
    Q_obs = np.zeros((n_subj, len(PRICES)))
    E = np.zeros((n_subj, 6))
    SV = np.zeros((n_subj, 6))
    B = np.zeros(n_subj)
    for i, s in enumerate(subjects):
        _, q = simulate_purchase_task(s, rng=rng)
        e, sv = simulate_effort_discounting(s, rng=rng)
        Q_obs[i] = q
        E[i] = e
        SV[i] = sv
        B[i] = s.B
    return PRICES, Q_obs, E, SV, B, np.array([s.alpha for s in subjects])


def test_priors_default_is_frozen() -> None:
    """``Priors`` is a frozen dataclass; mutation must raise."""
    with pytest.raises((AttributeError, Exception)):
        DEFAULT_PRIORS.log_alpha_mean = 0.0  # type: ignore[misc]


def test_priors_can_be_overridden() -> None:
    p = Priors(log_alpha_mean=-1.5, k_mean=2.5)
    assert p.log_alpha_mean == -1.5
    assert p.k_mean == 2.5


@pytest.mark.slow
def test_demand_hierarchical_runs_and_recovers() -> None:
    """Hierarchical Koffarnus fit with short chains converges (R-hat < 1.1)."""
    P, Q_obs, _, _, _, true_alpha = _build_data(n_subj=6, seed=0)
    idata = fit_demand_hierarchical(
        P,
        Q_obs,
        n_warmup=150,
        n_samples=150,
        n_chains=2,
        seed=1,
    )
    import arviz as az

    summary = az.summary(idata, var_names=["mu_log_alpha", "mu_log_Q0", "mu_k", "sigma_obs"])
    assert (summary["r_hat"] < 1.1).all(), f"R-hat exceeds threshold: {summary['r_hat'].values}"

    # Per-subject alpha posterior-mean should track the truth in rank order.
    # We don't assert tight RMSE here because chains are short for speed.
    alpha_post = idata.posterior["alpha"].mean(dim=("chain", "draw")).values
    rho = np.corrcoef(np.log(alpha_post), np.log(true_alpha))[0, 1]
    assert rho > 0.5, f"per-subject alpha rank correlation too low: {rho:.2f}"


@pytest.mark.slow
def test_unified_hierarchical_runs_with_shared_k() -> None:
    """The default hierarchical unified fit (share_k=True) should converge."""
    P, Q_obs, E, SV_obs, B, true_alpha = _build_data(n_subj=6, seed=0)
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
    # share_k=True means we get k_shared (a scalar) rather than mu_k/sigma_k.
    assert "k_shared" in idata.posterior
    assert "mu_k" not in idata.posterior

    import arviz as az

    summary = az.summary(
        idata, var_names=["mu_log_alpha", "k_shared", "sigma_purchase", "sigma_sv"]
    )
    assert (summary["r_hat"] < 1.1).all()

    alpha_post = idata.posterior["alpha"].mean(dim=("chain", "draw")).values
    rho = np.corrcoef(np.log(alpha_post), np.log(true_alpha))[0, 1]
    assert rho > 0.5


@pytest.mark.slow
def test_unified_hierarchical_runs_with_per_subject_k() -> None:
    """The non-default share_k=False mode should also work and produce mu_k."""
    P, Q_obs, E, SV_obs, B, _ = _build_data(n_subj=6, seed=0)
    idata = fit_unified_hierarchical(
        P,
        Q_obs,
        E,
        SV_obs,
        B,
        share_k=False,
        n_warmup=150,
        n_samples=150,
        n_chains=2,
        seed=1,
    )
    assert "mu_k" in idata.posterior
    assert "sigma_k" in idata.posterior
    assert "k_shared" not in idata.posterior


@pytest.mark.slow
def test_fit_demand_rejects_no_obs() -> None:
    """Calling the model function directly with Q_obs=None should error."""
    import jax.numpy as jnp

    from edu.fitting.bayesian import hierarchical_demand

    with pytest.raises(ValueError, match="Q_obs is required"):
        hierarchical_demand(jnp.array([1.0]), Q_obs=None)
