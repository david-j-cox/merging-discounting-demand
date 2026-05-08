"""Smoke tests for the hierarchical Bayesian fitting module.

The full validation is in ``notebooks/04_simulation_recovery.ipynb``; these
tests just verify the orchestration runs end-to-end at small n with short
chains. Marked ``slow`` because each fit takes several seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from edu.fitting.bayesian import (
    DEFAULT_PRIORS,
    Priors,
    fit_demand_hierarchical,
)

from .conftest import Cohort, fit_cohort


def test_priors_default_is_frozen() -> None:
    """``Priors`` is a frozen dataclass; mutation must raise."""
    with pytest.raises((AttributeError, Exception)):
        DEFAULT_PRIORS.log_alpha_mean = 0.0  # type: ignore[misc]


def test_priors_can_be_overridden() -> None:
    p = Priors(log_alpha_mean=-1.5, k_mean=2.5)
    assert p.log_alpha_mean == -1.5
    assert p.k_mean == 2.5


@pytest.mark.slow
def test_demand_hierarchical_runs_and_recovers(small_cohort: Cohort) -> None:
    """Hierarchical Koffarnus fit with short chains converges (R-hat < 1.1)."""
    idata = fit_demand_hierarchical(
        small_cohort.P,
        small_cohort.Q_obs,
        n_warmup=150,
        n_samples=150,
        n_chains=2,
        seed=1,
    )
    import arviz as az

    summary = az.summary(idata, var_names=["mu_log_alpha", "mu_log_Q0", "mu_k", "sigma_obs"])
    assert (summary["r_hat"] < 1.1).all(), f"R-hat exceeds threshold: {summary['r_hat'].values}"

    # Short chains: rank correlation rather than tight RMSE.
    alpha_post = idata.posterior["alpha"].mean(dim=("chain", "draw")).values
    rho = np.corrcoef(np.log(alpha_post), np.log(small_cohort.true_alpha))[0, 1]
    assert rho > 0.5, f"per-subject alpha rank correlation too low: {rho:.2f}"


@pytest.mark.slow
def test_unified_hierarchical_runs(small_cohort: Cohort) -> None:
    """The hierarchical unified fit converges and exposes the expected variables."""
    idata = fit_cohort(small_cohort)
    assert "k_shared" in idata.posterior
    assert "mu_k" not in idata.posterior  # the per-subject-k branch is no longer in the API

    import arviz as az

    summary = az.summary(
        idata, var_names=["mu_log_alpha", "k_shared", "sigma_purchase", "sigma_sv"]
    )
    assert (summary["r_hat"] < 1.1).all()

    alpha_post = idata.posterior["alpha"].mean(dim=("chain", "draw")).values
    rho = np.corrcoef(np.log(alpha_post), np.log(small_cohort.true_alpha))[0, 1]
    assert rho > 0.5


@pytest.mark.slow
def test_fit_demand_rejects_no_obs() -> None:
    """Calling the model function directly with Q_obs=None should error."""
    import jax.numpy as jnp

    from edu.fitting.bayesian import hierarchical_demand

    with pytest.raises(ValueError, match="Q_obs is required"):
        hierarchical_demand(jnp.array([1.0]), Q_obs=None)
