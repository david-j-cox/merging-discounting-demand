"""Shared pytest fixtures.

Reused across ``test_bayesian.py`` and ``test_analysis.py``: building a
synthetic cohort matrix-form payload and (optionally) running a short
hierarchical Bayesian fit on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from edu.simulation.generate import (
    PRICE_ARRAY_17,
    SubjectParams,
    sample_population,
    simulate_effort_discounting,
    simulate_purchase_task,
)


@dataclass(frozen=True)
class Cohort:
    """Matrix-form synthetic cohort ready for the Bayesian fitter."""

    P: np.ndarray
    Q_obs: np.ndarray
    E: np.ndarray
    SV_obs: np.ndarray
    B: np.ndarray
    subjects: list[SubjectParams]

    @property
    def true_alpha(self) -> np.ndarray:
        return np.array([s.alpha for s in self.subjects])


def build_cohort(n_subj: int, *, seed: int = 0) -> Cohort:
    """Generate ``n_subj`` synthetic subjects and stack their data."""
    rng = np.random.default_rng(seed)
    subjects = sample_population(n_subj, rng=rng)
    Q_obs = np.zeros((n_subj, len(PRICE_ARRAY_17)))
    E = np.zeros((n_subj, 6))
    SV_obs = np.zeros((n_subj, 6))
    B = np.zeros(n_subj)
    for i, s in enumerate(subjects):
        _, q = simulate_purchase_task(s, rng=rng)
        e, sv = simulate_effort_discounting(s, rng=rng)
        Q_obs[i] = q
        E[i] = e
        SV_obs[i] = sv
        B[i] = s.B
    return Cohort(
        P=PRICE_ARRAY_17,
        Q_obs=Q_obs,
        E=E,
        SV_obs=SV_obs,
        B=B,
        subjects=subjects,
    )


@pytest.fixture
def small_cohort() -> Cohort:
    """6-subject synthetic cohort, deterministic (seed=0)."""
    return build_cohort(n_subj=6, seed=0)


def fit_cohort(
    cohort: Cohort,
    *,
    arm_index: np.ndarray | None = None,
    seed: int = 1,
    n_warmup: int = 150,
    n_samples: int = 150,
    n_chains: int = 2,
) -> Any:
    """Run the hierarchical unified fit on ``cohort``.

    Short chains by default; tests pay ~10s per call. Pass ``arm_index``
    to enable the arm-aware variant.
    """
    from edu.fitting.bayesian import fit_unified_hierarchical

    return fit_unified_hierarchical(
        cohort.P,
        cohort.Q_obs,
        cohort.E,
        cohort.SV_obs,
        cohort.B,
        arm_index=arm_index,
        n_warmup=n_warmup,
        n_samples=n_samples,
        n_chains=n_chains,
        seed=seed,
    )
