"""Functional-form crossover simulation (CLAUDE.md H2).

Same unified-model truth, sampled at narrow (10-60% capability) vs
wide (10-85%) effort ranges. Asks which single-form discount function
wins by AIC in each. Phase 3 found the original H2 prediction
(narrow→exponential, wide→power) was too sharp at the population
level; the convex/concave aggregate is the robust statistic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from edu.models._base import ParametricModel
from edu.models.discounting import (
    Exponential,
    Hyperbolic,
    Hyperboloid,
    Parabolic,
    Power,
    fit_discount,
)
from edu.simulation.generate import (
    SubjectParams,
    sample_population,
    simulate_effort_discounting,
)

CONVEX_FORMS = {"Exponential", "Hyperbolic", "Hyperboloid"}
CONCAVE_FORMS = {"Parabolic", "Power"}


@dataclass(frozen=True)
class CrossoverResult:
    """Per-population AIC-winner counts for one effort range."""

    range_label: str
    best_form_counts: dict[str, int]
    n_subjects: int

    def fraction(self, form: str) -> float:
        return self.best_form_counts.get(form, 0) / self.n_subjects

    def fraction_concave(self) -> float:
        """Fraction of subjects whose AIC-best form is concave (Parabolic / Power)."""
        n_concave = sum(self.best_form_counts.get(f, 0) for f in CONCAVE_FORMS)
        return n_concave / self.n_subjects if self.n_subjects > 0 else float("nan")


CANDIDATES: list[type[ParametricModel]] = [
    Exponential,
    Hyperbolic,
    Hyperboloid,
    Parabolic,
    Power,
]


def crossover_test(
    subjects: list[SubjectParams],
    *,
    rng: np.random.Generator,
    range_label: str,
    effort_fractions: np.ndarray,
    A: float = 10.0,
    sv_noise_sd: float = 0.5,
) -> CrossoverResult:
    """Simulate effort discounting at ``effort_fractions``; return per-subject AIC winners."""
    counts: Counter[str] = Counter()
    n_used = 0
    for s in subjects:
        E, SV = simulate_effort_discounting(
            s,
            rng=rng,
            A=A,
            effort_fractions=effort_fractions,
            sv_noise_sd=sv_noise_sd,
        )
        # Skip degenerate cases where titration produced no variance
        if np.std(SV) < 1e-6:
            continue
        best_aic = np.inf
        best_name = ""
        any_success = False
        for cls in CANDIDATES:
            try:
                fit = fit_discount(cls(), E, SV)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
            if fit.success and fit.aic < best_aic:
                any_success = True
                best_aic = fit.aic
                best_name = cls.__name__
        if any_success:
            counts[best_name] += 1
            n_used += 1

    return CrossoverResult(
        range_label=range_label,
        best_form_counts=dict(counts),
        n_subjects=n_used,
    )


# CLAUDE.md §4 Phase 3 step 4: narrow vs wide effort ranges.
NARROW_FRACTIONS = np.array([0.10, 0.20, 0.30, 0.45, 0.60])
WIDE_FRACTIONS = np.array([0.10, 0.25, 0.40, 0.55, 0.70, 0.85])


def run_narrow_vs_wide(
    n: int = 200,
    *,
    seed: int = 2025,
    sv_noise_sd: float = 0.5,
) -> tuple[CrossoverResult, CrossoverResult]:
    """Convenience: run narrow + wide crossover at the standard sample size."""
    rng_narrow = np.random.default_rng(seed)
    rng_wide = np.random.default_rng(seed + 1)
    rng_subj = np.random.default_rng(seed + 2)
    subjects = sample_population(n, rng=rng_subj)

    narrow = crossover_test(
        subjects,
        rng=rng_narrow,
        range_label="narrow",
        effort_fractions=NARROW_FRACTIONS,
        sv_noise_sd=sv_noise_sd,
    )
    wide = crossover_test(
        subjects,
        rng=rng_wide,
        range_label="wide",
        effort_fractions=WIDE_FRACTIONS,
        sv_noise_sd=sv_noise_sd,
    )
    return narrow, wide


__all__ = [
    "CONCAVE_FORMS",
    "CONVEX_FORMS",
    "NARROW_FRACTIONS",
    "WIDE_FRACTIONS",
    "CrossoverResult",
    "crossover_test",
    "run_narrow_vs_wide",
]
