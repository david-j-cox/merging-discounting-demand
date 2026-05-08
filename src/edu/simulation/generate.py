"""Synthetic-data generators for the simulation suite.

Sample subject-level parameters from realistic priors and produce the three
task datasets the Phase 4 experiment will collect (CLAUDE.md §4):

1. **Standard purchase task** — 17-price array, demand-curve fit yields
   subject-level ``alpha``, ``Q0``, ``k``.
2. **Effort discounting** — six effort levels (10/25/40/55/70/85% of
   capability), titrated SV via adjusting-amount procedure.
3. **Effort purchase task** — novel; price array in effort units, fixed
   reward, returns "how many effort units would you perform" answers.

All generators take an explicit ``np.random.Generator`` so simulations are
reproducible from a single seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from edu.models.demand import Koffarnus
from edu.models.unified import UnifiedCapabilityBounded

FloatArray = NDArray[np.floating[Any]]


# ---------------------------------------------------------------------------
# Population priors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectParams:
    """Per-subject parameters for the unified capability-bounded model.

    All fields are scalar. The simulation code holds a list/array of these
    when running across many subjects.
    """

    alpha: float  # demand decay rate
    Q0: float  # demand at zero price
    k: float  # log-span (often shared)
    B: float  # effort capability (anchored to measured MVC in real data)


def sample_population(
    n: int,
    *,
    rng: np.random.Generator,
    alpha_loc: float = -2.0,  # log10 mean ~ 0.01
    alpha_scale: float = 0.4,
    Q0_loc: float = 1.0,  # log10 mean ~ 10
    Q0_scale: float = 0.3,
    k_mean: float = 3.0,  # k is often fixed; here we let it vary
    k_sd: float = 0.3,
    B_loc: float = 0.0,  # log10 mean ~ 1.0 (effort-units; arbitrary scale)
    B_scale: float = 0.4,
) -> list[SubjectParams]:
    """Sample ``n`` subjects from log-normal priors on each parameter.

    Defaults span the realistic range described in CLAUDE.md §3 (k ~ 2-4,
    α ~ 0.001-0.1) plus a B distribution centred at unity. Tune via the
    keyword args from a YAML config when running the suite.
    """
    log_alpha = rng.normal(alpha_loc, alpha_scale, size=n)
    log_Q0 = rng.normal(Q0_loc, Q0_scale, size=n)
    log_B = rng.normal(B_loc, B_scale, size=n)
    k_vals = np.clip(rng.normal(k_mean, k_sd, size=n), 1.5, 5.0)

    return [
        SubjectParams(
            alpha=float(10 ** log_alpha[i]),
            Q0=float(10 ** log_Q0[i]),
            k=float(k_vals[i]),
            B=float(10 ** log_B[i]),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Task 1: standard purchase task
# ---------------------------------------------------------------------------


# CLAUDE.md §4 Phase 4 step 3: 17-price array spanning four orders.
PRICE_ARRAY_17 = np.array(
    [
        0.01,
        0.05,
        0.13,
        0.25,
        0.50,
        1.0,
        2.0,
        5.0,
        13.0,
        25.0,
        50.0,
        100.0,
        200.0,
        350.0,
        500.0,
        800.0,
        1120.0,
    ]
)


def simulate_purchase_task(
    subject: SubjectParams,
    *,
    rng: np.random.Generator,
    prices: FloatArray = PRICE_ARRAY_17,
    log_noise_sd: float = 0.10,
) -> tuple[FloatArray, FloatArray]:
    """Generate one subject's purchase-task consumption data.

    Uses the Koffarnus exponentiated form (handles zero consumption) with
    multiplicative log-normal noise — the canonical noise model in the
    behavioural-economic demand literature.

    Returns ``(prices, consumption)``.
    """
    ko = Koffarnus()
    Q_clean = ko.value({"Q0": subject.Q0, "alpha": subject.alpha, "k": subject.k}, prices)
    noise = rng.normal(0.0, log_noise_sd, size=prices.shape)
    Q_obs = Q_clean * np.exp(noise)
    return prices.astype(float), np.asarray(Q_obs, dtype=float)


# ---------------------------------------------------------------------------
# Task 2: effort discounting (titrated SV)
# ---------------------------------------------------------------------------


# CLAUDE.md §4 Phase 4 step 4: six effort levels, expressed as fraction of
# the participant's measured capability B.
EFFORT_FRACTIONS_DEFAULT = np.array([0.10, 0.25, 0.40, 0.55, 0.70, 0.85])


def simulate_effort_discounting(
    subject: SubjectParams,
    *,
    rng: np.random.Generator,
    A: float = 10.0,
    effort_fractions: FloatArray = EFFORT_FRACTIONS_DEFAULT,
    sv_noise_sd: float = 0.5,
) -> tuple[FloatArray, FloatArray]:
    """Generate one subject's effort-discounting SV at each effort level.

    The ``effort_fractions`` are interpreted as fractions of the subject's
    capability ``B`` — so the actual effort axis is ``f * B``. This matches
    the experimental procedure where each subject's effort grid is scaled
    by their own MVC / N-back threshold.

    Returns ``(effort_array, sv_array)`` where effort is in absolute units.
    """
    m = UnifiedCapabilityBounded()
    effort = effort_fractions * subject.B
    params = {
        "A": A,
        "Q0": subject.Q0,
        "alpha": subject.alpha,
        "k": subject.k,
        "B": subject.B,
    }
    sv_clean = m.value(params, effort)
    noise = rng.normal(0.0, sv_noise_sd, size=effort.shape)
    sv_obs = np.maximum(sv_clean + noise, 0.0)  # clip non-negative
    return effort.astype(float), np.asarray(sv_obs, dtype=float)


# ---------------------------------------------------------------------------
# Task 3: effort purchase task (novel)
# ---------------------------------------------------------------------------


# Effort-units price array. This is hand-tuned for the simulation; the real
# experimental array will be calibrated per-subject during Phase 4 piloting.
EFFORT_PRICE_DEFAULT = np.array([0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0])


def simulate_effort_purchase_task(
    subject: SubjectParams,
    *,
    rng: np.random.Generator,
    effort_prices: FloatArray = EFFORT_PRICE_DEFAULT,
    log_noise_sd: float = 0.15,
) -> tuple[FloatArray, FloatArray]:
    """Generate one subject's effort-purchase-task data.

    Each "price" is an effort cost per acquisition; the dependent
    measure is how many acquisitions the subject reports they would
    make. Consumption follows Koffarnus demand:

        Q(P_effort) = Q0 * 10^(k(exp(-alpha*Q0*P_effort) - 1))

    Returns ``(effort_prices, consumption)``.
    """
    ko = Koffarnus()
    Q_clean = ko.value({"Q0": subject.Q0, "alpha": subject.alpha, "k": subject.k}, effort_prices)
    noise = rng.normal(0.0, log_noise_sd, size=effort_prices.shape)
    Q_obs = Q_clean * np.exp(noise)
    return effort_prices.astype(float), np.asarray(Q_obs, dtype=float)


__all__ = [
    "EFFORT_FRACTIONS_DEFAULT",
    "EFFORT_PRICE_DEFAULT",
    "PRICE_ARRAY_17",
    "SubjectParams",
    "sample_population",
    "simulate_effort_discounting",
    "simulate_effort_purchase_task",
    "simulate_purchase_task",
]
