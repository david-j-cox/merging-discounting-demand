"""Reproduce the Phase 3 simulation suite from a fixed seed.

Runs all four sub-experiments (recovery, identifiability, linkage, crossover)
and prints summary numbers. The full figures live in
``notebooks/04_simulation_recovery.ipynb``; this script is for reproducibility
checks and CI use.

Usage::

    uv run python scripts/run_simulation.py [--seed 2025]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from edu.simulation.crossover import run_narrow_vs_wide
from edu.simulation.generate import sample_population
from edu.simulation.identifiability import (
    profile_alpha_B_unified,
    profile_alpha_k_demand,
    ridge_score,
)
from edu.simulation.linkage import linkage_test
from edu.simulation.recovery import sample_size_sweep


def main(seed: int) -> int:
    print(f"=== Phase 3 simulation suite, seed = {seed} ===\n")

    print("--- 1. Parameter recovery (single free k / single fixed k / joint) ---")
    for mode in ["single_free_k", "single_fixed_k", "joint"]:
        print(f"\nmode: {mode}")
        for r in sample_size_sweep([30, 60, 120, 240], mode=mode, seed=seed):
            print("  " + r.summary_row().replace("\n", "\n  "))

    print("\n--- 2. Identifiability (3 representative subjects) ---")
    rng = np.random.default_rng(seed)
    subjects = sample_population(3, rng=rng)
    for i, s in enumerate(subjects):
        prof_ak = profile_alpha_k_demand(s, rng=np.random.default_rng(seed + i))
        prof_aB = profile_alpha_B_unified(s, rng=np.random.default_rng(seed + i + 10))
        print(
            f"  subject {i}: alpha-k ridge_score={ridge_score(prof_ak):.3f}  "
            f"alpha-B ridge_score={ridge_score(prof_aB):.3f}"
        )

    print("\n--- 3. Cross-task linkage (shared vs independent alpha) ---")
    for n in [60, 120, 240]:
        subjects = sample_population(n, rng=np.random.default_rng(seed + n))
        res = linkage_test(subjects, rng=np.random.default_rng(seed + n + 1))
        print(
            f"  n={n}: median ΔBIC={res.median_delta:+.2f}, "
            f"%-favoring-shared={res.pct_favoring_shared:.0%}"
        )

    print("\n--- 4. Crossover (narrow vs wide effort range, n=200) ---")
    narrow, wide = run_narrow_vs_wide(n=200, seed=seed)
    print(f"  narrow: %concave={narrow.fraction_concave():.1%}  (n={narrow.n_subjects})")
    print(f"  wide  : %concave={wide.fraction_concave():.1%}  (n={wide.n_subjects})")
    print(
        "  NOTE: H2 held only within the manipulation-window stratum; see "
        "docs/analysis_plan.md for the reformulated H2 and notebook 04 for "
        "the stratified analysis."
    )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=2025, help="Master RNG seed.")
    args = parser.parse_args()
    sys.exit(main(args.seed))
