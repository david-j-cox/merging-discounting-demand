"""Fit the unified hierarchical model to pilot/main-study data.

Usage::

    uv run python scripts/fit_pilot.py --data data/raw/ --output results/

Reads every JSON in ``--data`` (typically ``data/raw/``) using the loader
in :mod:`edu.analysis.load_pilot`, runs the hierarchical Bayesian fit
with arm-aware covariate, runs the secondary analyses (H1-stronger and
H3), and writes:

* ``results/fit_<timestamp>.nc`` - InferenceData (NetCDF) for re-use.
* ``results/tables/fit_summary_<timestamp>.csv`` - posterior summary.
* ``results/tables/secondary_<timestamp>.csv`` - H1-stronger and H3
  results in a single row.

The CLI is intentionally minimal; expand if/when needed (e.g. priors
override, n_warmup tuning).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from edu.analysis.group import arm_contrast, linkage_correlation
from edu.analysis.individual import summarise_subjects
from edu.analysis.load_pilot import load_directory
from edu.fitting.bayesian import fit_unified_hierarchical


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data", type=Path, default=Path("data/raw"), help="Directory of session JSON files."
    )
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory.")
    parser.add_argument("--n-warmup", type=int, default=1000)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--no-arm",
        action="store_true",
        help="Fit without the arm covariate (single-population). Use only "
        "for sensitivity checks; produces a biased H3 contrast.",
    )
    args = parser.parse_args(argv)

    cohort = load_directory(args.data)
    n = len(cohort.sessions)
    print(f"loaded n={n} sessions from {args.data}")
    print(
        f"  arms: low={int((cohort.arm == 'low').sum())}, high={int((cohort.arm == 'high').sum())}"
    )
    print(f"  P shape: {cohort.P.shape}")
    print(f"  E shape: {cohort.E.shape}")

    arm_index = None if args.no_arm else np.where(cohort.arm == "low", 0, 1).astype(int)

    print(
        f"\nFitting (n_warmup={args.n_warmup}, n_samples={args.n_samples}, "
        f"n_chains={args.n_chains}) ..."
    )
    idata = fit_unified_hierarchical(
        cohort.P,
        cohort.Q_obs,
        cohort.E,
        cohort.SV_obs,
        cohort.B_anchor,
        arm_index=arm_index,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        seed=args.seed,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.output.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fit_path = args.output / f"fit_{timestamp}.nc"
    idata.to_netcdf(str(fit_path))
    print(f"\nposterior saved: {fit_path}")

    # Posterior summary on the population parameters.
    var_names = [
        "mu_log_alpha_per_arm" if arm_index is not None else "mu_log_alpha",
        "sigma_log_alpha",
        "mu_log_Q0",
        "sigma_log_Q0",
        "k_shared",
        "sigma_purchase",
        "sigma_sv",
    ]
    if arm_index is not None:
        var_names.append("diff_log_alpha")
    summary = az.summary(idata, var_names=var_names)
    summary_path = tables_dir / f"fit_summary_{timestamp}.csv"
    summary.to_csv(summary_path)
    print(f"posterior summary saved: {summary_path}")

    # Secondary analyses.
    summaries = summarise_subjects(idata, A=10.0, B_anchor=cohort.B_anchor)
    alpha_means = np.array([s.alpha_mean for s in summaries])
    lambda_implied = np.array([s.lambda_unconstrained_mean for s in summaries])

    # H1-stronger uses the *posterior* alpha against an external steepness.
    # In the pilot we don't yet have a separate Task 2 fit, so we use the
    # model-implied lambda from the posterior summary as a self-consistency
    # check; in the main study the analysis swaps in a per-subject NLS
    # estimate of effort-discount steepness from Task 2 in isolation.
    h1 = linkage_correlation(idata, lambda_implied)

    # H3.
    h3 = arm_contrast(idata, cohort.arm)

    secondary = pd.DataFrame(
        {
            "metric": [
                "linkage_r_mean",
                "linkage_r_hdi_lo",
                "linkage_r_hdi_hi",
                "linkage_p_positive",
                "h3_diff_log10_alpha_mean",
                "h3_diff_log10_alpha_hdi_lo",
                "h3_diff_log10_alpha_hdi_hi",
                "h3_p_predicted_direction",
                "h3_n_low",
                "h3_n_high",
            ],
            "value": [
                h1.correlation_mean,
                h1.correlation_hdi[0],
                h1.correlation_hdi[1],
                h1.p_positive,
                h3.diff_log_alpha_mean,
                h3.diff_log_alpha_hdi[0],
                h3.diff_log_alpha_hdi[1],
                h3.p_predicted_direction,
                h3.n_low,
                h3.n_high,
            ],
        }
    )
    secondary_path = tables_dir / f"secondary_{timestamp}.csv"
    secondary.to_csv(secondary_path, index=False)
    print(f"secondary analyses saved: {secondary_path}")

    print("\n--- Headline results ---")
    print(
        f"H1-stronger linkage: r = {h1.correlation_mean:+.3f}, "
        f"HDI [{h1.correlation_hdi[0]:+.3f}, {h1.correlation_hdi[1]:+.3f}], "
        f"P(r > 0) = {h1.p_positive:.0%}"
    )
    print(
        f"H3 arm contrast: diff log10 alpha = {h3.diff_log_alpha_mean:+.3f}, "
        f"HDI [{h3.diff_log_alpha_hdi[0]:+.3f}, {h3.diff_log_alpha_hdi[1]:+.3f}], "
        f"P(diff > 0) = {h3.p_predicted_direction:.0%}"
    )
    print(f"  per-subject mean alpha range: [{alpha_means.min():.5f}, {alpha_means.max():.5f}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
