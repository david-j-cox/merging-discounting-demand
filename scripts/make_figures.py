"""Render publication figures from a saved hierarchical-fit posterior.

Usage::

    uv run python scripts/make_figures.py --fit results/fit_<ts>.nc \\
                                           --data data/raw/ \\
                                           --output results/figures/

Reads the InferenceData written by ``scripts/fit_pilot.py``, plus the
underlying session JSONs (so we can plot replicates against
observations), and produces a fixed set of figures destined for the
paper:

  fig01_alpha_per_arm.png         per-arm alpha posteriors
  fig02_h3_contrast.png           H3 arm-difference posterior with HDI
  fig03_linkage_correlation.png   per-subject alpha vs lambda, posterior r
  fig04_ppc_purchase.png          purchase-task posterior predictive
  fig05_ppc_sv.png                effort-discount posterior predictive

Each figure is saved as PNG and PDF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from edu.analysis.group import (
    arm_contrast,
    linkage_correlation,
    posterior_predictive_purchase,
    posterior_predictive_sv,
)
from edu.analysis.individual import _stack_samples, summarise_subjects
from edu.analysis.load_pilot import load_directory


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved: {path.with_suffix('.png').name} and .pdf")


def figure_alpha_per_arm(idata: az.InferenceData, output: Path) -> None:
    if "mu_log_alpha_per_arm" not in idata.posterior:
        print("[fig01] mu_log_alpha_per_arm not in posterior; skipping")
        return
    samples = _stack_samples(idata, "mu_log_alpha_per_arm")  # (D, 2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(samples[:, 0], bins=40, alpha=0.6, color="#1f77b4", label="low-IF arm")
    ax.hist(samples[:, 1], bins=40, alpha=0.6, color="#d62728", label="high-IF arm")
    ax.set_xlabel("population mean log10(alpha)")
    ax.set_ylabel("posterior density")
    ax.set_title("Per-arm alpha population means")
    ax.legend()
    fig.tight_layout()
    _save(fig, output / "fig01_alpha_per_arm")
    plt.close(fig)


def figure_h3_contrast(idata: az.InferenceData, arm: np.ndarray, output: Path) -> None:
    res = arm_contrast(idata, arm)
    fig, ax = plt.subplots(figsize=(6, 4))
    if "diff_log_alpha" in idata.posterior:
        diffs = _stack_samples(idata, "diff_log_alpha") / np.log(10.0)
    else:
        # Fallback: reconstruct from alpha posterior.
        alpha = _stack_samples(idata, "alpha")
        is_low = arm == "low"
        is_high = arm == "high"
        log_alpha = np.log10(np.maximum(alpha, 1e-12))
        diffs = log_alpha[:, is_low].mean(axis=1) - log_alpha[:, is_high].mean(axis=1)

    ax.hist(diffs, bins=50, color="#666", alpha=0.85)
    ax.axvline(0, color="#999", lw=0.8, ls=":")
    ax.axvline(
        res.diff_log_alpha_mean,
        color="#1f77b4",
        lw=2,
        label=f"mean = {res.diff_log_alpha_mean:+.3f}",
    )
    ax.axvspan(
        res.diff_log_alpha_hdi[0],
        res.diff_log_alpha_hdi[1],
        color="#1f77b4",
        alpha=0.15,
        label="95% HDI",
    )
    ax.set_xlabel("difference in log10(alpha): low - high")
    ax.set_ylabel("count")
    ax.set_title(f"H3 arm contrast (P(diff > 0) = {res.p_predicted_direction:.0%})")
    ax.legend()
    fig.tight_layout()
    _save(fig, output / "fig02_h3_contrast")
    plt.close(fig)


def figure_linkage(idata: az.InferenceData, output: Path) -> None:
    summaries = summarise_subjects(idata, A=10.0, B_anchor=None)
    alpha_means = np.array([s.alpha_mean for s in summaries])
    lambda_implied = np.array([s.lambda_unconstrained_mean for s in summaries])
    res = linkage_correlation(idata, lambda_implied)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Scatter: per-subject alpha mean vs implied lambda.
    ax = axes[0]
    ax.scatter(alpha_means, lambda_implied, color="#444", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("posterior-mean alpha")
    ax.set_ylabel("model-implied lambda")
    ax.set_title("Per-subject linkage")

    # Posterior of r.
    ax = axes[1]
    alpha_post = _stack_samples(idata, "alpha")
    n_draws = alpha_post.shape[0]
    rs = np.empty(n_draws)
    log_l = np.log(np.maximum(lambda_implied, 1e-12))
    for d in range(n_draws):
        log_a = np.log(np.maximum(alpha_post[d], 1e-12))
        a_c = log_a - log_a.mean()
        l_c = log_l - log_l.mean()
        denom = np.sqrt((a_c**2).sum() * (l_c**2).sum())
        rs[d] = float((a_c * l_c).sum() / denom) if denom > 0 else np.nan
    ax.hist(rs[np.isfinite(rs)], bins=40, color="#1f77b4", alpha=0.7)
    ax.axvline(
        res.correlation_mean, color="#d62728", lw=2, label=f"mean = {res.correlation_mean:+.3f}"
    )
    ax.axvspan(
        res.correlation_hdi[0], res.correlation_hdi[1], color="#d62728", alpha=0.15, label="95% HDI"
    )
    ax.set_xlabel("posterior r (log alpha vs log lambda)")
    ax.set_ylabel("count")
    ax.set_title("H1-stronger correlation posterior")
    ax.legend()

    fig.tight_layout()
    _save(fig, output / "fig03_linkage_correlation")
    plt.close(fig)


def figure_ppc_purchase(
    idata: az.InferenceData,
    P: np.ndarray,
    Q_obs: np.ndarray,
    output: Path,
) -> None:
    ppc = posterior_predictive_purchase(idata, P, Q_obs, n_draws=200)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    median = np.median(ppc.replicates, axis=(0, 1))
    lo = np.quantile(ppc.replicates, 0.025, axis=(0, 1))
    hi = np.quantile(ppc.replicates, 0.975, axis=(0, 1))
    for q in Q_obs:
        ax.plot(P, q, "o", color="#444", alpha=0.15)
    ax.plot(P, median, "-", color="#1f77b4", lw=2, label="posterior median")
    ax.fill_between(P, lo, hi, alpha=0.2, color="#1f77b4", label="95% PPC band")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("price (USD)")
    ax.set_ylabel("consumption Q")
    ax.set_title("Purchase task posterior predictive")
    ax.legend()
    fig.tight_layout()
    _save(fig, output / "fig04_ppc_purchase")
    plt.close(fig)


def figure_ppc_sv(
    idata: az.InferenceData,
    E: np.ndarray,
    SV_obs: np.ndarray,
    B_anchor: np.ndarray,
    output: Path,
) -> None:
    ppc = posterior_predictive_sv(idata, E, SV_obs, B_anchor, n_draws=200)
    # Common effort grid for plotting (fractions of B).
    fractions = np.linspace(0.10, 0.85, SV_obs.shape[1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sv in SV_obs:
        ax.plot(fractions, sv, "o", color="#444", alpha=0.15)
    median = np.median(ppc.replicates, axis=(0, 1))
    lo = np.quantile(ppc.replicates, 0.025, axis=(0, 1))
    hi = np.quantile(ppc.replicates, 0.975, axis=(0, 1))
    ax.plot(fractions, median, "-", color="#1f77b4", lw=2, label="posterior median")
    ax.fill_between(fractions, lo, hi, alpha=0.2, color="#1f77b4", label="95% PPC band")
    ax.set_xlabel("effort fraction E / pMax")
    ax.set_ylabel("subjective value SV")
    ax.set_title("Effort discounting posterior predictive")
    ax.legend()
    fig.tight_layout()
    _save(fig, output / "fig05_ppc_sv")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fit", type=Path, required=True, help="Saved fit (.nc).")
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/figures"), help="Figure output dir."
    )
    args = parser.parse_args(argv)

    idata = az.from_netcdf(str(args.fit))
    cohort = load_directory(args.data)
    args.output.mkdir(parents=True, exist_ok=True)

    figure_alpha_per_arm(idata, args.output)
    figure_h3_contrast(idata, cohort.arm, args.output)
    figure_linkage(idata, args.output)
    figure_ppc_purchase(idata, cohort.P, cohort.Q_obs, args.output)
    figure_ppc_sv(idata, cohort.E, cohort.SV_obs, cohort.B_anchor, args.output)
    print(f"\nall figures written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
