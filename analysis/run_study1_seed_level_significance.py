#!/usr/bin/env python3
"""
Fast seed-level significance testing for Table 3.

This script DOES NOT rerun participant-level permutation tests.
It reads the per-seed metrics CSV already produced by
analysis.run_performance_significance and performs exact paired
sign-flip/randomization tests across the five matched seeds.

Example:
    python analysis/run_seed_level_significance.py \
        --metrics_csv ./performance_significance/per_seed_metrics.csv \
        --output_csv ./performance_significance/seed_level_significance.csv
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metrics_csv",
        default="./performance_significance/per_seed_metrics.csv",
        help="CSV produced by the earlier performance significance script.",
    )
    p.add_argument(
        "--output_csv",
        default="./performance_significance/seed_level_significance.csv",
        help="Where to save the seed-level significance results.",
    )
    p.add_argument(
        "--metrics",
        default="accuracy,macro_f1",
        help="Comma-separated metrics to test. Default: accuracy,macro_f1",
    )
    return p.parse_args()


def exact_paired_sign_flip(differences):
    """
    Exact two-sided paired sign-flip/randomization test.

    Under H0, the sign of each paired seed-level difference is exchangeable.
    For n seeds, all 2^n sign assignments are enumerated exactly.

    Test statistic: absolute mean paired difference.
    """
    differences = np.asarray(differences, dtype=float)

    if differences.ndim != 1:
        raise ValueError("differences must be one-dimensional")
    if len(differences) == 0:
        raise ValueError("No paired differences supplied")

    observed_mean = float(np.mean(differences))
    observed_stat = abs(observed_mean)

    permuted_stats = []
    for signs in product([-1.0, 1.0], repeat=len(differences)):
        signs = np.asarray(signs, dtype=float)
        permuted_mean = np.mean(differences * signs)
        permuted_stats.append(abs(permuted_mean))

    permuted_stats = np.asarray(permuted_stats, dtype=float)

    # Exact p-value: fraction of all sign assignments at least as extreme
    p_value = float(
        np.mean(permuted_stats >= observed_stat - 1e-12)
    )

    return observed_mean, p_value, len(permuted_stats)


def holm_adjust(pvalues):
    """Holm family-wise error-rate adjustment."""
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)

    order = np.argsort(pvalues)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def main():
    args = parse_args()

    metrics_path = Path(args.metrics_csv)
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Could not find {metrics_path}\n"
            "Run the earlier significance script once, or point "
            "--metrics_csv to its per_seed_metrics.csv file."
        )

    df = pd.read_csv(metrics_path)

    required = {"model", "seed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for metric in metrics:
        if metric not in df.columns:
            raise ValueError(
                f"Metric '{metric}' not found. Available columns: {list(df.columns)}"
            )

    comparisons = [
        ("DAPF", "BERT-CLS"),
        ("DAPF", "BERT-CLS+Prompt"),
    ]

    rows = []

    for model_a, model_b in comparisons:
        a = df[df["model"] == model_a].set_index("seed").sort_index()
        b = df[df["model"] == model_b].set_index("seed").sort_index()

        common_seeds = sorted(set(a.index).intersection(b.index))
        if not common_seeds:
            raise ValueError(f"No matched seeds for {model_a} vs {model_b}")

        print("\n" + "=" * 72)
        print(f"{model_a} vs {model_b}")
        print(f"Matched seeds: {common_seeds}")

        for metric in metrics:
            vals_a = a.loc[common_seeds, metric].to_numpy(dtype=float)
            vals_b = b.loc[common_seeds, metric].to_numpy(dtype=float)
            diffs = vals_a - vals_b

            mean_diff, p_value, n_assignments = exact_paired_sign_flip(diffs)

            row = {
                "comparison": f"{model_a} vs {model_b}",
                "metric": metric,
                "n_seeds": len(common_seeds),
                "model_a_mean": float(np.mean(vals_a)),
                "model_b_mean": float(np.mean(vals_b)),
                "mean_difference": mean_diff,
                "std_difference": float(np.std(diffs, ddof=1))
                    if len(diffs) > 1 else np.nan,
                "exact_sign_flip_p": p_value,
                "n_sign_assignments": n_assignments,
                "seed_differences": repr(diffs.tolist()),
            }
            rows.append(row)

            print(f"\n{metric}")
            print(f"  {model_a} mean: {np.mean(vals_a):.6f}")
            print(f"  {model_b} mean: {np.mean(vals_b):.6f}")
            print(f"  mean difference: {mean_diff:.6f}")
            print(f"  differences by seed: {diffs.tolist()}")
            print(f"  exact two-sided sign-flip p: {p_value:.6f}")
            print(f"  sign assignments enumerated: {n_assignments}")

    results = pd.DataFrame(rows)

    # Also provide Holm-adjusted values across the tests in this file.
    # Raw exact p-values remain available in exact_sign_flip_p.
    results["p_holm_all_tests"] = holm_adjust(
        results["exact_sign_flip_p"].to_numpy()
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(
        results[
            [
                "comparison",
                "metric",
                "mean_difference",
                "exact_sign_flip_p",
                "p_holm_all_tests",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
