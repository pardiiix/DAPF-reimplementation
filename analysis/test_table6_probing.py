#!/usr/bin/env python3
"""Paired seed-level tests for representation-probing scores.

Default use: compare DAPF final-layer [MASK] and [CLS] macro-F1 across seeds.
Input files are representation_probe_results.csv under seed directories.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def find_probe_file(root: str, seed: int, pattern: str) -> str:
    rendered = pattern.format(seed=seed)
    matches = sorted(glob.glob(str(Path(root) / rendered), recursive=True))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Seed {seed}: expected exactly one probe file for {Path(root) / rendered}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def read_score(path: str, representation: str) -> float:
    df = pd.read_csv(path)

    required = {"representation", "target", "score_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")

    target_representation = representation.strip().lower()

    matched = df[
        (df["representation"].astype(str).str.strip().str.lower()
         == target_representation)
        &
        (df["target"].astype(str).str.strip().str.lower()
         == "diagnosis")
    ]

    if len(matched) != 1:
        available = df[
            ["representation", "target", "score_mean"]
        ].to_dict("records")

        raise ValueError(
            f"{path}: expected exactly one diagnosis row for "
            f"{representation!r}; found {len(matched)}. "
            f"Available rows: {available}"
        )

    return float(matched.iloc[0]["score_mean"])

def sign_flip_permutation_p(differences: np.ndarray, alternative: str) -> float:
    """Exact paired randomization test over all 2^n sign flips."""
    observed = float(np.mean(differences))
    n = len(differences)
    statistics = []
    for mask in range(1 << n):
        signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
        statistics.append(float(np.mean(differences * signs)))
    stats = np.asarray(statistics)
    if alternative == "greater":
        return float(np.mean(stats >= observed - 1e-15))
    return float(np.mean(np.abs(stats) >= abs(observed) - 1e-15))


def bootstrap_ci(differences: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(differences)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = np.mean(rng.choice(differences, size=n, replace=True))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def safe_wilcoxon(x: np.ndarray, y: np.ndarray, alternative: str) -> float:
    differences = x - y
    if np.allclose(differences, 0):
        return 1.0
    return float(wilcoxon(x, y, alternative=alternative, method="exact").pvalue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interp_root", required=True)
    parser.add_argument(
        "--pattern",
        default="seed_{seed}/representation_probe_results.csv",
        help="Path under interp_root; must contain {seed}. Recursive ** is allowed.",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--a", default="mask", help="Representation expected to score higher")
    parser.add_argument("--b", default="cls", help="Comparator representation")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--output_dir", default="table6_probing_significance")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for seed in seeds:
        path = find_probe_file(args.interp_root, seed, args.pattern)
        score_a = read_score(path, args.a)
        score_b = read_score(path, args.b)
        rows.append({
            "seed": seed,
            "probe_file": path,
            "representation_a": args.a,
            "score_a": score_a,
            "representation_b": args.b,
            "score_b": score_b,
            "difference_a_minus_b": score_a - score_b,
        })

    details = pd.DataFrame(rows).sort_values("seed")
    a = details["score_a"].to_numpy(float)
    b = details["score_b"].to_numpy(float)
    d = a - b

    ci_lo, ci_hi = bootstrap_ci(d, args.n_boot, args.random_seed)
    t_two = float(ttest_rel(a, b, alternative="two-sided").pvalue)
    t_greater = float(ttest_rel(a, b, alternative="greater").pvalue)

    summary = {
        "representation_a": args.a,
        "representation_b": args.b,
        "n_seed_pairs": len(d),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_paired_difference_a_minus_b": float(np.mean(d)),
        "std_paired_difference": float(np.std(d, ddof=1)) if len(d) > 1 else np.nan,
        "wins_a": int(np.sum(d > 0)),
        "ties": int(np.sum(np.isclose(d, 0))),
        "wins_b": int(np.sum(d < 0)),
        "bootstrap_95_ci_low": ci_lo,
        "bootstrap_95_ci_high": ci_hi,
        "wilcoxon_exact_two_sided_p": safe_wilcoxon(a, b, "two-sided"),
        "wilcoxon_exact_directional_a_greater_p": safe_wilcoxon(a, b, "greater"),
        "paired_t_two_sided_p": t_two,
        "paired_t_directional_a_greater_p": t_greater,
        "sign_flip_exact_two_sided_p": sign_flip_permutation_p(d, "two-sided"),
        "sign_flip_exact_directional_a_greater_p": sign_flip_permutation_p(d, "greater"),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "paired_seed_scores.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "probing_significance_results.csv", index=False)
    with (output_dir / "probing_significance_results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPaired seed scores")
    print(details[["seed", "score_a", "score_b", "difference_a_minus_b"]].to_string(index=False))
    print("\nSummary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
