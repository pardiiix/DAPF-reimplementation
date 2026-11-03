#!/usr/bin/env python3
"""Study 2: seed-level significance of the DAPF MASK-vs-CLS backbone interaction.

Runs on existing per-seed representation_probe_results.csv files. NO model
retraining or participant-level inference is performed.

Primary contrast: (MASK - CLS)_BERT - (MASK - CLS)_RoBERTa.
Secondary contrasts: MASK - CLS within each backbone.

Statistical unit: a matched *training seed*, not a test participant. The same
48 participants appear in each seed. An exact two-sided sign-flip test with
five seeds has minimum p = 0.0625; lack of statistical significance does NOT
establish equivalence or clinical unsuitability.

For a more informative participant-level analysis, use aligned OOF probe
predictions or rerun probes from saved hidden states, preserving participant
pairing and seed dependence. Do not treat 5 folds x 5 seeds as 25 independent
observations.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def read_score(path: Path, representation: str) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    required = {"representation", "target", "score_mean"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: expected columns {sorted(required)}; got {list(df.columns)}")
    matched = df.loc[(df["representation"] == representation) & (df["target"] == "diagnosis"), "score_mean"]
    if len(matched) != 1:
        raise ValueError(f"{path}: expected exactly one diagnosis row for {representation}, found {len(matched)}")
    value = float(matched.iloc[0])
    if not (np.isfinite(value) and 0 <= value <= 1):
        raise ValueError(f"{path}: invalid diagnosis macro-F1 {value}")
    return value


def exact_sign_flip_p(differences: np.ndarray) -> float:
    """Two-sided test of paired seed-level differences centered at zero."""
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(differences.mean()))
    extreme = 0
    for signs in product((-1, 1), repeat=len(differences)):
        candidate = abs(float(np.mean(differences * np.asarray(signs))))
        extreme += candidate >= observed - 1e-12
    return extreme / (2 ** len(differences))


def ci_t(differences: np.ndarray) -> tuple[float, float]:
    n = len(differences)
    if n < 2:
        return (float("nan"), float("nan"))
    center = float(np.mean(differences))
    halfwidth = float(student_t.ppf(.975, df=n - 1) * np.std(differences, ddof=1) / np.sqrt(n))
    return (center - halfwidth, center + halfwidth)


def ci_bootstrap(differences: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    n = len(differences)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return (float(low), float(high))


def holm_adjust(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    out = np.zeros(len(ps), dtype=float)
    running_max = 0.
    for rank, idx in enumerate(order):
        running_max = max(running_max, (len(ps) - rank) * ps[idx])
        out[idx] = min(1.0, running_max)
    return list(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bert_root", default="output_interpretability_multiseed")
    parser.add_argument("--roberta_root", default="output_roberta_interpretability")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output_dir", default="study2_significance/representation_seed_level")
    parser.add_argument("--n_bootstrap", type=int, default=20000)
    parser.add_argument("--random_state", type=int, default=2026)
    args = parser.parse_args()
    seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        parser.error("Supply at least two distinct training seed IDs")
    if args.n_bootstrap < 1000:
        parser.error("--n_bootstrap must be >= 1000")

    rows = []
    for seed in seeds:
        b_path = Path(args.bert_root) / f"seed_{seed}" / "representation_probe_results.csv"
        r_path = Path(args.roberta_root) / f"seed_{seed}" / "representation_probe_results.csv"
        bc = read_score(b_path, "cls_hidden_states")
        bm = read_score(b_path, "mask_hidden_states")
        rc = read_score(r_path, "cls_hidden_states")
        rm = read_score(r_path, "mask_hidden_states")
        rows.append({
            "seed": seed, "bert_cls": bc, "bert_mask": bm,
            "roberta_cls": rc, "roberta_mask": rm,
            "bert_gap_mask_minus_cls": bm - bc,
            "roberta_gap_mask_minus_cls": rm - rc,
            "interaction_bert_gap_minus_roberta_gap": (bm - bc) - (rm - rc),
        })
    df = pd.DataFrame(rows).sort_values("seed")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_seed_representation_gaps.csv", index=False)

    rng = np.random.default_rng(args.random_state)
    contrasts = [
        ("primary_backbone_by_representation_interaction", "interaction_bert_gap_minus_roberta_gap"),
        ("secondary_bert_mask_minus_cls", "bert_gap_mask_minus_cls"),
        ("secondary_roberta_mask_minus_cls", "roberta_gap_mask_minus_cls"),
    ]
    summary = []
    for label, col in contrasts:
        d = df[col].to_numpy(dtype=float)
        t_low, t_high = ci_t(d)
        b_low, b_high = ci_bootstrap(d, args.n_bootstrap, rng)
        summary.append({
            "contrast": label, "n_seed_pairs": len(d),
            "mean_difference": float(d.mean()),
            "sd_seed_differences": float(d.std(ddof=1)),
            "t_95_ci_low": t_low, "t_95_ci_high": t_high,
            "bootstrap_seed_95_ci_low_exploratory": b_low,
            "bootstrap_seed_95_ci_high_exploratory": b_high,
            "exact_two_sided_sign_flip_p": exact_sign_flip_p(d),
            "n_sign_assignments": 2 ** len(d),
        })
    adjusted = holm_adjust([r["exact_two_sided_sign_flip_p"] for r in summary])
    for row, adj in zip(summary, adjusted):
        row["holm_p_across_three_contrasts"] = adj
    results = pd.DataFrame(summary)
    results.to_csv(out / "representation_significance.csv", index=False)
    print("\nPER-SEED SCORES AND PAIRED GAPS\n")
    print(df.round(5).to_string(index=False))
    print("\nSIGNIFICANCE SUMMARY\n")
    print(results.round(5).to_string(index=False))
    print(f"\nSaved to: {out}")
    print("\nINTERPRETATION: Exploratory seed-level inference only. Five seeds cannot yield")
    print("p < .05 in an exact two-sided sign-flip test. CIs are based on only five seeds;")
    print("the bootstrap does not quantify uncertainty from sampling new participants.")
    print("A non-significant result is not evidence of equivalence or clinical failure.")


if __name__ == "__main__":
    main()
