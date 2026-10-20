#!/usr/bin/env python3
"""Plot RoBERTa faithfulness, stratified by true diagnosis.

Reads saved per-participant faithfulness curves from five seeds for
DAPF-RoBERTa, RoBERTa-CLS, and RoBERTa-CLS+Prompt.

Outputs a 2x2 figure (diagnosis group x deletion/insertion), seed-level
and cross-seed summary CSVs, and an intervention-count audit CSV.

Each point is the mean of five seed-wise participant-group means; error bars
are sample SD of those five seed means, NOT SD across 120 individuals.
No model training, attribution, or inference occurs.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ("DAPF-RoBERTa", "RoBERTa-CLS", "RoBERTa-CLS+Prompt")
LABELS = {0: "Healthy control", 1: "Dementia"}
METHODS = {"deletion": "probability_drop", "insertion": "probability_recovered"}
FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)
SEEDS = (0, 1, 2, 3, 4)
COLORS = {"DAPF-RoBERTa": "#256da5", "RoBERTa-CLS": "#db7818", "RoBERTa-CLS+Prompt": "#398c48"}
MARKERS = {"DAPF-RoBERTa": "o", "RoBERTa-CLS": "s", "RoBERTa-CLS+Prompt": "^"}


def load_one(path, model, seed, ranking, method):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {model} seed {seed}: {path}")
    df = pd.read_csv(path)
    metric = METHODS[method]
    required = {"id", "label", "pred_label", "fraction", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    df = df.copy()
    for col in ("label", "pred_label", "fraction", metric):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df[list(required)].isna().any().any():
        raise ValueError(f"{path}: missing values in required columns")
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path}: unexpected label codes")
    if not set(df["pred_label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path}: unexpected predicted label codes")
    if df.duplicated(["id", "fraction"]).any():
        raise ValueError(f"{path}: duplicate (id, fraction) rows")
    if not np.isfinite(df[metric].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: nonfinite probability effect")
    ids = df[["id", "label", "pred_label"]].drop_duplicates()
    if len(ids) != 48 or ids["id"].nunique() != 48:
        raise ValueError(f"{path}: expected 48 distinct participants, found {ids['id'].nunique()}")
    class_counts = ids["label"].value_counts().to_dict()
    if class_counts != {0: 24, 1: 24}:
        raise ValueError(f"{path}: expected 24 healthy and 24 dementia: {class_counts}")
    unique_fractions = sorted(df["fraction"].unique())
    if len(unique_fractions) != len(FRACTIONS) or not np.allclose(unique_fractions, FRACTIONS, atol=1e-6):
        raise ValueError(f"{path}: unexpected fractions {unique_fractions}")
    if len(df) != 48 * len(FRACTIONS):
        raise ValueError(f"{path}: expected 240 rows, got {len(df)}")
    group_sizes = df.groupby(["fraction", "label"])["id"].nunique()
    if not (group_sizes == 24).all():
        raise ValueError(f"{path}: some fractions/classes are incomplete")

    # Check that the same individuals have the same truth and prediction
    # at each fraction. Across seeds the model prediction may change.
    if df.groupby("id")[["label", "pred_label"]].nunique().gt(1).any().any():
        raise ValueError(f"{path}: label/prediction varies across fractions")

    means = (
        df.groupby(["label", "fraction"], as_index=False)[metric]
        .agg(mean_effect="mean", n_participants="count")
    )
    means["model"] = model
    means["seed"] = seed
    means["method"] = method
    means["ranking"] = ranking

    # The fraction columns have different counting conventions across the
    # original pipelines. Preserve actual intervention counts for auditing.
    if method == "deletion":
        count_col = next((col for col in ("num_masked", "n_selected_tokens") if col in df.columns), None)
    else:
        count_col = next((col for col in ("num_inserted", "n_selected_tokens") if col in df.columns), None)
    denom_col = next((col for col in ("num_positions_available", "n_ranked_tokens") if col in df.columns), None)
    reportable_col = "n_reportable_tokens" if "n_reportable_tokens" in df.columns else None
    if count_col is None:
        raise ValueError(f"{path}: cannot find actual number of perturbed tokens")
    audit_cols = ["id", "label", "fraction", count_col] + ([denom_col] if denom_col else []) + ([reportable_col] if reportable_col else [])
    audit = df[audit_cols].copy()
    audit = audit.rename(columns={count_col: "n_changed", **({denom_col: "n_ranked_available"} if denom_col else {}), **({reportable_col: "n_reportable_tokens"} if reportable_col else {})})
    if "n_ranked_available" not in audit.columns:
        audit["n_ranked_available"] = np.nan
    if "n_reportable_tokens" not in audit.columns:
        audit["n_reportable_tokens"] = np.nan
    if (audit["n_changed"] < 0).any():
        raise ValueError(f"{path}: negative token counts")
    if (audit["n_changed"] > audit["n_ranked_available"]).any():
        raise ValueError(f"{path}: changed count exceeds available ranked tokens")
    audit["model"] = model
    audit["seed"] = seed
    audit["method"] = method
    audit["ranking"] = ranking
    print(f"Loaded: {model}, seed {seed}, {ranking}/{method}: {len(df)} rows, 24 + 24 participants")
    return means, audit


def summarize(rows):
    summary = (
        rows.groupby(["ranking", "method", "model", "label", "fraction"], as_index=False)["mean_effect"]
        .agg(mean_effect="mean", sd_across_seeds="std", n_seeds="count")
    )
    if not (summary["n_seeds"] == 5).all():
        raise ValueError("Missing seed-level values in at least one model/diagnosis/fraction")
    return summary


def make_figure(summary, ranking, output):
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.7), sharex=True, sharey="col")
    for i, label in enumerate((0, 1)):
        for j, method in enumerate(("deletion", "insertion")):
            ax = axes[i, j]
            for model in MODELS:
                selected = summary.loc[
                    (summary["ranking"] == ranking)
                    & (summary["method"] == method)
                    & (summary["label"] == label)
                    & (summary["model"] == model)
                ].sort_values("fraction")
                if len(selected) != len(FRACTIONS):
                    raise ValueError(f"Incomplete chart data for {ranking}/{method}/{model}/{label}")
                ax.errorbar(
                    selected["fraction"].to_numpy() * 100,
                    selected["mean_effect"].to_numpy(),
                    yerr=selected["sd_across_seeds"].to_numpy(),
                    label=model, color=COLORS[model], marker=MARKERS[model],
                    linewidth=1.8, markersize=4.8, elinewidth=1.0, capsize=3,
                )
            ax.axhline(0, color="#777777", linewidth=1, linestyle=":")
            ax.set_title(f"{LABELS[label]} — {method.capitalize()}")
            ax.set_xticks([5, 10, 20, 30, 50])
            ax.grid(axis="y", alpha=0.18)
            ax.set_axisbelow(True)
            if i == 1:
                ax.set_xlabel("Requested fraction of ranked tokens (%)")
            if j == 0:
                ax.set_ylabel("Original − deleted P(dementia)")
            else:
                ax.set_ylabel("Restored − baseline P(dementia)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"RoBERTa faithfulness by true diagnosis — {ranking} IG (five-seed mean ± SD)", fontsize=14)
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    stem = f"roberta_faithfulness_{ranking}_by_diagnosis"
    for ext in ("png", "pdf"):
        dest = output / f"{stem}.{ext}"
        fig.savefig(dest, dpi=300, bbox_inches="tight")
        print(f"Saved {dest}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dapf-root", default="./output_roberta_interpretability")
    parser.add_argument("--cls-root", default="./output_roberta_cls_interpretability_multiseed")
    parser.add_argument("--prompt-root", default="./output_roberta_cls_prompt_interpretability_multiseed")
    parser.add_argument("--out-dir", default="./output_roberta_comparison/figures")
    parser.add_argument("--ranking", choices=("positive", "absolute", "both"), default="positive")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rankings = ("positive", "absolute") if args.ranking == "both" else (args.ranking,)
    roots = (Path(args.dapf_root), Path(args.cls_root), Path(args.prompt_root))
    pieces, audits = [], []
    for model, root in zip(MODELS, roots):
        for seed in SEEDS:
            for ranking in rankings:
                for method in METHODS:
                    path = root / f"seed_{seed}" / f"faithfulness_{ranking}" / f"{method}_curve.csv"
                    means, audit = load_one(path, model, seed, ranking, method)
                    pieces.append(means)
                    audits.append(audit)
    seed_means = pd.concat(pieces, ignore_index=True)
    intervention_audit = pd.concat(audits, ignore_index=True)
    summary = summarize(seed_means)
    seed_means.to_csv(out / "roberta_faithfulness_diagnosis_seed_means.csv", index=False)
    summary.to_csv(out / "roberta_faithfulness_diagnosis_mean_sd.csv", index=False)
    intervention_summary = intervention_audit.groupby(["ranking", "method", "model", "label", "fraction"], as_index=False).agg(
        mean_tokens_changed=("n_changed", "mean"),
        mean_ranked_tokens_available=("n_ranked_available", "mean"),
        mean_reportable_tokens=("n_reportable_tokens", "mean"),
    )
    intervention_summary.to_csv(out / "roberta_faithfulness_intervention_counts.csv", index=False)
    for ranking in rankings:
        make_figure(summary, ranking, out)
    print("\nSelected mean effects at 20% and 50% (signed P(dementia) differences):")
    print(summary.loc[summary["fraction"].isin((0.2, 0.5)), ["model", "label", "method", "fraction", "mean_effect", "sd_across_seeds"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nIntervention counts at 20%: important for model-to-model comparability")
    print(intervention_summary.loc[(intervention_summary["fraction"] == 0.2) & (intervention_summary["ranking"] == "positive"), ["model", "label", "method", "mean_tokens_changed", "mean_ranked_tokens_available", "mean_reportable_tokens"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nCOMPLETE. These are descriptive probability effects, not diagnosis-label-specific correctness scores.")


if __name__ == "__main__":
    main()
