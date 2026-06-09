import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= 0.5).astype(int)
    confidence = np.maximum(y_prob, 1.0 - y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)

        if not np.any(mask):
            continue

        bin_accuracy = np.mean(y_pred[mask] == y_true[mask])
        bin_confidence = np.mean(confidence[mask])
        bin_weight = np.mean(mask)

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def find_one(pattern):
    matches = glob.glob(pattern, recursive=True)
    matches = sorted(matches)

    if len(matches) == 0:
        return None

    return matches[-1]


def read_probe_metric(path, representation_keyword):
    if not os.path.exists(path):
        return np.nan

    df = pd.read_csv(path)

    rep_col = "representation"
    score_col = "score_mean"

    if rep_col not in df.columns or score_col not in df.columns:
        return np.nan

    sub = df[df[rep_col].astype(str).str.lower().str.contains(representation_keyword.lower())]

    if sub.empty:
        return np.nan

    return float(sub.iloc[0][score_col])


def summarize_scalar_metrics(args):
    rows = []

    for seed in args.seeds:
        seed_train_root = os.path.join(args.train_root, f"seed_{seed}")
        seed_interp_root = os.path.join(args.interp_root, f"seed_{seed}")

        results_csv = find_one(
            os.path.join(seed_train_root, "**", "checkpoints", "epoch-1", "test_results.csv")
        )

        if results_csv is None:
            print(f"Warning: missing test_results.csv for seed {seed}")
            continue

        df = pd.read_csv(results_csv)

        y_true = df["labels"].astype(int).to_numpy()
        y_pred = df["pred_labels"].astype(int).to_numpy()
        y_prob = df["probas"].astype(float).to_numpy()

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        auroc = roc_auc_score(y_true, y_prob)
        ece = expected_calibration_error(y_true, y_prob, n_bins=args.n_bins)

        probe_path = os.path.join(seed_interp_root, "representation_probe_results.csv")

        cls_probe_f1 = read_probe_metric(probe_path, "cls")
        mask_probe_f1 = read_probe_metric(probe_path, "mask")

        rows.extend([
            {"seed": seed, "metric": "accuracy", "value": accuracy},
            {"seed": seed, "metric": "macro_f1", "value": macro_f1},
            {"seed": seed, "metric": "auroc", "value": auroc},
            {"seed": seed, "metric": "ece", "value": ece},
            {"seed": seed, "metric": "cls_probe_f1", "value": cls_probe_f1},
            {"seed": seed, "metric": "mask_probe_f1", "value": mask_probe_f1},
        ])

    all_df = pd.DataFrame(rows)

    summary_df = (
        all_df.groupby("metric")
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    summary_df["mean_pm_std"] = summary_df.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}",
        axis=1,
    )

    return all_df, summary_df


def summarize_faithfulness(args):
    rows = []

    for seed in args.seeds:
        seed_interp_root = os.path.join(args.interp_root, f"seed_{seed}")

        for ranking in ["positive", "absolute"]:
            faith_dir = os.path.join(seed_interp_root, f"faithfulness_{ranking}")

            deletion_path = os.path.join(faith_dir, "deletion_curve.csv")
            insertion_path = os.path.join(faith_dir, "insertion_curve.csv")

            if os.path.exists(deletion_path):
                deletion = pd.read_csv(deletion_path)

                grouped = (
                    deletion.groupby("fraction")["probability_drop"]
                    .mean()
                    .reset_index()
                )

                for _, r in grouped.iterrows():
                    rows.append({
                        "seed": seed,
                        "ranking": ranking,
                        "test": "deletion",
                        "fraction": float(r["fraction"]),
                        "metric": "probability_drop",
                        "value": float(r["probability_drop"]),
                    })

            if os.path.exists(insertion_path):
                insertion = pd.read_csv(insertion_path)

                grouped = (
                    insertion.groupby("fraction")["probability_recovered"]
                    .mean()
                    .reset_index()
                )

                for _, r in grouped.iterrows():
                    rows.append({
                        "seed": seed,
                        "ranking": ranking,
                        "test": "insertion",
                        "fraction": float(r["fraction"]),
                        "metric": "probability_recovered",
                        "value": float(r["probability_recovered"]),
                    })

    all_df = pd.DataFrame(rows)

    if all_df.empty:
        summary_df = pd.DataFrame()
        return all_df, summary_df

    summary_df = (
        all_df.groupby(["ranking", "test", "fraction", "metric"])
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    summary_df["mean_pm_std"] = summary_df.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}",
        axis=1,
    )

    return all_df, summary_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_root", default="./output_multiseed")
    parser.add_argument("--interp_root", default="./output_interpretability_multiseed")
    parser.add_argument("--output_dir", default="./output_interpretability_multiseed_summary")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--n_bins", type=int, default=10)

    args = parser.parse_args()

    args.seeds = [int(x) for x in args.seeds.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)

    scalar_all, scalar_summary = summarize_scalar_metrics(args)
    faith_all, faith_summary = summarize_faithfulness(args)

    scalar_all_path = os.path.join(args.output_dir, "all_seed_scalar_metrics.csv")
    scalar_summary_path = os.path.join(args.output_dir, "multiseed_scalar_summary.csv")
    faith_all_path = os.path.join(args.output_dir, "all_seed_faithfulness_by_fraction.csv")
    faith_summary_path = os.path.join(args.output_dir, "multiseed_faithfulness_summary_by_fraction.csv")

    scalar_all.to_csv(scalar_all_path, index=False)
    scalar_summary.to_csv(scalar_summary_path, index=False)
    faith_all.to_csv(faith_all_path, index=False)
    faith_summary.to_csv(faith_summary_path, index=False)

    print("\nSaved:")
    print(scalar_all_path)
    print(scalar_summary_path)
    print(faith_all_path)
    print(faith_summary_path)

    print("\nScalar metric summary:")
    print(scalar_summary[["metric", "mean_pm_std", "n_seeds"]].to_string(index=False))

    if not faith_summary.empty:
        print("\nFaithfulness summary:")
        print(
            faith_summary[
                ["ranking", "test", "fraction", "metric", "mean_pm_std", "n_seeds"]
            ].to_string(index=False)
        )
    else:
        print("\nNo faithfulness results found.")


if __name__ == "__main__":
    main()