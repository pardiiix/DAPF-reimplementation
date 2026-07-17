import argparse
import os
import json
import ast

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from interpretability.metrics import expected_calibration_error
from interpretability.probing import run_linear_probe, run_random_label_probe


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--interpretability_pt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_bins", type=int, default=10)

    return parser.parse_args()


def normalize_labels(labels):
    label_map = {
        "healthy": 0,
        "control": 0,
        "0": 0,
        0: 0,
        0.0: 0,
        "dementia": 1,
        "ad": 1,
        "1": 1,
        1: 1,
        1.0: 1,
    }

    out = []
    for label in labels:
        if isinstance(label, str):
            key = label.strip().lower()
        else:
            key = label

        if key not in label_map:
            raise ValueError(f"Unknown label value: {label}")

        out.append(label_map[key])

    return np.asarray(out, dtype=int)


def compute_prediction_metrics(df, n_bins=10):
    y_true = normalize_labels(df["labels"].values)
    y_pred = normalize_labels(df["pred_labels"].values)
    y_prob = df["probas"].astype(float).values

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
    }

    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auroc"] = np.nan

    return metrics


def save_metrics(metrics, output_path):
    df = pd.DataFrame([metrics])
    df.to_csv(output_path, index=False)


def run_representation_probes(df, interp_obj, output_dir):
    y_true = normalize_labels(df["labels"].values)

    probe_rows = []

    if "cls_hidden_states" in interp_obj:
        X_cls = interp_obj["cls_hidden_states"].detach().cpu().numpy()

        # Original diagnosis probe
        cls_probe = run_linear_probe(
            X_cls,
            y_true,
            scoring="f1_macro",
        )

        probe_rows.append({
            "representation": "cls_hidden_states",
            "target": "diagnosis",
            "score_mean": cls_probe["mean"],
            "score_std": cls_probe["std"],
            "ci_95_low": np.nan,
            "ci_95_high": np.nan,
            "n_repeats": np.nan,
            "scores": json.dumps(cls_probe["scores"]),
        })

        # New random-label control
        cls_random_probe = run_random_label_probe(
            X_cls,
            y_true,
            n_repeats=100,
            random_seed=42,
        )

        probe_rows.append({
            "representation": "cls_hidden_states",
            "target": "random_labels",
            "score_mean": cls_random_probe["mean_macro_f1"],
            "score_std": cls_random_probe["std_macro_f1"],
            "ci_95_low": cls_random_probe["ci_95_low"],
            "ci_95_high": cls_random_probe["ci_95_high"],
            "n_repeats": cls_random_probe["n_valid_repeats"],
            "scores": json.dumps(cls_random_probe["all_scores"]),
        })

    if "mask_hidden_states" in interp_obj:
        X_mask = interp_obj["mask_hidden_states"].detach().cpu().numpy()

        # Original diagnosis probe
        mask_probe = run_linear_probe(
            X_mask,
            y_true,
            scoring="f1_macro",
        )

        probe_rows.append({
            "representation": "mask_hidden_states",
            "target": "diagnosis",
            "score_mean": mask_probe["mean"],
            "score_std": mask_probe["std"],
            "ci_95_low": np.nan,
            "ci_95_high": np.nan,
            "n_repeats": np.nan,
            "scores": json.dumps(mask_probe["scores"]),
        })

        # New random-label control
        mask_random_probe = run_random_label_probe(
            X_mask,
            y_true,
            n_repeats=100,
            random_seed=42,
        )

        probe_rows.append({
            "representation": "mask_hidden_states",
            "target": "random_labels",
            "score_mean": mask_random_probe["mean_macro_f1"],
            "score_std": mask_random_probe["std_macro_f1"],
            "ci_95_low": mask_random_probe["ci_95_low"],
            "ci_95_high": mask_random_probe["ci_95_high"],
            "n_repeats": mask_random_probe["n_valid_repeats"],
            "scores": json.dumps(mask_random_probe["all_scores"]),
        })

    if probe_rows:
        probe_df = pd.DataFrame(probe_rows)

        probe_df.to_csv(
            os.path.join(
                output_dir,
                "representation_probe_results.csv",
            ),
            index=False,
        )

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    interp_obj = torch.load(args.interpretability_pt, map_location="cpu")

    if len(df) != len(interp_obj["ids"]):
        raise ValueError(
            f"Mismatch: results CSV has {len(df)} rows, "
            f"but interpretability file has {len(interp_obj['ids'])} ids."
        )

    prediction_metrics = compute_prediction_metrics(df, n_bins=args.n_bins)

    save_metrics(
        prediction_metrics,
        os.path.join(args.output_dir, "prediction_metrics.csv")
    )

    run_representation_probes(
        df=df,
        interp_obj=interp_obj,
        output_dir=args.output_dir
    )

    print("Saved outputs to:", args.output_dir)
    print("Prediction metrics:")
    for key, value in prediction_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()