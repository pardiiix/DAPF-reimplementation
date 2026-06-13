"""
Integrated Gradients attribution for BERT-CLS+Prompt.

The model input includes a prompt and [MASK], but prediction is through
the [CLS] classification head. [MASK] is excluded from cleaned token
attributions because it is part of the prompt scaffold.
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except Exception:
        return ast.literal_eval(value)


def compute_ece(y_true, y_prob_dementia, n_bins=10):
    y_pred = (y_prob_dementia >= 0.5).astype(int)
    confidence = np.maximum(y_prob_dementia, 1.0 - y_prob_dementia)
    correctness = (y_pred == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        if end == 1.0:
            mask = (confidence >= start) & (confidence <= end)
        else:
            mask = (confidence >= start) & (confidence < end)

        if not np.any(mask):
            continue

        ece += mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean())

    return float(ece)


def get_probability_dementia(df):
    if "probability_dementia" in df.columns:
        return pd.to_numeric(df["probability_dementia"], errors="coerce").to_numpy()

    probs = np.asarray(df["probas"].apply(parse_list).tolist(), dtype=float)
    return probs[:, 1]


def compute_prediction_metrics(df, n_bins=10):
    y_true = pd.to_numeric(df["labels"], errors="coerce").to_numpy(dtype=int)
    y_pred = pd.to_numeric(df["pred_labels"], errors="coerce").to_numpy(dtype=int)
    y_prob = get_probability_dementia(df)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "ece": compute_ece(y_true, y_prob, n_bins=n_bins),
    }

    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auroc"] = float("nan")

    return metrics


def tensor_to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def clean_nan_rows(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    keep = ~np.isnan(X).any(axis=1)
    return X[keep], y[keep]


def run_probe(X, y, n_splits=5, random_state=42):
    X, y = clean_nan_rows(X, y)

    if len(X) == 0:
        return {"mean_score": float("nan"), "std_score": float("nan"), "fold_scores": [], "n_splits": 0}

    _, counts = np.unique(y, return_counts=True)
    max_splits = int(counts.min())

    if max_splits < 2:
        return {"mean_score": float("nan"), "std_score": float("nan"), "fold_scores": [], "n_splits": 0}

    actual_splits = min(n_splits, max_splits)

    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    scores = []

    for train_idx, test_idx in cv.split(X, y):
        probe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
            ]
        )
        probe.fit(X[train_idx], y[train_idx])
        pred = probe.predict(X[test_idx])
        scores.append(float(f1_score(y[test_idx], pred, average="macro", zero_division=0)))

    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "fold_scores": scores,
        "n_splits": actual_splits,
    }


def main():
    parser = argparse.ArgumentParser(description="Run interpretability for BERT-CLS+Prompt.")
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--interpretability_pt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    y = pd.to_numeric(df["labels"], errors="coerce").to_numpy(dtype=int)

    metrics = compute_prediction_metrics(df, n_bins=args.n_bins)
    pd.DataFrame([metrics]).to_csv(output_dir / "prediction_metrics.csv", index=False)

    payload = torch.load(args.interpretability_pt, map_location="cpu")

    cls_hidden = tensor_to_numpy(payload.get("cls_hidden_states"))
    mask_hidden = tensor_to_numpy(payload.get("mask_hidden_states"))
    all_layer_cls = tensor_to_numpy(payload.get("all_layer_cls_hidden_states"))
    all_layer_mask = tensor_to_numpy(payload.get("all_layer_mask_hidden_states"))

    probe_rows = []

    for name, X in [
        ("cls_hidden_states", cls_hidden),
        ("auxiliary_mask_hidden_states", mask_hidden),
    ]:
        if X is None:
            continue

        result = run_probe(X, y, n_splits=args.n_splits, random_state=args.random_state)
        probe_rows.append(
            {
                "representation": name,
                "target": "diagnosis",
                "mean_score": result["mean_score"],
                "std_score": result["std_score"],
                "n_splits": result["n_splits"],
                "fold_scores": json.dumps(result["fold_scores"]),
            }
        )

    pd.DataFrame(probe_rows).to_csv(output_dir / "representation_probe_results.csv", index=False)

    layer_rows = []

    for rep_name, arr in [
        ("cls", all_layer_cls),
        ("auxiliary_mask", all_layer_mask),
    ]:
        if arr is None:
            continue

        if arr.ndim != 3:
            raise ValueError(f"Expected [n_samples, n_layers, hidden_dim], got {arr.shape}")

        for layer_idx in range(arr.shape[1]):
            result = run_probe(arr[:, layer_idx, :], y, n_splits=args.n_splits, random_state=args.random_state)
            layer_rows.append(
                {
                    "layer": layer_idx + 1,
                    "representation": rep_name,
                    "mean_score": result["mean_score"],
                    "std_score": result["std_score"],
                    "n_splits": result["n_splits"],
                    "fold_scores": json.dumps(result["fold_scores"]),
                }
            )

    pd.DataFrame(layer_rows).to_csv(output_dir / "layerwise_probe_results.csv", index=False)

    print("Saved outputs to:", output_dir)
    print("\nPrediction metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nRepresentation probes:")
    print(pd.DataFrame(probe_rows).to_string(index=False))


if __name__ == "__main__":
    main()
