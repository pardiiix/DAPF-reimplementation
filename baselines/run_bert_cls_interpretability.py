"""
Post-hoc interpretability for the standard BERT-CLS baseline.

Inputs:
    output_bert_cls_multiseed/seed_X/test_results.csv
    output_bert_cls_multiseed/seed_X/test_results_interpretability.pt

Outputs:
    prediction_metrics.csv
    representation_probe_results.csv
    layerwise_probe_results.csv

This script mirrors the DAPF post-hoc interpretability script, but for a
standard BERT sequence classifier. There is no [MASK] diagnosis position.
"""

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional

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
    """Parse list-like strings safely."""
    if isinstance(value, list):
        return value

    if isinstance(value, np.ndarray):
        return value.tolist()

    if not isinstance(value, str):
        return value

    value = value.strip()

    try:
        return json.loads(value)
    except Exception:
        pass

    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def compute_ece(
    y_true: np.ndarray,
    y_prob_dementia: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error for binary classification."""
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

        bin_acc = correctness[mask].mean()
        bin_conf = confidence[mask].mean()
        bin_weight = mask.mean()
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def get_probability_dementia(df: pd.DataFrame) -> np.ndarray:
    """Recover dementia probability from probability_dementia or probas column."""
    if "probability_dementia" in df.columns:
        return pd.to_numeric(df["probability_dementia"], errors="coerce").to_numpy()

    if "probas" not in df.columns:
        raise ValueError("Need either probability_dementia or probas column.")

    probs = df["probas"].apply(parse_list).tolist()
    probs = np.asarray(probs, dtype=float)

    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(f"Unexpected probas shape: {probs.shape}")

    return probs[:, 1]


def compute_prediction_metrics(df: pd.DataFrame, n_bins: int = 10) -> Dict[str, float]:
    """Compute classification and calibration metrics."""
    y_true = pd.to_numeric(df["labels"], errors="coerce").to_numpy(dtype=int)
    y_pred = pd.to_numeric(df["pred_labels"], errors="coerce").to_numpy(dtype=int)
    y_prob = get_probability_dementia(df)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "ece": compute_ece(y_true, y_prob, n_bins=n_bins),
    }

    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auroc"] = float("nan")

    return metrics


def run_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict:
    """Train a standardized logistic-regression probe with stratified CV."""
    y = np.asarray(y, dtype=int)
    X = np.asarray(X, dtype=float)

    _, class_counts = np.unique(y, return_counts=True)
    max_splits = int(class_counts.min())

    if max_splits < 2:
        return {
            "mean_score": float("nan"),
            "std_score": float("nan"),
            "fold_scores": [],
            "n_splits": 0,
        }

    actual_splits = min(n_splits, max_splits)

    cv = StratifiedKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=random_state,
    )

    scores = []

    for train_idx, test_idx in cv.split(X, y):
        probe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        )

        probe.fit(X[train_idx], y[train_idx])
        pred = probe.predict(X[test_idx])
        score = f1_score(y[test_idx], pred, average="macro", zero_division=0)
        scores.append(float(score))

    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "fold_scores": scores,
        "n_splits": actual_splits,
    }


def tensor_to_numpy(x):
    """Convert tensor/list to numpy array."""
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run post-hoc interpretability for BERT-CLS baseline."
    )

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
    all_layer_cls = tensor_to_numpy(payload.get("all_layer_cls_hidden_states"))

    probe_rows = []

    if cls_hidden is not None:
        result = run_probe(
            cls_hidden,
            y,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        probe_rows.append(
            {
                "representation": "cls_hidden_states",
                "target": "diagnosis",
                "mean_score": result["mean_score"],
                "std_score": result["std_score"],
                "n_splits": result["n_splits"],
                "fold_scores": json.dumps(result["fold_scores"]),
            }
        )

    pd.DataFrame(probe_rows).to_csv(
        output_dir / "representation_probe_results.csv",
        index=False,
    )

    layer_rows = []

    if all_layer_cls is not None:
        # Expected shape: [n_samples, n_layers, hidden_dim].
        if all_layer_cls.ndim != 3:
            raise ValueError(
                "Expected all_layer_cls_hidden_states to have shape "
                f"[n_samples, n_layers, hidden_dim], got {all_layer_cls.shape}"
            )

        n_layers = all_layer_cls.shape[1]

        for layer_idx in range(n_layers):
            X_layer = all_layer_cls[:, layer_idx, :]
            result = run_probe(
                X_layer,
                y,
                n_splits=args.n_splits,
                random_state=args.random_state,
            )

            layer_rows.append(
                {
                    "layer": layer_idx + 1,
                    "representation": "cls",
                    "mean_score": result["mean_score"],
                    "std_score": result["std_score"],
                    "n_splits": result["n_splits"],
                    "fold_scores": json.dumps(result["fold_scores"]),
                }
            )

    pd.DataFrame(layer_rows).to_csv(
        output_dir / "layerwise_probe_results.csv",
        index=False,
    )

    print("Saved outputs to:", output_dir)
    print("\nPrediction metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    if probe_rows:
        print("\nRepresentation probing:")
        print(pd.DataFrame(probe_rows).to_string(index=False))

    if layer_rows:
        print("\nLayer-wise probing:")
        print(pd.DataFrame(layer_rows)[["layer", "representation", "mean_score", "std_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
