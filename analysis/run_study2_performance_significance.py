#!/usr/bin/env python3
"""Study 2: participant-paired performance inference across five seeds.

Run from DAPF-main (branch paired-error-analysis):
    python analysis/significance/run_study2_performance_significance.py \
        --output_dir significance_results/study2/performance

Read-only inputs: participant predictions in five CSV model families and two
DAPF hidden-state PT model families. No models are trained or modified.

The participant is the resampling/permutation unit. A given participant's five
seed predictions are always swapped as a group in the paired permutation test.
Bootstrap confidence intervals resample the *same* stratified participants for
both models and the *same* seed indices for both models. Thus repeated patient
predictions are NOT counted as independent data points.

Caveats: permutation p-values require exchangeability of the two prediction
vectors under the null; they are not tests of a causal backbone effect. Five
seeds limit uncertainty quantification for training-run variability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SEEDS_DEFAULT = (0, 1, 2, 3, 4)

PATHS = {
    "DAPF-BERT": "output_interpretability_multiseed/seed_{seed}/layerwise_probing/layerwise_hidden_states.pt",
    "DAPF-RoBERTa": "output_roberta_interpretability/seed_{seed}/layerwise_probing/layerwise_hidden_states.pt",
    "BERT-CLS": "output_bert_cls_multiseed/seed_{seed}/test_results.csv",
    "BERT-CLS+Prompt": "output_bert_cls_prompt_multiseed/seed_{seed}/test_results.csv",
    "RoBERTa-CLS": "output_roberta_cls_multiseed/seed_{seed}/test_results.csv",
    "RoBERTa-CLS+Prompt": "output_roberta_cls_prompt_multiseed/seed_{seed}/test_results.csv",
}

# Manuscript Table 1. Used only as a consistency check, never as test inputs.
EXPECTED_MACRO_F1 = {
    "DAPF-BERT": 0.8287,
    "BERT-CLS": 0.8063,
    "BERT-CLS+Prompt": 0.8065,
    "DAPF-RoBERTa": 0.7555,
    "RoBERTa-CLS": 0.7736,
    "RoBERTa-CLS+Prompt": 0.7883,
}

# Pre-specified family for Holm correction; Study 1 BERT-only contrasts excluded.
CONTRASTS = [
    ("DAPF-BERT", "DAPF-RoBERTa"),
    ("DAPF-RoBERTa", "RoBERTa-CLS"),
    ("DAPF-RoBERTa", "RoBERTa-CLS+Prompt"),
    ("BERT-CLS", "RoBERTa-CLS"),
    ("BERT-CLS+Prompt", "RoBERTa-CLS+Prompt"),
]


def array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_records(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required prediction file not found: {path}")
    if path.suffix == ".pt":
        # Load ONLY your trusted, locally generated checkpoint outputs.
        obj = torch.load(path, map_location="cpu", weights_only=False)
        required = ("ids", "labels", "pred_labels")
        missing = [k for k in required if k not in obj]
        if missing:
            raise ValueError(f"{path} missing keys: {missing}")
        out = pd.DataFrame({
            "id": [str(x) for x in obj["ids"]],
            "label": array(obj["labels"]).reshape(-1),
            "prediction": array(obj["pred_labels"]).reshape(-1),
        })
    else:
        df = pd.read_csv(path, dtype={"id": str})
        required = {"id", "labels", "pred_labels"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} missing columns: {sorted(required-set(df.columns))}")
        out = df[["id", "labels", "pred_labels"]].rename(
            columns={"labels": "label", "pred_labels": "prediction"}
        ).copy()
    out["id"] = out["id"].astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    out["prediction"] = pd.to_numeric(out["prediction"], errors="raise").astype(int)
    if out["id"].duplicated().any():
        raise ValueError(f"Duplicate participant IDs: {path}")
    if len(out) != 48 or set(out["label"]) != {0, 1} or not set(out["prediction"]).issubset({0, 1}):
        raise ValueError(f"Expected 48 binary-labeled participants and binary predictions: {path}")
    return out.set_index("id").sort_index()


def load_all(root: Path, seeds: tuple[int, ...]):
    pred = {}
    manifests = []
    common_ids = None
    common_labels = None
    for model, pattern in PATHS.items():
        rows = []
        for seed in seeds:
            path = root / pattern.format(seed=seed)
            data = load_records(path)
            ids = data.index.tolist()
            labels = data["label"].to_numpy()
            if common_ids is None:
                common_ids, common_labels = ids, labels
            if ids != common_ids:
                missing = set(common_ids).difference(ids)
                extra = set(ids).difference(common_ids)
                raise ValueError(f"ID alignment failure for {model} seed={seed}: missing={list(missing)[:5]}, extra={list(extra)[:5]}")
            if not np.array_equal(labels, common_labels):
                raise ValueError(f"Label alignment failure for {model} seed={seed}: {path}")
            rows.append(data["prediction"].to_numpy(dtype=np.int8))
            manifests.append({"model": model, "seed": seed, "input_path": str(path)})
        pred[model] = np.stack(rows, axis=0)
    return pred, common_labels.astype(np.int8), manifests


def macro_f1_per_seed(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Vector of per-seed binary macro-F1; p shape (seeds, observations)."""
    if p.ndim != 2 or y.ndim != 1 or p.shape[1] != y.shape[0]:
        raise ValueError("Unexpected shape for macro-F1 calculation")
    pos_tp = ((p == 1) & (y[None, :] == 1)).sum(axis=1)
    pos_fp = ((p == 1) & (y[None, :] == 0)).sum(axis=1)
    pos_fn = ((p == 0) & (y[None, :] == 1)).sum(axis=1)
    neg_tp = ((p == 0) & (y[None, :] == 0)).sum(axis=1)
    neg_fp = pos_fn
    neg_fn = pos_fp
    f1_pos = np.divide(2 * pos_tp, 2 * pos_tp + pos_fp + pos_fn,
                       out=np.zeros(p.shape[0], dtype=float), where=(2 * pos_tp + pos_fp + pos_fn) != 0)
    f1_neg = np.divide(2 * neg_tp, 2 * neg_tp + neg_fp + neg_fn,
                       out=np.zeros(p.shape[0], dtype=float), where=(2 * neg_tp + neg_fp + neg_fn) != 0)
    return (f1_pos + f1_neg) / 2


def mean_difference(y, a, b) -> float:
    return float(np.mean(macro_f1_per_seed(y, a) - macro_f1_per_seed(y, b)))


def participant_permutation(y, a, b, n_perm, rng):
    observed = mean_difference(y, a, b)
    more_extreme = 0
    for _ in range(n_perm):
        swap = rng.random(len(y)) < 0.5
        # Same swap for every seed of a given participant.
        aa = np.where(swap[None, :], b, a)
        bb = np.where(swap[None, :], a, b)
        val = mean_difference(y, aa, bb)
        more_extreme += abs(val) >= abs(observed) - 1e-12
    return float((more_extreme + 1) / (n_perm + 1))


def paired_bootstrap_ci(y, a, b, n_boot, rng, resample_seeds=False):
    class0 = np.flatnonzero(y == 0)
    class1 = np.flatnonzero(y == 1)
    values = np.empty(n_boot, dtype=float)
    for iteration in range(n_boot):
        # Identical participants, by class, for both models and across seeds.
        obs = np.concatenate((rng.choice(class0, len(class0), replace=True),
                              rng.choice(class1, len(class1), replace=True)))
        # Optionally incorporate variation in the five (paired) runs.
        s = rng.choice(a.shape[0], a.shape[0], replace=True) if resample_seeds else np.arange(a.shape[0])
        values[iteration] = mean_difference(y[obs], a[s][:, obs], b[s][:, obs])
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))


def holm_adjust(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    corrected = np.empty(len(p), dtype=float)
    last = 0.0
    for rank, original_idx in enumerate(order):
        last = max(last, min(1.0, (len(p) - rank) * p[original_idx]))
        corrected[original_idx] = last
    return corrected.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--output_dir", type=Path, default=Path("significance_results/study2/performance"))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--n_permutations", type=int, default=10000)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--random_state", type=int, default=2026)
    parser.add_argument("--skip_manuscript_metric_check", action="store_true", help="Diagnostic only: do not publish results without reconciling Table 1")
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    if not seeds or args.n_permutations <= 0 or args.n_bootstrap <= 0:
        parser.error("Seeds, permutations, and bootstrap counts must be nonempty/positive")
    preds, labels, manifest = load_all(args.repo_root, seeds)
    rng = np.random.default_rng(args.random_state)
    metrics = []
    for name, values in preds.items():
        seed_f1 = macro_f1_per_seed(labels, values)
        mean_f1 = float(seed_f1.mean())
        expected = EXPECTED_MACRO_F1.get(name)
        metrics.append({"model": name, "mean_macro_f1": mean_f1, "sd_macro_f1": float(seed_f1.std(ddof=1)) if len(seeds) > 1 else np.nan, "seed_macro_f1": json.dumps(seed_f1.tolist()), "manuscript_mean_macro_f1": expected})
    df_metrics = pd.DataFrame(metrics)
    print("\nVERIFICATION AGAINST STUDY 2 TABLE 1:\n" + df_metrics[["model", "mean_macro_f1", "sd_macro_f1", "manuscript_mean_macro_f1"]].to_string(index=False))
    mismatched = df_metrics[np.abs(df_metrics.mean_macro_f1 - df_metrics.manuscript_mean_macro_f1) > 0.002]
    if len(mismatched) and not args.skip_manuscript_metric_check:
        raise RuntimeError("\nSaved predictions do not reproduce Study 2 Table 1! Check seed files and checkpoint choices before inference.\n" + mismatched[["model", "mean_macro_f1", "manuscript_mean_macro_f1"]].to_string(index=False))
    rows, raw_p = [], []
    for name_a, name_b in CONTRASTS:
        print(f"Testing {name_a} - {name_b}...", flush=True)
        a, b = preds[name_a], preds[name_b]
        observed = mean_difference(labels, a, b)
        seed_diffs = macro_f1_per_seed(labels, a) - macro_f1_per_seed(labels, b)
        p = participant_permutation(labels, a, b, args.n_permutations, rng)
        ci_part = paired_bootstrap_ci(labels, a, b, args.n_bootstrap, rng, False)
        ci_both = paired_bootstrap_ci(labels, a, b, args.n_bootstrap, rng, True)
        rows.append({"model_a": name_a, "model_b": name_b, "delta_macro_f1_mean_across_seeds": observed,
                     "seed_level_differences": json.dumps(seed_diffs.tolist()),
                     "paired_participant_permutation_p_two_sided": p,
                     "participant_bootstrap_95_ci_low": ci_part[0], "participant_bootstrap_95_ci_high": ci_part[1],
                     "participant_and_seed_bootstrap_95_ci_low": ci_both[0], "participant_and_seed_bootstrap_95_ci_high": ci_both[1],
                     "n_participants": len(labels), "n_seeds": len(seeds),
                     "n_permutations": args.n_permutations, "n_bootstrap": args.n_bootstrap})
        raw_p.append(p)
    p_adjusted = holm_adjust(raw_p)
    for row, adjusted in zip(rows, p_adjusted):
        row["holm_adjusted_p_for_five_contrasts"] = adjusted
    df_results = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(args.output_dir / "six_model_seed_metrics.csv", index=False)
    df_results.to_csv(args.output_dir / "performance_significance.csv", index=False)
    pd.DataFrame(manifest).to_csv(args.output_dir / "input_file_manifest.csv", index=False)
    print("\nSIGNIFICANCE SUMMARY:\n" + df_results[["model_a", "model_b", "delta_macro_f1_mean_across_seeds", "paired_participant_permutation_p_two_sided", "holm_adjusted_p_for_five_contrasts", "participant_and_seed_bootstrap_95_ci_low", "participant_and_seed_bootstrap_95_ci_high"]].to_string(index=False))
    print(f"\nSaved to {args.output_dir}")
    print("CAUTION: Exploratory, participant-paired analysis. No causal effect of backbone can be inferred. Five seeds and one test cohort limit generalization.")

if __name__ == "__main__":
    main()
