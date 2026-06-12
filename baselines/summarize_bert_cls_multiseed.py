"""
Summarize multi-seed BERT-CLS baseline outputs.

Reads:
    output_bert_cls_multiseed/seed_X/prediction_metrics.csv
    output_bert_cls_interpretability_multiseed/seed_X/prediction_metrics.csv
    output_bert_cls_interpretability_multiseed/seed_X/representation_probe_results.csv
    output_bert_cls_interpretability_multiseed/seed_X/layerwise_probe_results.csv
    output_bert_cls_interpretability_multiseed/seed_X/faithfulness_positive/
    output_bert_cls_interpretability_multiseed/seed_X/faithfulness_absolute/

Writes:
    scalar_metrics_by_seed.csv
    scalar_metrics_summary.csv
    representation_probe_by_seed.csv
    representation_probe_summary.csv
    layerwise_probe_by_seed.csv
    layerwise_probe_summary.csv
    faithfulness_by_seed.csv
    faithfulness_summary.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def parse_seeds(seed_string: str) -> List[int]:
    return [int(s.strip()) for s in seed_string.split(",") if s.strip()]


def read_first_existing(paths: List[Path]) -> Optional[pd.DataFrame]:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return None


def mean_pm_std(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "nan"
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def summarize_numeric(
    df: pd.DataFrame,
    group_cols: List[str],
    value_cols: List[str],
) -> pd.DataFrame:
    rows = []

    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: key for col, key in zip(group_cols, keys)}
        row["n_seeds"] = group["seed"].nunique() if "seed" in group.columns else len(group)

        for col in value_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=0))
            row[f"{col}_mean_pm_std"] = mean_pm_std(
                row[f"{col}_mean"],
                row[f"{col}_std"],
            )

        rows.append(row)

    return pd.DataFrame(rows)


def collect_scalar_metrics(
    train_root: Path,
    interp_root: Path,
    seeds: List[int],
) -> pd.DataFrame:
    rows = []

    for seed in seeds:
        candidates = [
            interp_root / f"seed_{seed}" / "prediction_metrics.csv",
            train_root / f"seed_{seed}" / "prediction_metrics.csv",
        ]

        df = read_first_existing(candidates)

        if df is None or df.empty:
            print(f"Warning: missing prediction metrics for seed {seed}")
            continue

        row = df.iloc[0].to_dict()
        row["seed"] = seed
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    cols = ["seed"] + [c for c in rows[0].keys() if c != "seed"]
    return pd.DataFrame(rows)[cols]


def collect_representation_probe(
    interp_root: Path,
    seeds: List[int],
) -> pd.DataFrame:
    rows = []

    for seed in seeds:
        path = interp_root / f"seed_{seed}" / "representation_probe_results.csv"

        if not path.exists():
            print(f"Warning: missing representation probe for seed {seed}")
            continue

        df = pd.read_csv(path)
        if df.empty:
            continue

        df["seed"] = seed
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    cols = ["seed"] + [c for c in out.columns if c != "seed"]
    return out[cols]


def collect_layerwise_probe(
    interp_root: Path,
    seeds: List[int],
) -> pd.DataFrame:
    rows = []

    for seed in seeds:
        path = interp_root / f"seed_{seed}" / "layerwise_probe_results.csv"

        if not path.exists():
            print(f"Warning: missing layerwise probe for seed {seed}")
            continue

        df = pd.read_csv(path)
        if df.empty:
            continue

        df["seed"] = seed
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    cols = ["seed"] + [c for c in out.columns if c != "seed"]
    return out[cols]


def collect_faithfulness(
    interp_root: Path,
    seeds: List[int],
) -> pd.DataFrame:
    """
    Collect per-seed faithfulness curves and summarize each seed by fraction.

    Expected:
        faithfulness_positive/deletion_curve.csv
        faithfulness_positive/insertion_curve.csv
        faithfulness_absolute/deletion_curve.csv
        faithfulness_absolute/insertion_curve.csv
    """
    rows = []

    for seed in seeds:
        seed_dir = interp_root / f"seed_{seed}"

        for ranking in ["positive", "absolute"]:
            base = seed_dir / f"faithfulness_{ranking}"

            deletion_path = base / "deletion_curve.csv"
            insertion_path = base / "insertion_curve.csv"

            if deletion_path.exists():
                deletion = pd.read_csv(deletion_path)

                if "probability_drop" in deletion.columns:
                    grouped = (
                        deletion.groupby("fraction")["probability_drop"]
                        .agg(["mean", "std", "count"])
                        .reset_index()
                    )

                    for _, r in grouped.iterrows():
                        rows.append(
                            {
                                "seed": seed,
                                "ranking": ranking,
                                "test": "deletion",
                                "fraction": float(r["fraction"]),
                                "metric": "probability_drop",
                                "value": float(r["mean"]),
                                "within_seed_std": float(r["std"])
                                if not pd.isna(r["std"])
                                else 0.0,
                                "n_examples": int(r["count"]),
                            }
                        )
                else:
                    print(f"Warning: no probability_drop in {deletion_path}")

            else:
                print(f"Warning: missing {deletion_path}")

            if insertion_path.exists():
                insertion = pd.read_csv(insertion_path)

                if "probability_recovered" in insertion.columns:
                    grouped = (
                        insertion.groupby("fraction")["probability_recovered"]
                        .agg(["mean", "std", "count"])
                        .reset_index()
                    )

                    for _, r in grouped.iterrows():
                        rows.append(
                            {
                                "seed": seed,
                                "ranking": ranking,
                                "test": "insertion",
                                "fraction": float(r["fraction"]),
                                "metric": "probability_recovered",
                                "value": float(r["mean"]),
                                "within_seed_std": float(r["std"])
                                if not pd.isna(r["std"])
                                else 0.0,
                                "n_examples": int(r["count"]),
                            }
                        )
                else:
                    print(f"Warning: no probability_recovered in {insertion_path}")

            else:
                print(f"Warning: missing {insertion_path}")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize multi-seed BERT-CLS baseline results."
    )

    parser.add_argument("--train_root", type=str, default="./output_bert_cls_multiseed")
    parser.add_argument(
        "--interp_root",
        type=str,
        default="./output_bert_cls_interpretability_multiseed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_bert_cls_interpretability_multiseed_summary",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")

    args = parser.parse_args()

    train_root = Path(args.train_root)
    interp_root = Path(args.interp_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)

    print("Summarizing BERT-CLS baseline")
    print("Train root:", train_root)
    print("Interpretability root:", interp_root)
    print("Output dir:", output_dir)
    print("Seeds:", seeds)

    scalar_by_seed = collect_scalar_metrics(train_root, interp_root, seeds)

    if not scalar_by_seed.empty:
        scalar_by_seed.to_csv(output_dir / "scalar_metrics_by_seed.csv", index=False)

        numeric_cols = [
            c
            for c in scalar_by_seed.columns
            if c != "seed" and pd.api.types.is_numeric_dtype(scalar_by_seed[c])
        ]

        summary_rows = []

        for col in numeric_cols:
            values = pd.to_numeric(scalar_by_seed[col], errors="coerce")
            summary_rows.append(
                {
                    "metric": col,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "n_seeds": int(values.notna().sum()),
                    "mean_pm_std": mean_pm_std(
                        float(values.mean()),
                        float(values.std(ddof=0)),
                    ),
                }
            )

        scalar_summary = pd.DataFrame(summary_rows)
        scalar_summary.to_csv(output_dir / "scalar_metrics_summary.csv", index=False)

        print("\nScalar metrics summary:")
        print(scalar_summary.to_string(index=False))
    else:
        print("No scalar metrics found.")

    probe_by_seed = collect_representation_probe(interp_root, seeds)

    if not probe_by_seed.empty:
        probe_by_seed.to_csv(
            output_dir / "representation_probe_by_seed.csv",
            index=False,
        )

        probe_summary = summarize_numeric(
            probe_by_seed,
            group_cols=["representation", "target"],
            value_cols=["mean_score"],
        )

        probe_summary.to_csv(
            output_dir / "representation_probe_summary.csv",
            index=False,
        )

        print("\nRepresentation probe summary:")
        print(probe_summary.to_string(index=False))
    else:
        print("No representation probe results found.")

    layer_by_seed = collect_layerwise_probe(interp_root, seeds)

    if not layer_by_seed.empty:
        layer_by_seed.to_csv(output_dir / "layerwise_probe_by_seed.csv", index=False)

        layer_summary = summarize_numeric(
            layer_by_seed,
            group_cols=["layer", "representation"],
            value_cols=["mean_score"],
        )

        layer_summary = layer_summary.sort_values(["representation", "layer"])
        layer_summary.to_csv(output_dir / "layerwise_probe_summary.csv", index=False)

        print("\nLayer-wise probe summary:")
        print(layer_summary.to_string(index=False))
    else:
        print("No layer-wise probe results found.")

    faith_by_seed = collect_faithfulness(interp_root, seeds)

    if not faith_by_seed.empty:
        faith_by_seed.to_csv(output_dir / "faithfulness_by_seed.csv", index=False)

        faith_summary = summarize_numeric(
            faith_by_seed,
            group_cols=["ranking", "test", "fraction", "metric"],
            value_cols=["value"],
        )

        faith_summary = faith_summary.sort_values(["ranking", "test", "fraction"])
        faith_summary.to_csv(output_dir / "faithfulness_summary.csv", index=False)

        print("\nFaithfulness summary:")
        print(faith_summary.to_string(index=False))
    else:
        print("No faithfulness results found yet.")

    print("\nSaved summary outputs to:", output_dir)


if __name__ == "__main__":
    main()
