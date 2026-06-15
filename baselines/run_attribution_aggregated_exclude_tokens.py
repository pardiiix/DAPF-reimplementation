"""
Aggregate token attributions across seeds while excluding prompt-scaffold tokens.

Use this for BERT-CLS+Prompt, where prompt words such as "patient",
"diagnosis", "participant", and "narration" can dominate attribution rankings.

Input:
    output_bert_cls_prompt_interpretability_multiseed/seed_X/
        integrated_gradients_token_attributions_clean.csv

Output:
    output_bert_cls_prompt_interpretability_multiseed_summary/
        attribution_aggregated_no_prompt_tokens/
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EXCLUDE_TOKENS = {
    "participant",
    "participants",
    "participant's",
    "narration",
    "on",
    "source",
    "target",
    "patient",
    "has",
    "diagnosis",
    "diagnoses",
    "mask",
}


def parse_seeds(seed_string: str):
    return [int(s.strip()) for s in seed_string.split(",") if s.strip()]


def normalize_token(token: str) -> str:
    token = str(token).strip().lower()

    # Remove BERT wordpiece marker.
    token = token.replace("##", "")

    # Remove surrounding punctuation.
    token = re.sub(r"^[^\w']+|[^\w']+$", "", token)

    return token


def load_seed_attributions(interp_root: Path, seed: int) -> pd.DataFrame:
    path = interp_root / f"seed_{seed}" / "integrated_gradients_token_attributions_clean.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing attribution file for seed {seed}: {path}")

    df = pd.read_csv(path)
    df["seed"] = seed
    df["token_norm"] = df["token"].apply(normalize_token)

    return df


def aggregate_token_level(df: pd.DataFrame, min_count: int, min_samples: int) -> pd.DataFrame:
    grouped = (
        df.groupby("token_norm")
        .agg(
            total_count=("attribution", "size"),
            n_seeds=("seed", "nunique"),
            n_samples=("id", "nunique"),
            mean_samples_per_seed=("id", lambda x: x.nunique() / df["seed"].nunique()),
            mean_attribution=("attribution", "mean"),
            median_attribution=("attribution", "median"),
            std_attribution=("attribution", "std"),
            mean_abs_attribution=("abs_attribution", "mean"),
            max_abs_attribution=("abs_attribution", "max"),
        )
        .reset_index()
    )

    grouped["std_attribution"] = grouped["std_attribution"].fillna(0.0)

    filtered = grouped[
        (grouped["total_count"] >= min_count)
        & (grouped["n_samples"] >= min_samples)
    ].copy()

    filtered = filtered.sort_values("mean_abs_attribution", ascending=False)

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate attribution while excluding specified prompt-scaffold tokens."
    )

    parser.add_argument(
        "--interp_root",
        type=str,
        default="./output_bert_cls_prompt_interpretability_multiseed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_bert_cls_prompt_interpretability_multiseed_summary/attribution_aggregated_no_prompt_tokens",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--min_count", type=int, default=10)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--loose_min_count", type=int, default=5)
    parser.add_argument("--loose_min_samples", type=int, default=3)
    parser.add_argument(
        "--exclude_tokens",
        type=str,
        default=",".join(sorted(DEFAULT_EXCLUDE_TOKENS)),
        help="Comma-separated normalized tokens to exclude.",
    )

    args = parser.parse_args()

    interp_root = Path(args.interp_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    exclude_tokens = {
        normalize_token(tok)
        for tok in args.exclude_tokens.split(",")
        if tok.strip()
    }

    all_dfs = []
    for seed in seeds:
        seed_df = load_seed_attributions(interp_root, seed)
        all_dfs.append(seed_df)

    df = pd.concat(all_dfs, ignore_index=True)

    before_rows = len(df)
    before_tokens = df["token_norm"].nunique()

    df_filtered = df[~df["token_norm"].isin(exclude_tokens)].copy()

    after_rows = len(df_filtered)
    after_tokens = df_filtered["token_norm"].nunique()

    df.to_csv(output_dir / "all_seed_token_occurrences_before_prompt_filter.csv", index=False)
    df_filtered.to_csv(output_dir / "all_seed_token_occurrences_after_prompt_filter.csv", index=False)

    pd.DataFrame(
        {
            "excluded_token": sorted(exclude_tokens),
        }
    ).to_csv(output_dir / "excluded_prompt_tokens.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "n_seeds": len(seeds),
                "rows_before_filter": before_rows,
                "rows_after_filter": after_rows,
                "unique_tokens_before_filter": before_tokens,
                "unique_tokens_after_filter": after_tokens,
                "removed_rows": before_rows - after_rows,
                "removed_unique_tokens": before_tokens - after_tokens,
            }
        ]
    )
    summary.to_csv(output_dir / "filter_summary.csv", index=False)

    main_filtered = aggregate_token_level(
        df_filtered,
        min_count=args.min_count,
        min_samples=args.min_samples,
    )

    loose_filtered = aggregate_token_level(
        df_filtered,
        min_count=args.loose_min_count,
        min_samples=args.loose_min_samples,
    )

    main_filtered.to_csv(
        output_dir / f"cross_seed_main_filtered_count{args.min_count}_samples{args.min_samples}.csv",
        index=False,
    )

    loose_filtered.to_csv(
        output_dir / f"cross_seed_loose_filtered_count{args.loose_min_count}_samples{args.loose_min_samples}.csv",
        index=False,
    )

    for prefix, table in [
        ("cross_seed_main", main_filtered),
        ("cross_seed_loose", loose_filtered),
    ]:
        table.sort_values("mean_attribution", ascending=False).head(100).to_csv(
            output_dir / f"{prefix}_top_positive_mean_attribution.csv",
            index=False,
        )

        table.sort_values("mean_attribution", ascending=True).head(100).to_csv(
            output_dir / f"{prefix}_top_negative_mean_attribution.csv",
            index=False,
        )

        table.sort_values("mean_abs_attribution", ascending=False).head(100).to_csv(
            output_dir / f"{prefix}_top_mean_abs_attribution.csv",
            index=False,
        )

    print("Saved prompt-filtered attribution aggregation to:", output_dir)
    print(summary.to_string(index=False))

    print("\nTop positive after prompt-token exclusion:")
    print(
        main_filtered.sort_values("mean_attribution", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    print("\nTop negative after prompt-token exclusion:")
    print(
        main_filtered.sort_values("mean_attribution", ascending=True)
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
