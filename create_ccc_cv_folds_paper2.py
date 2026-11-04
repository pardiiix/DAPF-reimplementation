#!/usr/bin/env python3
"""Create reproducible, participant-disjoint CCC cross-validation folds for DAPF.

Run from the DAPF-main repository root:
    python3 create_ccc_cv_folds.py

Defaults:
    Input:  data/ccc_train_all.csv
    Output: latest_tmp_dir/five_fold_ccc105.json
    Folds:  5 stratified folds, seed 42

The output JSON is compatible with prompt_finetune.py's --val_file_dir
argument and has 'train_speaker' and 'test_speaker' keys containing the
actual values of the CSV's 'filename' column.

Options:
    python3 create_ccc_cv_folds.py --input data/ccc_train_all.csv \
        --output latest_tmp_dir/five_fold_ccc105.json --seed 42 --folds 5
    python3 create_ccc_cv_folds.py --help

If the CSV contains repeated participants, we use StratifiedGroupKFold so
all recordings from one participant remain in the same test fold.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def create_folds(csv_path: Path, n_splits: int, seed: int) -> list[dict[str, list[str]]]:
    df = pd.read_csv(csv_path, dtype={"filename": "string", "id": "string"})
    required = {"filename", "id", "ad"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CCC columns: {sorted(missing)}")

    if df.empty or df[list(required)].isna().any().any():
        raise ValueError("CCC must contain rows with nonmissing filename, id, and ad")
    if df["filename"].duplicated().any():
        raise ValueError("CCC has duplicate filenames; each transcript needs a unique filename")
    if not set(df["ad"]) <= {"dementia", "healthy"}:
        raise ValueError(f"Unexpected labels in ad column: {sorted(df['ad'].unique())}")
    if df["ad"].nunique() != 2:
        raise ValueError("Both 'dementia' and 'healthy' labels are required")

    # A participant should never switch diagnostic classes across visits in
    # this binary experiment without an explicit longitudinal-labeling policy.
    labels_per_person = df.groupby("id")["ad"].nunique()
    if (labels_per_person > 1).any():
        raise ValueError("Some participant IDs have conflicting labels; review these first")
    people_per_label = df.groupby("ad")["id"].nunique()
    if people_per_label.min() < n_splits:
        raise ValueError("Each class needs at least as many distinct participants as folds")

    duplicate_people = df["id"].duplicated().any()
    if duplicate_people:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_indices = splitter.split(df, df["ad"], groups=df["id"])
        method = "StratifiedGroupKFold (repeated participants)"
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_indices = splitter.split(df, df["ad"])
        method = "StratifiedKFold (one transcript per participant)"

    folds: list[dict[str, list[str]]] = []
    test_filenames: list[str] = []
    all_names = set(df["filename"].tolist())

    print(f"Input: {csv_path}")
    print(f"Rows: {len(df)}, participants: {df['id'].nunique()}, labels: {df['ad'].value_counts().to_dict()}")
    print(f"Method: {method}; n_splits={n_splits}; seed={seed}")

    for i, (train_idx, test_idx) in enumerate(split_indices):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        train_names = train_df["filename"].tolist()
        test_names = test_df["filename"].tolist()
        if set(train_names) & set(test_names):
            raise AssertionError(f"Fold {i}: train/test transcript overlap")
        if set(train_df["id"]) & set(test_df["id"]):
            raise AssertionError(f"Fold {i}: train/test participant overlap")
        if set(train_names) | set(test_names) != all_names:
            raise AssertionError(f"Fold {i}: some CCC examples omitted")
        if len(test_names) == 0 or len(train_names) == 0:
            raise AssertionError(f"Fold {i}: an empty partition was created")

        folds.append({"train_speaker": train_names, "test_speaker": test_names})
        test_filenames.extend(test_names)
        print(f"Fold {i}: train={len(train_names)}, test={len(test_names)}, "
              f"test_labels={dict(Counter(test_df['ad']))}")

    if len(test_filenames) != len(df) or len(set(test_filenames)) != len(df):
        raise AssertionError("Every CCC transcript must appear in exactly one test fold")
    if set(test_filenames) != all_names:
        raise AssertionError("Some CCC transcripts are missing from the test folds")
    return folds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("data/ccc_train_all.csv"))
    p.add_argument("--output", type=Path, default=Path("latest_tmp_dir/five_fold_ccc105.json"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="Allow replacing a different existing split file")
    args = p.parse_args()
    if args.folds < 2:
        p.error("--folds must be >= 2")

    folds = create_folds(args.input, args.folds, args.seed)
    output_text = json.dumps(folds, indent=2, ensure_ascii=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = args.output.read_text(encoding="utf-8")
        if existing == output_text:
            print(f"\nPASS: Split file already matches; unchanged: {args.output}")
            return
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output} already exists with different folds. "
                "Use another --output filename, or --overwrite intentionally."
            )
    args.output.write_text(output_text, encoding="utf-8")
    print(f"\nPASS: Every transcript held out once, no participant leakage.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
