"""
Partition SHAP utilities for token-level text attribution.

The code is model-agnostic. The caller provides a prediction function that:

    1. accepts a list of transcript strings;
    2. returns an array of shape (n_examples, 2);
    3. contains class probabilities in this order:
           column 0: Control
           column 1: Dementia

The caller remains responsible for reconstructing the model's complete prompt
around each transcript before inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import shap


PredictionFunction = Callable[[list[str]], np.ndarray]


def _validate_probabilities(
    probabilities: np.ndarray,
    expected_rows: int,
) -> np.ndarray:
    """Validate the output returned by the model wrapper."""

    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.ndim != 2:
        raise ValueError(
            "The prediction function must return a two-dimensional array. "
            f"Received shape {probabilities.shape}."
        )

    if probabilities.shape != (expected_rows, 2):
        raise ValueError(
            "The prediction function must return shape "
            f"({expected_rows}, 2), but returned {probabilities.shape}."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "The prediction function returned NaN or infinite values."
        )

    if (probabilities < -1e-7).any() or (probabilities > 1.0 + 1e-7).any():
        raise ValueError(
            "The prediction function must return probabilities between "
            "zero and one."
        )

    row_sums = probabilities.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(
            "Each prediction row must sum to one. "
            f"Observed row sums: {row_sums[:10]}"
        )

    return probabilities


def build_partition_explainer(
    tokenizer,
    predict_proba: PredictionFunction,
    algorithm: str = "partition",
):
    """
    Construct a SHAP text explainer.

    Parameters
    ----------
    tokenizer
        Hugging Face tokenizer used by the model.
    predict_proba
        Function mapping transcript strings to class probabilities.
    algorithm
        Use "partition" for the reported Partition SHAP experiment.
        "auto" is available only for exploratory checks.
    """

    if algorithm not in {"partition", "auto"}:
        raise ValueError(
            "algorithm must be either 'partition' or 'auto'."
        )

    def checked_predict(texts):
        # SHAP may provide a NumPy array rather than a Python list.
        normalized_texts = [str(text) for text in list(texts)]

        probabilities = predict_proba(normalized_texts)

        return _validate_probabilities(
            probabilities=probabilities,
            expected_rows=len(normalized_texts),
        )

    masker = shap.maskers.Text(
        tokenizer=tokenizer,
        mask_token=tokenizer.mask_token,
        collapse_mask_token=True,
        output_type="string",
    )

    explainer = shap.Explainer(
        model=checked_predict,
        masker=masker,
        algorithm=algorithm,
        output_names=["Control", "Dementia"],
    )

    return explainer


def run_partition_shap(
    *,
    tokenizer,
    predict_proba: PredictionFunction,
    sample_ids: Sequence[str],
    transcripts: Sequence[str],
    labels: Sequence[int],
    output_dir: str | Path,
    max_evals: int = 500,
    batch_size: int = 8,
    algorithm: str = "partition",
) -> pd.DataFrame:
    """
    Run Partition SHAP and save token-level attributions.

    The primary saved attribution is the contribution to the Dementia
    probability, which is output index 1.

    Returns
    -------
    pandas.DataFrame
        One row per token per transcript.
    """

    sample_ids = [str(value) for value in sample_ids]
    transcripts = [str(value) for value in transcripts]
    labels = [int(value) for value in labels]

    n_samples = len(transcripts)

    if len(sample_ids) != n_samples or len(labels) != n_samples:
        raise ValueError(
            "sample_ids, transcripts, and labels must have equal lengths. "
            f"Received {len(sample_ids)}, {len(transcripts)}, "
            f"and {len(labels)}."
        )

    if n_samples == 0:
        raise ValueError("No transcripts were supplied.")

    if max_evals < 2:
        raise ValueError("max_evals must be at least 2.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = build_partition_explainer(
        tokenizer=tokenizer,
        predict_proba=predict_proba,
        algorithm=algorithm,
    )

    print(f"SHAP explainer class: {type(explainer).__name__}")
    print(f"Number of transcripts: {n_samples}")
    print(f"Maximum evaluations per transcript: {max_evals}")

    all_rows: list[dict] = []
    transcript_rows: list[dict] = []

    for index, (sample_id, transcript, label) in enumerate(
        zip(sample_ids, transcripts, labels)
    ):
        print(
            f"[{index + 1}/{n_samples}] "
            f"Explaining sample {sample_id}"
        )

        explanation = explainer(
            [transcript],
            max_evals=max_evals,
            batch_size=batch_size,
            fixed_context=1,
        )

        # For two-output classification, expected shape is:
        # values: (1, number_of_tokens, 2)
        values = np.asarray(explanation.values)
        token_data = np.asarray(explanation.data, dtype=object)

        if values.ndim != 3 or values.shape[0] != 1:
            raise ValueError(
                "Unexpected SHAP values shape. "
                f"Expected (1, tokens, outputs), received {values.shape}."
            )

        if values.shape[2] != 2:
            raise ValueError(
                "Expected two model outputs, but SHAP returned "
                f"{values.shape[2]}."
            )

        tokens = [str(token) for token in token_data[0].tolist()]
        dementia_values = values[0, :, 1]
        control_values = values[0, :, 0]

        if len(tokens) != len(dementia_values):
            raise ValueError(
                f"Token/value mismatch for sample {sample_id}: "
                f"{len(tokens)} tokens versus "
                f"{len(dementia_values)} values."
            )

        # Obtain the model's unmasked prediction for reference.
        prediction = _validate_probabilities(
            predict_proba([transcript]),
            expected_rows=1,
        )[0]

        predicted_label = int(np.argmax(prediction))

        for token_index, (
            token,
            control_value,
            dementia_value,
        ) in enumerate(
            zip(tokens, control_values, dementia_values)
        ):
            all_rows.append(
                {
                    "id": sample_id,
                    "label": label,
                    "predicted_label": predicted_label,
                    "control_probability": float(prediction[0]),
                    "dementia_probability": float(prediction[1]),
                    "token_index": token_index,
                    "token": token,
                    "control_shap": float(control_value),
                    "dementia_shap": float(dementia_value),
                    "absolute_dementia_shap": float(
                        abs(dementia_value)
                    ),
                }
            )

        transcript_rows.append(
            {
                "id": sample_id,
                "label": label,
                "predicted_label": predicted_label,
                "control_probability": float(prediction[0]),
                "dementia_probability": float(prediction[1]),
                "number_of_shap_tokens": len(tokens),
                "sum_dementia_shap": float(
                    np.sum(dementia_values)
                ),
                "sum_absolute_dementia_shap": float(
                    np.sum(np.abs(dementia_values))
                ),
            }
        )

    token_df = pd.DataFrame(all_rows)
    transcript_df = pd.DataFrame(transcript_rows)

    token_path = output_dir / "partition_shap_token_attributions.csv"
    transcript_path = (
        output_dir / "partition_shap_transcript_summary.csv"
    )

    token_df.to_csv(token_path, index=False)
    transcript_df.to_csv(transcript_path, index=False)

    metadata = {
        "algorithm_requested": algorithm,
        "explainer_class": type(explainer).__name__,
        "max_evals": int(max_evals),
        "batch_size": int(batch_size),
        "number_of_transcripts": int(n_samples),
        "output_names": ["Control", "Dementia"],
        "primary_output": "Dementia",
        "primary_output_index": 1,
        "token_attribution_file": str(token_path),
        "transcript_summary_file": str(transcript_path),
    }

    with open(
        output_dir / "partition_shap_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved token attributions to: {token_path}")
    print(f"Saved transcript summary to: {transcript_path}")

    return token_df


def aggregate_partition_shap(
    token_attribution_files: Sequence[str | Path],
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Aggregate token importance across seeds.

    Tokens are normalized by stripping surrounding whitespace and converting
    to lowercase. Both signed and absolute Dementia attributions are retained.
    """

    frames = []

    for seed, path in enumerate(token_attribution_files):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(
            path,
            keep_default_na=False,
        )
        frame["seed"] = seed
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    combined["normalized_token"] = (
        combined["token"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    combined = combined.loc[
        combined["normalized_token"].ne("")
    ].copy()

    aggregated = (
        combined.groupby("normalized_token", as_index=False)
        .agg(
            mean_dementia_shap=(
                "dementia_shap",
                "mean",
            ),
            mean_absolute_dementia_shap=(
                "absolute_dementia_shap",
                "mean",
            ),
            standard_deviation_absolute_dementia_shap=(
                "absolute_dementia_shap",
                "std",
            ),
            occurrence_count=(
                "absolute_dementia_shap",
                "size",
            ),
            transcript_count=("id", "nunique"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(
            "mean_absolute_dementia_shap",
            ascending=False,
        )
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_path, index=False)

    return aggregated
