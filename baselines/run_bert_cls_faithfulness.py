"""
Deletion/insertion faithfulness tests for BERT-CLS attribution.

This script evaluates whether high-attribution token occurrences actually
control the model's dementia probability.

Deletion:
    Start from the original transcript.
    Mask top-ranked attributed tokens.
    Measure:
        probability_drop = P_original(dementia) - P_masked(dementia)

Insertion:
    Start from a baseline where all reportable tokens are masked.
    Restore top-ranked attributed tokens.
    Measure:
        probability_recovered = P_restored(dementia) - P_baseline(dementia)

For BERT-CLS there is no diagnosis [MASK] token to preserve. We only avoid
masking [CLS], [SEP], [PAD], and other special tokens because the attribution
CSV is expected to contain only reportable tokens when using the clean file.
"""

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_list(value):
    """Parse JSON/list-like strings from CSV."""
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
    except Exception as exc:
        raise ValueError(f"Could not parse list-like value: {value[:100]}") from exc


def parse_fractions(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def get_device(device_arg: str, gpu_num: int = 0) -> torch.device:
    """Resolve device."""
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device(f"cuda:{gpu_num}")

    if device_arg == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    # auto
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_num}")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model_and_tokenizer(model_dir: Path, device: torch.device):
    """Load saved BERT-CLS model and tokenizer."""
    hf_model_dir = model_dir / "hf_model"
    tokenizer_dir = model_dir / "tokenizer"

    if not hf_model_dir.exists():
        raise FileNotFoundError(f"Missing HuggingFace model directory: {hf_model_dir}")

    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Missing tokenizer directory: {tokenizer_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(hf_model_dir))
    model.to(device)
    model.eval()

    return model, tokenizer


@torch.no_grad()
def probability_dementia(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    """Compute P(dementia)."""
    outputs = model(
        input_ids=input_ids.long(),
        attention_mask=attention_mask.long(),
        return_dict=True,
    )
    probs = torch.softmax(outputs.logits, dim=-1)
    return float(probs[:, 1].detach().cpu().item())


def select_ranked_positions(
    attr_df: pd.DataFrame,
    ranking: str,
) -> List[int]:
    """
    Select rankable positions for a sample.

    ranking='positive':
        only tokens with attribution > 0, sorted descending.

    ranking='absolute':
        all tokens, sorted by absolute attribution descending.
    """
    df = attr_df.copy()

    if "is_reportable_token" in df.columns:
        df = df[df["is_reportable_token"].astype(bool)]

    if ranking == "positive":
        df = df[pd.to_numeric(df["attribution"], errors="coerce") > 0]
        df = df.sort_values("attribution", ascending=False)
    elif ranking == "absolute":
        if "abs_attribution" not in df.columns:
            df["abs_attribution"] = pd.to_numeric(df["attribution"], errors="coerce").abs()
        df = df.sort_values("abs_attribution", ascending=False)
    else:
        raise ValueError("ranking must be 'positive' or 'absolute'.")

    return pd.to_numeric(df["token_position"], errors="coerce").dropna().astype(int).tolist()


def mask_positions(
    input_ids: List[int],
    positions: Set[int],
    mask_token_id: int,
) -> List[int]:
    """Return copy of input IDs with selected positions masked."""
    out = list(input_ids)

    for pos in positions:
        if 0 <= pos < len(out):
            out[pos] = mask_token_id

    return out


def restore_positions(
    baseline_ids: List[int],
    original_ids: List[int],
    positions: Set[int],
) -> List[int]:
    """Restore selected positions from original input into baseline."""
    out = list(baseline_ids)

    for pos in positions:
        if 0 <= pos < len(out):
            out[pos] = original_ids[pos]

    return out


def tensorize(input_ids: List[int], attention_mask: List[int], device: torch.device):
    return (
        torch.tensor([input_ids], dtype=torch.long, device=device),
        torch.tensor([attention_mask], dtype=torch.long, device=device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deletion/insertion faithfulness for BERT-CLS baseline."
    )

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--attributions_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--ranking",
        type=str,
        required=True,
        choices=["positive", "absolute"],
    )
    parser.add_argument(
        "--fractions",
        type=str,
        default="0.05,0.10,0.20,0.30,0.50",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--gpu_num", type=int, default=0)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fractions = parse_fractions(args.fractions)

    device = get_device(args.device, gpu_num=args.gpu_num)
    print("Using device:", device)

    model_dir = Path(args.model_dir)
    model, tokenizer = load_model_and_tokenizer(model_dir, device)

    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer has no mask_token_id, but faithfulness uses [MASK].")

    mask_token_id = tokenizer.mask_token_id

    results_df = pd.read_csv(args.results_csv)
    attr_df = pd.read_csv(args.attributions_csv)

    required_result_cols = {"id", "labels", "pred_labels", "input_ids", "attention_mask"}
    missing_result = required_result_cols - set(results_df.columns)
    if missing_result:
        raise ValueError(f"Missing result columns: {sorted(missing_result)}")

    required_attr_cols = {"id", "token_position", "attribution"}
    missing_attr = required_attr_cols - set(attr_df.columns)
    if missing_attr:
        raise ValueError(f"Missing attribution columns: {sorted(missing_attr)}")

    results_df["id"] = results_df["id"].astype(str)
    attr_df["id"] = attr_df["id"].astype(str)

    attr_by_id: Dict[str, pd.DataFrame] = {
        sample_id: group.copy()
        for sample_id, group in attr_df.groupby("id")
    }

    deletion_rows = []
    insertion_rows = []

    for _, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Faithfulness"):
        sample_id = str(row["id"])
        label = int(row["labels"])
        pred_label = int(row["pred_labels"])

        input_ids = parse_list(row["input_ids"])
        attention_mask = parse_list(row["attention_mask"])

        if sample_id not in attr_by_id:
            print(f"Warning: no attributions for sample {sample_id}; skipping.")
            continue

        sample_attr = attr_by_id[sample_id]
        ranked_positions = select_ranked_positions(sample_attr, ranking=args.ranking)

        # All reportable positions in this attribution file.
        all_reportable_positions = set(
            pd.to_numeric(sample_attr["token_position"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )

        original_input_tensor, attention_tensor = tensorize(input_ids, attention_mask, device)
        original_prob = probability_dementia(model, original_input_tensor, attention_tensor)

        # Insertion baseline: mask all reportable tokens.
        baseline_ids = mask_positions(
            input_ids=input_ids,
            positions=all_reportable_positions,
            mask_token_id=mask_token_id,
        )
        baseline_input_tensor, _ = tensorize(baseline_ids, attention_mask, device)
        baseline_prob = probability_dementia(model, baseline_input_tensor, attention_tensor)

        n_ranked = len(ranked_positions)

        for fraction in fractions:
            if n_ranked == 0:
                selected_positions = set()
            else:
                k = max(1, int(math.ceil(fraction * n_ranked)))
                selected_positions = set(ranked_positions[:k])

            # Deletion: original -> mask selected.
            deleted_ids = mask_positions(
                input_ids=input_ids,
                positions=selected_positions,
                mask_token_id=mask_token_id,
            )
            deleted_input_tensor, _ = tensorize(deleted_ids, attention_mask, device)
            deleted_prob = probability_dementia(model, deleted_input_tensor, attention_tensor)

            probability_drop = original_prob - deleted_prob

            deletion_rows.append(
                {
                    "id": sample_id,
                    "label": label,
                    "pred_label": pred_label,
                    "ranking": args.ranking,
                    "fraction": fraction,
                    "n_ranked_tokens": n_ranked,
                    "n_selected_tokens": len(selected_positions),
                    "n_reportable_tokens": len(all_reportable_positions),
                    "original_probability_dementia": original_prob,
                    "perturbed_probability_dementia": deleted_prob,
                    "probability_drop": probability_drop,
                }
            )

            # Insertion: baseline with all reportable tokens masked -> restore selected.
            restored_ids = restore_positions(
                baseline_ids=baseline_ids,
                original_ids=input_ids,
                positions=selected_positions,
            )
            restored_input_tensor, _ = tensorize(restored_ids, attention_mask, device)
            restored_prob = probability_dementia(model, restored_input_tensor, attention_tensor)

            probability_recovered = restored_prob - baseline_prob

            insertion_rows.append(
                {
                    "id": sample_id,
                    "label": label,
                    "pred_label": pred_label,
                    "ranking": args.ranking,
                    "fraction": fraction,
                    "n_ranked_tokens": n_ranked,
                    "n_selected_tokens": len(selected_positions),
                    "n_reportable_tokens": len(all_reportable_positions),
                    "baseline_probability_dementia": baseline_prob,
                    "restored_probability_dementia": restored_prob,
                    "probability_recovered": probability_recovered,
                }
            )

    deletion_df = pd.DataFrame(deletion_rows)
    insertion_df = pd.DataFrame(insertion_rows)

    deletion_df.to_csv(output_dir / "deletion_curve.csv", index=False)
    insertion_df.to_csv(output_dir / "insertion_curve.csv", index=False)

    deletion_summary = (
        deletion_df.groupby("fraction")["probability_drop"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    insertion_summary = (
        insertion_df.groupby("fraction")["probability_recovered"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    deletion_summary.to_csv(output_dir / "deletion_summary.csv", index=False)
    insertion_summary.to_csv(output_dir / "insertion_summary.csv", index=False)

    print("\nSaved faithfulness outputs to:", output_dir)

    print("\nDeletion summary:")
    print(deletion_summary.to_string(index=False))

    print("\nInsertion summary:")
    print(insertion_summary.to_string(index=False))


if __name__ == "__main__":
    main()
