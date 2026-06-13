"""
Integrated Gradients attribution for the standard BERT-CLS baseline.

This script assumes that baselines/train_bert_cls.py has already produced:

    output_bert_cls_multiseed/seed_X/
    ├── hf_model/
    ├── tokenizer/
    └── test_results.csv

It computes token-level Integrated Gradients attributions with respect to
the dementia class probability/logit for a standard BERT sequence classifier.

Outputs:
    integrated_gradients_token_attributions.csv
    integrated_gradients_token_attributions_clean.csv
    top_positive_attributions_raw.csv
    top_negative_attributions_raw.csv
    top_positive_attributions_clean.csv
    top_negative_attributions_clean.csv

The output format intentionally mirrors the DAPF attribution output so that
the existing aggregation script can be reused.
"""

import argparse
import ast
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from captum.attr import LayerIntegratedGradients
except ImportError as exc:
    raise ImportError(
        "Captum is required for Integrated Gradients. Install with: pip install captum"
    ) from exc


SPECIAL_TOKENS = {
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[MASK]",
    "[UNK]",
    "<s>",
    "</s>",
    "<pad>",
    "<mask>",
}


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


def is_punctuation_only(token: str) -> bool:
    """Return True if token contains no alphanumeric characters."""
    return not bool(re.search(r"[A-Za-z0-9]", token))


def is_reportable_token(token: str, attention_value: int) -> bool:
    """Decide whether a token should be included in the cleaned attribution file."""
    if int(attention_value) == 0:
        return False

    if token in SPECIAL_TOKENS:
        return False

    if token.strip() == "":
        return False

    if is_punctuation_only(token.replace("##", "")):
        return False

    return True


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


def make_baseline_input_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    """
    Build a baseline input for Integrated Gradients.

    We replace non-special, attended tokens with [PAD], while preserving
    [CLS], [SEP], padding positions, and any other special tokens.
    """
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    mask_id = tokenizer.mask_token_id
    unk_id = tokenizer.unk_token_id

    baseline = torch.full_like(input_ids, fill_value=pad_id)

    preserve = attention_mask.eq(0)

    for special_id in [cls_id, sep_id, mask_id, unk_id, pad_id]:
        if special_id is not None:
            preserve = preserve | input_ids.eq(special_id)

    baseline[preserve] = input_ids[preserve]

    return baseline


def attribution_for_one_sample(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_class: int,
    n_steps: int,
    internal_batch_size: int,
) -> Tuple[np.ndarray, float]:
    """
    Compute Layer Integrated Gradients for one sample.

    Returns:
        token_attributions: [seq_len]
        convergence_delta: float
    """

    def forward_func(forward_input_ids, forward_attention_mask):
        outputs = model(
            input_ids=forward_input_ids.long(),
            attention_mask=forward_attention_mask.long(),
            return_dict=True,
        )
        return outputs.logits

    # Use the word embedding layer as attribution layer.
    lig = LayerIntegratedGradients(
        forward_func,
        model.bert.embeddings.word_embeddings,
    )

    baseline_ids = make_baseline_input_ids(input_ids, attention_mask, tokenizer)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=target_class,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )

    # Shape: [1, seq_len, hidden_dim] -> [seq_len]
    token_attributions = attributions.sum(dim=-1).squeeze(0)

    return (
        token_attributions.detach().cpu().numpy(),
        float(delta.detach().cpu().reshape(-1)[0].item()),
    )


@torch.no_grad()
def predict_probability(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
    """Return P(dementia)."""
    outputs = model(
        input_ids=input_ids.long(),
        attention_mask=attention_mask.long(),
        return_dict=True,
    )
    probs = torch.softmax(outputs.logits, dim=-1)
    return float(probs[:, 1].detach().cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Integrated Gradients attribution for BERT-CLS baseline."
    )

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--internal_batch_size", type=int, default=8)
    parser.add_argument("--top_n", type=int, default=100)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Use cpu if Captum has trouble on MPS.",
    )
    parser.add_argument("--gpu_num", type=int, default=0)

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device, gpu_num=args.gpu_num)
    print("Using device:", device)

    model, tokenizer = load_model_and_tokenizer(model_dir, device)

    df = pd.read_csv(args.results_csv)

    required_cols = {"id", "labels", "pred_labels", "input_ids", "attention_mask"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in results CSV: {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Attributing"):
        sample_id = str(row["id"])
        label = int(row["labels"])
        pred_label = int(row["pred_labels"])

        input_ids_list = parse_list(row["input_ids"])
        attention_mask_list = parse_list(row["attention_mask"])

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        attention_mask = torch.tensor([attention_mask_list], dtype=torch.long, device=device)

        probability_dementia = (
            float(row["probability_dementia"])
            if "probability_dementia" in row and not pd.isna(row["probability_dementia"])
            else predict_probability(model, input_ids, attention_mask)
        )

        token_attributions, convergence_delta = attribution_for_one_sample(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_class=args.target_class,
            n_steps=args.n_steps,
            internal_batch_size=args.internal_batch_size,
        )

        tokens = tokenizer.convert_ids_to_tokens(input_ids_list)

        for pos, (token_id, attn, token, attr) in enumerate(
            zip(input_ids_list, attention_mask_list, tokens, token_attributions)
        ):
            reportable = is_reportable_token(token, int(attn))

            rows.append(
                {
                    "id": sample_id,
                    "label": label,
                    "pred_label": pred_label,
                    "probability_dementia": probability_dementia,
                    "token_position": pos,
                    "token_id": int(token_id),
                    "token": token,
                    "attribution": float(attr),
                    "abs_attribution": float(abs(attr)),
                    "attention_mask": int(attn),
                    "is_reportable_token": bool(reportable),
                    "convergence_delta": convergence_delta,
                }
            )

    attr_df = pd.DataFrame(rows)
    clean_df = attr_df[attr_df["is_reportable_token"]].copy()

    raw_path = output_dir / "integrated_gradients_token_attributions.csv"
    clean_path = output_dir / "integrated_gradients_token_attributions_clean.csv"

    attr_df.to_csv(raw_path, index=False)
    clean_df.to_csv(clean_path, index=False)

    attr_df.sort_values("attribution", ascending=False).head(args.top_n).to_csv(
        output_dir / "top_positive_attributions_raw.csv",
        index=False,
    )
    attr_df.sort_values("attribution", ascending=True).head(args.top_n).to_csv(
        output_dir / "top_negative_attributions_raw.csv",
        index=False,
    )

    clean_df.sort_values("attribution", ascending=False).head(args.top_n).to_csv(
        output_dir / "top_positive_attributions_clean.csv",
        index=False,
    )
    clean_df.sort_values("attribution", ascending=True).head(args.top_n).to_csv(
        output_dir / "top_negative_attributions_clean.csv",
        index=False,
    )

    print("\nSaved attribution outputs to:", output_dir)
    print("Raw attribution rows:", len(attr_df))
    print("Clean attribution rows:", len(clean_df))
    print("Raw file:", raw_path)
    print("Clean file:", clean_path)

    if len(clean_df) > 0:
        print("\nTop positive clean attributions:")
        print(
            clean_df.sort_values("attribution", ascending=False)
            .head(20)[["id", "token_position", "token", "attribution"]]
            .to_string(index=False)
        )

        print("\nTop negative clean attributions:")
        print(
            clean_df.sort_values("attribution", ascending=True)
            .head(20)[["id", "token_position", "token", "attribution"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
