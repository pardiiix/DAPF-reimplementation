"""Transcript-only Layer Integrated Gradients for RoBERTa-CLS+Prompt.

Run after baselines.train_roberta_cls_prompt has saved the HuggingFace model,
its tokenizer, and test_results.csv. Unlike RoBERTa-CLS without a prompt,
the fixed prompt prefix and the diagnosis <mask> suffix must remain unchanged.

Attributions target the dementia logit (class index 1), and the reference
replaces ONLY transcript token IDs with <unk>. This avoids changing RoBERTa
position-ID behavior through the use of <pad> as a replacement.

The clean CSV contains only reportable transcript token occurrences. Raw CSV
contains all positions and an explicit token_region field. To identify regions,
the script reconstructs the *exact* offset/truncation procedure from
train_roberta_cls_prompt.py and verifies alignment against saved input_ids.
"""

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from captum.attr import LayerIntegratedGradients
except ImportError as exc:
    raise ImportError("Captum required: pip install captum") from exc


def parse_list(value):
    if isinstance(value, (list, np.ndarray)):
        return value.tolist() if isinstance(value, np.ndarray) else value
    if not isinstance(value, str):
        return value
    value = value.strip()
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Cannot parse saved list: {value[:100]}") from exc


def get_device(name, gpu_num=0):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return torch.device(f"cuda:{gpu_num}")
    if name == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS not available")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_num}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_dir, device):
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir / "hf_model"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    if model.config.model_type != "roberta":
        raise ValueError(f"Expected RoBERTa, got {model.config.model_type!r}")
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer is required for accurate character offsets")
    if tokenizer.unk_token_id is None or tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer needs both <unk> and <mask> token IDs")
    model.to(device).eval()
    return model, tokenizer


def saved_token_regions(row, tokenizer, input_ids, attention_mask):
    """Recreate trained input with offsets; return original token provenance.

    This intentionally mirrors PromptedTranscriptDataset.__getitem__: tokenize
    the entire prompted string, then preserve the right-hand diagnosis suffix
    and the leftmost preceding tokens whenever the full length exceeds 512.
    """
    prompt = str(row["prompted_text"])
    transcript = str(row["text"]).strip()
    domain = str(row["domain"]).strip()
    prefix = f"Participant's narration on {domain}. "
    suffix = " Patient has diagnosis <mask>."
    expected = prefix + transcript + suffix
    if prompt != expected:
        raise ValueError(
            f"Sample {row['id']}: unexpected template. Expected exact dapf_template7 "
            "with prefix and trailing diagnosis <mask>; cannot safely identify transcript tokens."
        )

    encoded = tokenizer(
        prompt, truncation=False, padding=False, return_offsets_mapping=True
    )
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if len(ids) > len(input_ids):
        suffix_start_char = len(prompt) - len(suffix)
        suffix_token_start = next(
            i for i, (start, end) in enumerate(offsets) if end > suffix_start_char
        )
        suffix_ids = ids[suffix_token_start:]
        n_head = len(input_ids) - len(suffix_ids)
        if not (1 <= n_head <= suffix_token_start):
            raise ValueError(f"Sample {row['id']}: invalid prompt-preserving truncation")
        ids = ids[:n_head] + ids[suffix_token_start:]
        offsets = offsets[:n_head] + offsets[suffix_token_start:]

    if ids != list(input_ids[: len(ids)]):
        raise ValueError(
            f"Sample {row['id']}: reconstructed token IDs differ from saved test_results. "
            "Do not calculate attributions with misaligned transcript positions."
        )
    if any(int(v) != 1 for v in attention_mask[: len(ids)]):
        raise ValueError(f"Sample {row['id']}: attention mask does not match tokenized input")
    if any(int(v) != 0 for v in attention_mask[len(ids) :]):
        raise ValueError(f"Sample {row['id']}: unexpected nonpadding attention positions")
    if any(int(v) != tokenizer.pad_token_id for v in input_ids[len(ids) :]):
        raise ValueError(f"Sample {row['id']}: unexpected padding IDs")

    left = len(prefix)
    right = left + len(transcript)
    special_ids = set(tokenizer.all_special_ids)
    regions = []
    for tok_id, (start, end) in zip(ids, offsets):
        if tok_id == tokenizer.mask_token_id:
            region = "diagnosis_mask"
        elif tok_id in special_ids:
            region = "special"
        elif end > left and start < right:
            # Overlap rather than strict containment keeps the first transcript
            # word when its Ġ-leading space falls in the fixed prefix.
            region = "transcript"
        elif end <= left:
            region = "prompt_prefix"
        else:
            region = "prompt_suffix"
        regions.append(region)
    regions.extend(["padding"] * (len(input_ids) - len(regions)))

    mask_pos = int(row["diagnosis_mask_position"])
    if not (0 <= mask_pos < len(regions)):
        raise ValueError(f"Sample {row['id']}: invalid diagnosis_mask_position")
    if regions[mask_pos] != "diagnosis_mask" or input_ids[mask_pos] != tokenizer.mask_token_id:
        raise ValueError(f"Sample {row['id']}: <mask> not at saved diagnosis position")
    if sum(int(x == tokenizer.mask_token_id) for x in input_ids) != 1:
        raise ValueError(f"Sample {row['id']}: expected exactly one diagnosis <mask>")
    if "transcript" not in regions:
        raise ValueError(f"Sample {row['id']}: no transcript tokens survived truncation")
    return regions


def is_reportable_token(token, attention_value, region):
    return (
        int(attention_value) == 1
        and region == "transcript"
        and bool(re.search(r"[A-Za-z0-9]", token))
    )


def attribute_one(model, tokenizer, input_ids, attention_mask, regions, n_steps, internal_batch_size, target_class):
    # Maintain fixed prompt, diagnosis <mask>, structural tokens and padding.
    baseline_ids = input_ids.clone()
    transcript_positions = torch.tensor(
        [r == "transcript" for r in regions], dtype=torch.bool, device=input_ids.device
    ).unsqueeze(0)
    baseline_ids[transcript_positions] = tokenizer.unk_token_id

    def forward_func(ids, mask):
        return model(input_ids=ids.long(), attention_mask=mask.long(), return_dict=True).logits

    lig = LayerIntegratedGradients(forward_func, model.get_input_embeddings())
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=target_class,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    delta = float(delta.detach().cpu().reshape(-1)[0].item())
    return scores, delta


@torch.no_grad()
def predict_probability(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return float(torch.softmax(logits, dim=-1)[0, 1].cpu().item())


def main():
    parser = argparse.ArgumentParser(description="RoBERTa-CLS+Prompt transcript-only IG")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_class", type=int, default=1, choices=[0, 1])
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--internal_batch_size", type=int, default=4)
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=None, help="Smoke-test limit")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--gpu_num", type=int, default=0)
    args = parser.parse_args()

    if args.n_steps < 2 or args.internal_batch_size < 1:
        parser.error("n_steps must be >= 2 and internal_batch_size >= 1")
    device = get_device(args.device, args.gpu_num)
    model, tokenizer = load_model_and_tokenizer(Path(args.model_dir), device)
    df = pd.read_csv(args.results_csv)
    required = {"id", "labels", "pred_labels", "input_ids", "attention_mask", "diagnosis_mask_position", "prompted_text", "text", "domain"}
    if required - set(df.columns):
        raise ValueError(f"Missing CSV columns: {sorted(required - set(df.columns))}")
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()
    if df.empty:
        raise ValueError("No samples to attribute")
    print("Using device:", device, flush=True)
    print("IG reference: <unk> at transcript tokens; fixed prompt and <mask> preserved", flush=True)
    print("Attribution output: dementia logit", flush=True)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Attributing"):
        ids = parse_list(row["input_ids"])
        mask = parse_list(row["attention_mask"])
        regions = saved_token_regions(row, tokenizer, ids, mask)
        if len(ids) != len(mask) or len(ids) != len(regions):
            raise ValueError(f"Sample {row['id']}: length mismatch")
        ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        mask_tensor = torch.tensor([mask], dtype=torch.long, device=device)
        score, delta = attribute_one(
            model, tokenizer, ids_tensor, mask_tensor, regions,
            args.n_steps, args.internal_batch_size, args.target_class
        )
        prob = float(row["probability_dementia"]) if (
            "probability_dementia" in row and pd.notna(row["probability_dementia"])
        ) else predict_probability(model, ids_tensor, mask_tensor)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        for pos, (tok_id, attn, token, attr, region) in enumerate(
            zip(ids, mask, tokens, score, regions)
        ):
            rows.append({
                "id": str(row["id"]),
                "label": int(row["labels"]),
                "pred_label": int(row["pred_labels"]),
                "probability_dementia": prob,
                "token_position": pos,
                "token_id": int(tok_id),
                "token": token,
                "attribution": float(attr),
                "abs_attribution": float(abs(attr)),
                "attention_mask": int(attn),
                "token_region": region,
                "is_reportable_token": is_reportable_token(token, attn, region),
                "convergence_delta": delta,
            })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_df = pd.DataFrame(rows)
    clean_df = all_df[all_df["is_reportable_token"]].copy()
    all_df.to_csv(out / "integrated_gradients_token_attributions.csv", index=False)
    clean_df.to_csv(out / "integrated_gradients_token_attributions_clean.csv", index=False)
    for name, dataframe in [("raw", all_df), ("clean", clean_df)]:
        dataframe.nlargest(args.top_n, "attribution").to_csv(out / f"top_positive_attributions_{name}.csv", index=False)
        dataframe.nsmallest(args.top_n, "attribution").to_csv(out / f"top_negative_attributions_{name}.csv", index=False)

    by_sample_delta = all_df.groupby("id")["convergence_delta"].first().abs()
    print(f"Saved to: {out}")
    print(f"Samples: {len(df)}; raw positions: {len(all_df)}; clean transcript positions: {len(clean_df)}")
    print(f"Convergence |delta|: mean={by_sample_delta.mean():.6g}; max={by_sample_delta.max():.6g}")
    print("Clean-token regions:", clean_df["token_region"].value_counts().to_dict())
    if not clean_df.empty:
        print("Top positive transcript tokens:")
        print(clean_df.nlargest(10, "attribution")[["id", "token_position", "token", "attribution"]].to_string(index=False))
        print("Top negative transcript tokens:")
        print(clean_df.nsmallest(10, "attribution")[["id", "token_position", "token", "attribution"]].to_string(index=False))


if __name__ == "__main__":
    main()
