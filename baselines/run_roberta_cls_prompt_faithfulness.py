"""Deletion/insertion faithfulness for RoBERTa-CLS+Prompt transcript attributions.

Reads the *clean* transcript-only CSV from run_roberta_cls_prompt_attribution.
Preserves all fixed prompt tokens and the diagnosis <mask> position. As with
prior CLS faithfulness controls, perturbations replace rankable transcript
tokens with <mask> and measure changes in P(dementia).

Deletion: P(original) - P(with selected transcript tokens masked)
Insertion: P(with selected transcript tokens restored) - P(all reportable transcript tokens masked)

Note that the faithfulness <mask> perturbation differs from the IG <unk>
reference. Signed dementia-supporting IG attributions are most directly
interpreted with the 'positive' ranking; 'absolute' mixes signs.
"""

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_list(value):
    if isinstance(value, (list, np.ndarray)):
        return value.tolist() if isinstance(value, np.ndarray) else value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid list: {value[:100]}") from exc


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


def parse_fractions(raw):
    fractions = [float(x) for x in raw.split(",") if x.strip()]
    if not fractions or any(not (0 < x <= 1) for x in fractions):
        raise ValueError("Each fraction must be in (0,1]")
    return fractions


@torch.no_grad()
def dementia_prob(model, ids, mask, device):
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    mask_t = torch.tensor([mask], dtype=torch.long, device=device)
    logits = model(input_ids=ids_t, attention_mask=mask_t, return_dict=True).logits
    return float(torch.softmax(logits, dim=-1)[0, 1].cpu().item())


def perturb(ids, positions, token_id):
    updated = list(ids)
    for pos in positions:
        updated[pos] = token_id
    return updated


def main():
    p = argparse.ArgumentParser(description="RoBERTa-CLS+Prompt transcript-only faithfulness")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--results_csv", required=True)
    p.add_argument("--attributions_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ranking", choices=["positive", "absolute"], required=True)
    p.add_argument("--fractions", default="0.05,0.10,0.20,0.30,0.50")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--gpu_num", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=None, help="Smoke-test limit")
    args = p.parse_args()

    device = get_device(args.device, args.gpu_num)
    model_dir = Path(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir / "hf_model"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    if model.config.model_type != "roberta":
        raise ValueError(f"Expected RoBERTa; found {model.config.model_type!r}")
    model.to(device).eval()
    if tokenizer.mask_token_id is None:
        raise ValueError("Missing RoBERTa <mask> token")

    results = pd.read_csv(args.results_csv)
    attr = pd.read_csv(args.attributions_csv)
    if args.max_samples is not None:
        results = results.head(args.max_samples).copy()
    required_results = {"id", "labels", "pred_labels", "input_ids", "attention_mask", "diagnosis_mask_position"}
    required_attr = {"id", "token_position", "token_id", "attribution", "is_reportable_token", "token_region"}
    if required_results - set(results.columns):
        raise ValueError(f"Missing test_results columns: {sorted(required_results - set(results.columns))}")
    if required_attr - set(attr.columns):
        raise ValueError(
            "Attribution file lacks token provenance; use the new "
            "run_roberta_cls_prompt_attribution CLEAN CSV. Missing: "
            f"{sorted(required_attr - set(attr.columns))}"
        )
    if results.empty or attr.empty:
        raise ValueError("Empty results/attribution data")
    if not (attr["token_region"] == "transcript").all():
        raise ValueError("Clean attribution CSV unexpectedly includes fixed prompt/special tokens")
    # pandas parses this column as bool when produced by the IG script.
    flag = attr["is_reportable_token"].astype(str).str.lower()
    if not flag.eq("true").all():
        raise ValueError("Attribution CSV includes nonreportable tokens")
    if attr.duplicated(["id", "token_position"]).any():
        raise ValueError("Duplicate token positions found in attribution CSV")

    results["id"] = results["id"].astype(str)
    attr["id"] = attr["id"].astype(str)
    by_id = {sample_id: group for sample_id, group in attr.groupby("id")}
    fractions = parse_fractions(args.fractions)
    deletion_rows, insertion_rows = [], []
    print("Using device:", device, flush=True)
    print(f"Faithfulness ranking: {args.ranking}; perturbing only clean transcript tokens", flush=True)

    for _, row in tqdm(results.iterrows(), total=len(results), desc="Faithfulness"):
        sample_id = str(row["id"])
        if sample_id not in by_id:
            raise ValueError(f"No clean attribution rows for sample {sample_id}")
        ids = [int(x) for x in parse_list(row["input_ids"])]
        attention_mask = [int(x) for x in parse_list(row["attention_mask"])]
        diag_pos = int(row["diagnosis_mask_position"])
        if not (0 <= diag_pos < len(ids) and ids[diag_pos] == tokenizer.mask_token_id):
            raise ValueError(f"Bad diagnosis mask for sample {sample_id}")
        if ids.count(tokenizer.mask_token_id) != 1:
            raise ValueError(f"Expected exactly one original <mask> for sample {sample_id}")

        sample_attr = by_id[sample_id].copy()
        sample_attr["attribution"] = pd.to_numeric(sample_attr["attribution"], errors="raise")
        positions = [int(p) for p in sample_attr["token_position"].tolist()]
        token_ids = [int(t) for t in sample_attr["token_id"].tolist()]
        for pos, tok_id in zip(positions, token_ids):
            if not (0 <= pos < len(ids)) or ids[pos] != tok_id:
                raise ValueError(f"Token/position mismatch in sample {sample_id}, position {pos}")
            if not attention_mask[pos] or pos == diag_pos or ids[pos] in tokenizer.all_special_ids:
                raise ValueError(f"Attempted to mask a fixed/special token at {sample_id}:{pos}")
        all_reportable = set(positions)

        if args.ranking == "positive":
            ranked = sample_attr[sample_attr["attribution"] > 0].sort_values("attribution", ascending=False)
        else:
            sample_attr["abs_attribution"] = sample_attr["attribution"].abs()
            ranked = sample_attr.sort_values("abs_attribution", ascending=False)
        ranked_positions = [int(x) for x in ranked["token_position"]]
        orig_prob = dementia_prob(model, ids, attention_mask, device)
        baseline_ids = perturb(ids, all_reportable, tokenizer.mask_token_id)
        baseline_prob = dementia_prob(model, baseline_ids, attention_mask, device)

        for fraction in fractions:
            n_ranked = len(ranked_positions)
            k = max(1, math.ceil(fraction * n_ranked)) if n_ranked else 0
            chosen = set(ranked_positions[:k])
            deleted_ids = perturb(ids, chosen, tokenizer.mask_token_id)
            p_deleted = dementia_prob(model, deleted_ids, attention_mask, device)
            restored_ids = list(baseline_ids)
            for pos in chosen:
                restored_ids[pos] = ids[pos]
            p_restored = dementia_prob(model, restored_ids, attention_mask, device)
            deletion_rows.append({
                "id": sample_id, "label": int(row["labels"]), "pred_label": int(row["pred_labels"]),
                "ranking": args.ranking, "fraction": fraction, "n_ranked_tokens": n_ranked,
                "n_selected_tokens": len(chosen), "n_reportable_tokens": len(all_reportable),
                "original_probability_dementia": orig_prob,
                "perturbed_probability_dementia": p_deleted,
                "probability_drop": orig_prob - p_deleted,
            })
            insertion_rows.append({
                "id": sample_id, "label": int(row["labels"]), "pred_label": int(row["pred_labels"]),
                "ranking": args.ranking, "fraction": fraction, "n_ranked_tokens": n_ranked,
                "n_selected_tokens": len(chosen), "n_reportable_tokens": len(all_reportable),
                "baseline_probability_dementia": baseline_prob,
                "restored_probability_dementia": p_restored,
                "probability_recovered": p_restored - baseline_prob,
            })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    deletion = pd.DataFrame(deletion_rows)
    insertion = pd.DataFrame(insertion_rows)
    deletion.to_csv(out / "deletion_curve.csv", index=False)
    insertion.to_csv(out / "insertion_curve.csv", index=False)
    dsum = deletion.groupby("fraction")["probability_drop"].agg(["mean", "std", "count"]).reset_index()
    isum = insertion.groupby("fraction")["probability_recovered"].agg(["mean", "std", "count"]).reset_index()
    dsum.to_csv(out / "deletion_summary.csv", index=False)
    isum.to_csv(out / "insertion_summary.csv", index=False)
    print("Saved outputs to:", out)
    print("Deletion summary:\n" + dsum.to_string(index=False))
    print("Insertion summary:\n" + isum.to_string(index=False))


if __name__ == "__main__":
    main()
