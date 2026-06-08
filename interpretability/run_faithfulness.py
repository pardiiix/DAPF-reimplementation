import argparse
import ast
import os

import numpy as np
import pandas as pd
import torch

from openprompt import PromptForClassification
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt.plms import load_plm


CLASS_LABELS = ["healthy", "dementia"]


def get_device(gpu_num=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_num}")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def parse_list_column(value):
    if isinstance(value, list):
        return value
    return ast.literal_eval(value)


def build_loss_ids(input_ids, tokenizer):
    loss_ids = torch.zeros_like(input_ids)

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer has no mask_token_id.")

    for i in range(input_ids.shape[0]):
        positions = torch.nonzero(input_ids[i] == mask_token_id, as_tuple=False).view(-1)
        if len(positions) == 0:
            raise ValueError(f"No [MASK] token found in sample {i}.")
        loss_ids[i, positions[0]] = 1

    return loss_ids


def load_prompt_model(args, device):
    model_path = os.path.join(args.off_line_model_dir, args.model_name)

    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, model_path)

    template_path = os.path.join(
        args.project_root,
        args.scripts_path,
        "manual_template.txt"
    )

    template = ManualTemplate(tokenizer=tokenizer).from_file(
        template_path,
        choice=args.template_id
    )

    verbalizer = ManualVerbalizer(
        classes=CLASS_LABELS,
        label_words={
            "dementia": ["dementia", "alzheimer's", "disfluent", "disordered"],
            "healthy": ["healthy"],
        },
        tokenizer=tokenizer,
    )

    prompt_model = PromptForClassification(
        plm=plm,
        template=template,
        verbalizer=verbalizer,
        freeze_plm=False,
        plm_eval_mode=False,
    )

    state = torch.load(args.checkpoint, map_location="cpu")
    prompt_model.load_state_dict(state)

    prompt_model.to(device)
    prompt_model.eval()

    return prompt_model, tokenizer


def predict_dementia_probability(prompt_model, tokenizer, input_ids_list, attention_mask_list, device):
    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask_list], dtype=torch.long).to(device)
    loss_ids = build_loss_ids(input_ids, tokenizer).to(device)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_ids": loss_ids,
    }

    with torch.no_grad():
        logits = prompt_model(batch)
        if isinstance(logits, dict):
            logits = logits["logits"]

        probs = torch.softmax(logits, dim=-1)

    return float(probs[0, 1].detach().cpu().item())


def get_ranked_positions(attr_df_for_sample, ranking="positive"):
    """
    ranking:
        positive = rank by strongest positive attribution toward dementia
        absolute = rank by largest absolute attribution magnitude
    """
    df = attr_df_for_sample.copy()

    if ranking == "positive":
        df = df[df["attribution"] > 0].copy()
        df = df.sort_values("attribution", ascending=False)

    elif ranking == "absolute":
        df = df.sort_values("abs_attribution", ascending=False)

    else:
        raise ValueError("ranking must be either 'positive' or 'absolute'.")

    # Deduplicate positions in case anything repeated
    positions = []
    seen = set()

    for pos in df["token_position"].tolist():
        pos = int(pos)
        if pos not in seen:
            positions.append(pos)
            seen.add(pos)

    return positions


def deletion_curve_for_sample(
    prompt_model,
    tokenizer,
    input_ids_list,
    attention_mask_list,
    ranked_positions,
    device,
    steps,
):
    """
    Deletion faithfulness:
    Mask top-ranked attributed tokens and observe whether P(dementia) drops.

    We mask rather than physically delete tokens because this preserves BERT/OpenPrompt
    sequence structure and keeps the prompt position stable.
    """
    rows = []

    original_prob = predict_dementia_probability(
        prompt_model,
        tokenizer,
        input_ids_list,
        attention_mask_list,
        device,
    )

    n_positions = len(ranked_positions)

    if n_positions == 0:
        return rows

    mask_token_id = tokenizer.mask_token_id

    for frac in steps:
        k = max(1, int(np.ceil(n_positions * frac)))
        selected = set(ranked_positions[:k])

        perturbed_ids = list(input_ids_list)

        for pos in selected:
            if 0 <= pos < len(perturbed_ids):
                perturbed_ids[pos] = mask_token_id

        perturbed_prob = predict_dementia_probability(
            prompt_model,
            tokenizer,
            perturbed_ids,
            attention_mask_list,
            device,
        )

        rows.append({
            "fraction": frac,
            "num_positions_available": n_positions,
            "num_masked": k,
            "original_probability": original_prob,
            "perturbed_probability": perturbed_prob,
            "probability_change": perturbed_prob - original_prob,
            "probability_drop": original_prob - perturbed_prob,
        })

    return rows


def insertion_curve_for_sample(
    prompt_model,
    tokenizer,
    input_ids_list,
    attention_mask_list,
    ranked_positions,
    all_reportable_positions,
    device,
    steps,
):
    """
    Insertion faithfulness:
    Start from a baseline where all reportable transcript tokens are masked,
    then restore top-ranked tokens and observe whether P(dementia) recovers.
    """
    rows = []

    original_prob = predict_dementia_probability(
        prompt_model,
        tokenizer,
        input_ids_list,
        attention_mask_list,
        device,
    )

    n_positions = len(ranked_positions)

    if n_positions == 0:
        return rows

    mask_token_id = tokenizer.mask_token_id

    baseline_ids = list(input_ids_list)

    for pos in all_reportable_positions:
        pos = int(pos)
        if 0 <= pos < len(baseline_ids):
            baseline_ids[pos] = mask_token_id

    baseline_prob = predict_dementia_probability(
        prompt_model,
        tokenizer,
        baseline_ids,
        attention_mask_list,
        device,
    )

    for frac in steps:
        k = max(1, int(np.ceil(n_positions * frac)))
        selected = set(ranked_positions[:k])

        perturbed_ids = list(baseline_ids)

        for pos in selected:
            if 0 <= pos < len(perturbed_ids):
                perturbed_ids[pos] = input_ids_list[pos]

        perturbed_prob = predict_dementia_probability(
            prompt_model,
            tokenizer,
            perturbed_ids,
            attention_mask_list,
            device,
        )

        rows.append({
            "fraction": frac,
            "num_positions_available": n_positions,
            "num_inserted": k,
            "original_probability": original_prob,
            "baseline_probability": baseline_prob,
            "perturbed_probability": perturbed_prob,
            "probability_recovered": perturbed_prob - baseline_prob,
            "recovery_fraction_of_original": (
                (perturbed_prob - baseline_prob) / (original_prob - baseline_prob)
                if abs(original_prob - baseline_prob) > 1e-12
                else np.nan
            ),
        })

    return rows


def run_faithfulness(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device(args.gpu_num)
    print(f"Using device: {device}")

    prompt_model, tokenizer = load_prompt_model(args, device)

    results_df = pd.read_csv(args.results_csv)
    attr_df = pd.read_csv(args.attributions_csv)

    required_results_cols = {
        "id",
        "labels",
        "pred_labels",
        "probas",
        "input_ids",
        "attention_mask",
    }

    required_attr_cols = {
        "id",
        "token_position",
        "token",
        "attribution",
        "abs_attribution",
    }

    missing_results = required_results_cols - set(results_df.columns)
    missing_attr = required_attr_cols - set(attr_df.columns)

    if missing_results:
        raise ValueError(f"Missing required columns in results CSV: {missing_results}")

    if missing_attr:
        raise ValueError(f"Missing required columns in attributions CSV: {missing_attr}")

    if args.max_samples is not None:
        keep_ids = results_df["id"].head(args.max_samples).tolist()
        results_df = results_df[results_df["id"].isin(keep_ids)].copy()
        attr_df = attr_df[attr_df["id"].isin(keep_ids)].copy()

    steps = tuple(float(x) for x in args.steps.split(","))

    deletion_rows = []
    insertion_rows = []

    for _, row in results_df.iterrows():
        sample_id = row["id"]

        input_ids_list = parse_list_column(row["input_ids"])
        attention_mask_list = parse_list_column(row["attention_mask"])

        sample_attr = attr_df[attr_df["id"] == sample_id].copy()

        if sample_attr.empty:
            print(f"Warning: no attribution rows found for {sample_id}; skipping.")
            continue

        ranked_positions = get_ranked_positions(
            sample_attr,
            ranking=args.ranking
        )

        all_reportable_positions = sorted(
            set(int(x) for x in sample_attr["token_position"].tolist())
        )

        deletion_sample_rows = deletion_curve_for_sample(
            prompt_model=prompt_model,
            tokenizer=tokenizer,
            input_ids_list=input_ids_list,
            attention_mask_list=attention_mask_list,
            ranked_positions=ranked_positions,
            device=device,
            steps=steps,
        )

        for r in deletion_sample_rows:
            r.update({
                "id": sample_id,
                "label": row["labels"],
                "pred_label": row["pred_labels"],
                "ranking": args.ranking,
            })
            deletion_rows.append(r)

        insertion_sample_rows = insertion_curve_for_sample(
            prompt_model=prompt_model,
            tokenizer=tokenizer,
            input_ids_list=input_ids_list,
            attention_mask_list=attention_mask_list,
            ranked_positions=ranked_positions,
            all_reportable_positions=all_reportable_positions,
            device=device,
            steps=steps,
        )

        for r in insertion_sample_rows:
            r.update({
                "id": sample_id,
                "label": row["labels"],
                "pred_label": row["pred_labels"],
                "ranking": args.ranking,
            })
            insertion_rows.append(r)

    deletion_df = pd.DataFrame(deletion_rows)
    insertion_df = pd.DataFrame(insertion_rows)

    deletion_path = os.path.join(args.output_dir, "deletion_curve.csv")
    insertion_path = os.path.join(args.output_dir, "insertion_curve.csv")

    deletion_df.to_csv(deletion_path, index=False)
    insertion_df.to_csv(insertion_path, index=False)

    summary_rows = []

    if not deletion_df.empty:
        deletion_summary = (
            deletion_df.groupby("fraction")
            .agg(
                mean_probability_drop=("probability_drop", "mean"),
                std_probability_drop=("probability_drop", "std"),
                mean_perturbed_probability=("perturbed_probability", "mean"),
                n=("id", "nunique"),
            )
            .reset_index()
        )

        deletion_summary["method"] = "deletion"
        summary_rows.append(deletion_summary)

    if not insertion_df.empty:
        insertion_summary = (
            insertion_df.groupby("fraction")
            .agg(
                mean_probability_recovered=("probability_recovered", "mean"),
                std_probability_recovered=("probability_recovered", "std"),
                mean_perturbed_probability=("perturbed_probability", "mean"),
                n=("id", "nunique"),
            )
            .reset_index()
        )

        insertion_summary["method"] = "insertion"
        summary_rows.append(insertion_summary)

    if summary_rows:
        summary_df = pd.concat(summary_rows, ignore_index=True, sort=False)
    else:
        summary_df = pd.DataFrame()

    summary_path = os.path.join(args.output_dir, "faithfulness_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved deletion curve to: {deletion_path}")
    print(f"Saved insertion curve to: {insertion_path}")
    print(f"Saved faithfulness summary to: {summary_path}")

    print("\nDeletion summary:")
    if deletion_df.empty:
        print("No deletion rows.")
    else:
        print(
            deletion_df.groupby("fraction")["probability_drop"]
            .agg(["mean", "std", "count"])
        )

    print("\nInsertion summary:")
    if insertion_df.empty:
        print("No insertion rows.")
    else:
        print(
            insertion_df.groupby("fraction")["probability_recovered"]
            .agg(["mean", "std", "count"])
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", default="./")
    parser.add_argument("--scripts_path", default="template/")
    parser.add_argument("--off_line_model_dir", required=True)
    parser.add_argument("--model", default="bert")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--template_id", type=int, default=7)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--attributions_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--steps", default="0.05,0.10,0.20,0.30,0.50")
    parser.add_argument(
        "--ranking",
        default="positive",
        choices=["positive", "absolute"],
        help="positive ranks only positive dementia-directed attributions; absolute ranks by attribution magnitude."
    )
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    run_faithfulness(args)


if __name__ == "__main__":
    main()
