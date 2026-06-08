import argparse
import ast
import os

import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients

from openprompt import PromptForClassification
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt.plms import load_plm


CLASS_LABELS = ["healthy", "dementia"]


# Prompt/template/special tokens that should not be interpreted as participant speech.
# These are still saved in the raw attribution file, but excluded from the cleaned file.
PROMPT_TOKENS_TO_EXCLUDE = {
    # Special/model tokens
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[MASK]",

    # Prompt/template tokens
    "participant",
    "'",
    "s",
    "narration",
    "on",
    "picture",
    "description",
    "patient",
    "has",
    "diagnosis",

    # Punctuation and transcript-control symbols
    ".",
    ",",
    ";",
    ":",
    "!",
    "?",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "<",
    ">",
    "+",
    "&",
    "/",
    "\\",
    "=",
    "*",
    '"',

    # Common CHAT/transcription fragments or artifacts
    "ex",
}


def get_device(gpu_num=0):
    """
    Select CUDA if available, then Apple Silicon MPS, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_num}")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def parse_list_column(value):
    """
    Convert CSV string form of a Python list back into a list of ints.
    """
    if isinstance(value, list):
        return value

    return ast.literal_eval(value)


def is_reportable_token(token):
    """
    Keep tokens that are plausibly participant-produced lexical/disfluency tokens.
    Exclude prompt tokens, model special tokens, punctuation, and obvious transcript artifacts.
    """
    if not isinstance(token, str):
        return False

    token = token.strip()

    if token in PROMPT_TOKENS_TO_EXCLUDE:
        return False

    if token.startswith("[") and token.endswith("]"):
        return False

    # Remove BERT subword fragments that are only one or two characters after ##
    # e.g., ##c, ##s. Longer subwords such as ##ing may still be linguistically useful.
    if token.startswith("##") and len(token.replace("##", "")) <= 2:
        return False

    # Remove isolated single-character alphabetic fragments like f, g, o.
    # Keep "i" and "a" because they are real English tokens.
    if len(token) == 1 and token.isalpha() and token not in {"i", "a"}:
        return False

    return True


def build_loss_ids(input_ids, tokenizer):
    """
    Build OpenPrompt-style loss_ids by marking the [MASK] token position.
    """
    loss_ids = torch.zeros_like(input_ids)

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError(
            "Tokenizer has no mask_token_id. This attribution runner currently expects BERT-style MLM."
        )

    for i in range(input_ids.shape[0]):
        positions = torch.nonzero(input_ids[i] == mask_token_id, as_tuple=False).view(-1)

        if len(positions) == 0:
            raise ValueError(f"No [MASK] token found in sample {i}.")

        loss_ids[i, positions[0]] = 1

    return loss_ids


def load_prompt_model(args, device):
    """
    Rebuild the same prompt model used in prompt_finetune.py and load the trained checkpoint.
    """
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


def make_forward_func(prompt_model, target_class):
    """
    Forward function used by Captum LayerIntegratedGradients.

    It returns the scalar logit/probability target for the requested class.
    Here target_class=1 corresponds to dementia because:
        CLASS_LABELS = ["healthy", "dementia"]
    """
    def forward_func(input_ids, attention_mask, loss_ids):
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_ids": loss_ids,
        }

        logits = prompt_model(batch)

        if isinstance(logits, dict):
            logits = logits["logits"]

        return logits[:, target_class]

    return forward_func


def run_layer_integrated_gradients(
    prompt_model,
    tokenizer,
    df,
    device,
    target_class=1,
    n_steps=32,
    max_samples=None,
):
    """
    Run Layer Integrated Gradients over the input embedding layer.

    Returns a token-level attribution DataFrame.
    """
    embedding_layer = prompt_model.plm.get_input_embeddings()
    forward_func = make_forward_func(prompt_model, target_class)

    lig = LayerIntegratedGradients(
        forward_func,
        embedding_layer
    )

    if max_samples is not None:
        df = df.head(max_samples).copy()

    rows = []

    for row_idx, row in df.iterrows():
        input_ids_list = parse_list_column(row["input_ids"])
        attention_mask_list = parse_list_column(row["attention_mask"])

        input_ids = torch.tensor(
            [input_ids_list],
            dtype=torch.long
        ).to(device)

        attention_mask = torch.tensor(
            [attention_mask_list],
            dtype=torch.long
        ).to(device)

        loss_ids = build_loss_ids(input_ids, tokenizer).to(device)

        # Baseline: PAD-token sequence.
        # This is a common practical baseline for token attribution.
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        baseline = torch.full_like(
            input_ids,
            fill_value=pad_id
        ).to(device)

        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline,
            additional_forward_args=(attention_mask, loss_ids),
            n_steps=n_steps,
            return_convergence_delta=True,
        )

        # Sum attribution over embedding dimensions to get one score per token.
        token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().tolist()

        tokens = tokenizer.convert_ids_to_tokens(
            input_ids.squeeze(0).detach().cpu().tolist()
        )

        valid_len = int(attention_mask.sum().item())
        convergence_delta = float(delta.detach().cpu().view(-1)[0])

        for pos in range(valid_len):
            rows.append({
                "id": row["id"],
                "label": row["labels"],
                "pred_label": row["pred_labels"],
                "probability_dementia": row["probas"],
                "token_position": pos,
                "token": tokens[pos],
                "attribution": token_scores[pos],
                "abs_attribution": abs(token_scores[pos]),
                "is_reportable_token": is_reportable_token(tokens[pos]),
                "convergence_delta": convergence_delta,
            })

    return pd.DataFrame(rows)


def save_top_token_summaries(attr_df, output_dir, top_k=30):
    """
    Save convenience summaries of top positive/negative attributions.

    Raw summaries include prompt tokens.
    Clean summaries include only reportable participant-text tokens.
    """
    raw_positive = (
        attr_df.sort_values("attribution", ascending=False)
        .head(top_k)
    )

    raw_negative = (
        attr_df.sort_values("attribution", ascending=True)
        .head(top_k)
    )

    clean_df = attr_df[attr_df["is_reportable_token"]].copy()

    clean_positive = (
        clean_df.sort_values("attribution", ascending=False)
        .head(top_k)
    )

    clean_negative = (
        clean_df.sort_values("attribution", ascending=True)
        .head(top_k)
    )

    raw_positive.to_csv(
        os.path.join(output_dir, "top_positive_attributions_raw.csv"),
        index=False
    )

    raw_negative.to_csv(
        os.path.join(output_dir, "top_negative_attributions_raw.csv"),
        index=False
    )

    clean_positive.to_csv(
        os.path.join(output_dir, "top_positive_attributions_clean.csv"),
        index=False
    )

    clean_negative.to_csv(
        os.path.join(output_dir, "top_negative_attributions_clean.csv"),
        index=False
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
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=30)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device(args.gpu_num)
    print(f"Using device: {device}")

    prompt_model, tokenizer = load_prompt_model(args, device)

    df = pd.read_csv(args.results_csv)

    required_cols = {
        "id",
        "labels",
        "pred_labels",
        "probas",
        "input_ids",
        "attention_mask",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {missing}")

    attr_df = run_layer_integrated_gradients(
        prompt_model=prompt_model,
        tokenizer=tokenizer,
        df=df,
        device=device,
        target_class=args.target_class,
        n_steps=args.n_steps,
        max_samples=args.max_samples,
    )

    raw_out_path = os.path.join(
        args.output_dir,
        "integrated_gradients_token_attributions.csv"
    )

    clean_out_path = os.path.join(
        args.output_dir,
        "integrated_gradients_token_attributions_clean.csv"
    )

    attr_df.to_csv(raw_out_path, index=False)

    clean_df = attr_df[attr_df["is_reportable_token"]].copy()
    clean_df.to_csv(clean_out_path, index=False)

    save_top_token_summaries(
        attr_df=attr_df,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )

    print(f"Saved raw Integrated Gradients attributions to: {raw_out_path}")
    print(f"Saved cleaned Integrated Gradients attributions to: {clean_out_path}")
    print(f"Saved top-token summaries to: {args.output_dir}")

    print("\nAttribution output summary:")
    print(f"Total attribution rows: {len(attr_df)}")
    print(f"Clean/reportable attribution rows: {len(clean_df)}")
    print(f"Unique samples: {attr_df['id'].nunique()}")

    if "convergence_delta" in attr_df.columns:
        print("\nConvergence delta summary:")
        print(attr_df.groupby("id")["convergence_delta"].first().describe())


if __name__ == "__main__":
    main()