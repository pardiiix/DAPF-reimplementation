import argparse
import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def find_original_prediction_mask_position(input_ids_list, tokenizer):
    mask_token_id = tokenizer.mask_token_id

    if mask_token_id is None:
        raise ValueError("Tokenizer has no mask_token_id.")

    positions = [
        i for i, token_id in enumerate(input_ids_list)
        if int(token_id) == int(mask_token_id)
    ]

    if len(positions) == 0:
        raise ValueError("No [MASK] token found in input_ids.")

    # DAPF template places diagnosis [MASK] near the end:
    # "Patient has diagnosis [MASK]."
    return int(positions[-1])


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


def get_encoder_model(prompt_model):
    """
    For BERT masked LM, prompt_model.plm is usually BertForMaskedLM,
    and the encoder is prompt_model.plm.bert.
    """
    plm = prompt_model.plm

    if hasattr(plm, "bert"):
        return plm.bert

    if hasattr(plm, "roberta"):
        return plm.roberta

    if hasattr(plm, "base_model"):
        return plm.base_model

    raise ValueError(
        "Could not identify encoder model. Expected .bert, .roberta, or .base_model."
    )


def extract_layerwise_representations(args, prompt_model, tokenizer, results_df, device):
    encoder = get_encoder_model(prompt_model)

    ids = []
    labels = []
    pred_labels = []
    probas = []

    cls_by_layer = None
    mask_by_layer = None

    all_cls = {}
    all_mask = {}

    for start in range(0, len(results_df), args.batch_size):
        batch_df = results_df.iloc[start:start + args.batch_size].copy()

        batch_ids = batch_df["id"].tolist()
        batch_labels = batch_df["labels"].astype(int).tolist()
        batch_pred_labels = batch_df["pred_labels"].astype(int).tolist()
        batch_probas = batch_df["probas"].astype(float).tolist()

        input_ids_lists = [
            parse_list_column(x) for x in batch_df["input_ids"].tolist()
        ]

        attention_mask_lists = [
            parse_list_column(x) for x in batch_df["attention_mask"].tolist()
        ]

        mask_positions = [
            find_original_prediction_mask_position(x, tokenizer)
            for x in input_ids_lists
        ]

        input_ids = torch.tensor(input_ids_lists, dtype=torch.long).to(device)
        attention_mask = torch.tensor(attention_mask_lists, dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states

        # hidden_states[0] is the embedding layer.
        # hidden_states[1] through hidden_states[12] are BERT layers 1-12.
        num_layers = len(hidden_states) - 1

        if cls_by_layer is None:
            cls_by_layer = {layer: [] for layer in range(1, num_layers + 1)}
            mask_by_layer = {layer: [] for layer in range(1, num_layers + 1)}

        batch_indices = torch.arange(input_ids.shape[0], device=device)
        mask_positions_tensor = torch.tensor(mask_positions, dtype=torch.long).to(device)

        for layer in range(1, num_layers + 1):
            layer_hidden = hidden_states[layer]

            cls_vecs = layer_hidden[:, 0, :]
            mask_vecs = layer_hidden[batch_indices, mask_positions_tensor, :]

            cls_by_layer[layer].append(cls_vecs.detach().cpu())
            mask_by_layer[layer].append(mask_vecs.detach().cpu())

        ids.extend(batch_ids)
        labels.extend(batch_labels)
        pred_labels.extend(batch_pred_labels)
        probas.extend(batch_probas)

    for layer in cls_by_layer:
        all_cls[layer] = torch.cat(cls_by_layer[layer], dim=0)
        all_mask[layer] = torch.cat(mask_by_layer[layer], dim=0)

    return {
        "ids": ids,
        "labels": torch.tensor(labels, dtype=torch.long),
        "pred_labels": torch.tensor(pred_labels, dtype=torch.long),
        "probas": torch.tensor(probas, dtype=torch.float),
        "cls_hidden_states_by_layer": all_cls,
        "mask_hidden_states_by_layer": all_mask,
    }


def run_probe(x, y, n_splits=5, random_state=42):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scores = cross_val_score(
        clf,
        x,
        y,
        cv=cv,
        scoring="f1_macro",
    )

    return scores


def run_layerwise_probes(layerwise_obj, args):
    y = layerwise_obj["labels"].detach().cpu().numpy()

    rows = []

    cls_by_layer = layerwise_obj["cls_hidden_states_by_layer"]
    mask_by_layer = layerwise_obj["mask_hidden_states_by_layer"]

    layers = sorted(cls_by_layer.keys())

    for layer in layers:
        cls_x = cls_by_layer[layer].detach().cpu().numpy()
        mask_x = mask_by_layer[layer].detach().cpu().numpy()

        cls_scores = run_probe(
            cls_x,
            y,
            n_splits=args.n_splits,
            random_state=args.cv_seed,
        )

        mask_scores = run_probe(
            mask_x,
            y,
            n_splits=args.n_splits,
            random_state=args.cv_seed,
        )

        rows.append({
            "layer": layer,
            "representation": "CLS",
            "target": "diagnosis",
            "score_mean": float(np.mean(cls_scores)),
            "score_std": float(np.std(cls_scores)),
            "scores": str([float(x) for x in cls_scores]),
        })

        rows.append({
            "layer": layer,
            "representation": "MASK",
            "target": "diagnosis",
            "score_mean": float(np.mean(mask_scores)),
            "score_std": float(np.std(mask_scores)),
            "scores": str([float(x) for x in mask_scores]),
        })

    return pd.DataFrame(rows)


def plot_layerwise_results(results_df, output_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for representation in ["CLS", "MASK"]:
        sub = results_df[results_df["representation"] == representation].copy()
        sub = sub.sort_values("layer")

        ax.plot(
            sub["layer"],
            sub["score_mean"],
            marker="o",
            label=representation,
        )

        ax.fill_between(
            sub["layer"],
            sub["score_mean"] - sub["score_std"],
            sub["score_mean"] + sub["score_std"],
            alpha=0.18,
        )

    ax.set_xlabel("BERT layer")
    ax.set_ylabel("Probe macro-F1")
    ax.set_title("Layer-wise diagnosis probing")
    ax.set_xticks(sorted(results_df["layer"].unique()))
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device(args.gpu_num)
    print(f"Using device: {device}")

    results_df = pd.read_csv(args.results_csv)

    required_cols = {"id", "labels", "pred_labels", "probas", "input_ids", "attention_mask"}
    missing_cols = required_cols - set(results_df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns in results CSV: {missing_cols}")

    prompt_model, tokenizer = load_prompt_model(args, device)

    print("Extracting layer-wise hidden states...")
    layerwise_obj = extract_layerwise_representations(
        args=args,
        prompt_model=prompt_model,
        tokenizer=tokenizer,
        results_df=results_df,
        device=device,
    )

    pt_path = os.path.join(args.output_dir, "layerwise_hidden_states.pt")
    torch.save(layerwise_obj, pt_path)

    print(f"Saved layer-wise hidden states to: {pt_path}")

    print("Running layer-wise logistic-regression probes...")
    probe_df = run_layerwise_probes(layerwise_obj, args)

    csv_path = os.path.join(args.output_dir, "layerwise_probe_results.csv")
    probe_df.to_csv(csv_path, index=False)

    plot_path = os.path.join(args.output_dir, "layerwise_probe_f1.png")
    plot_layerwise_results(probe_df, plot_path)

    print(f"Saved layer-wise probe results to: {csv_path}")
    print(f"Saved layer-wise probe plot to: {plot_path}")

    print("\nLayer-wise probing results:")
    print(
        probe_df
        .pivot(index="layer", columns="representation", values="score_mean")
        .reset_index()
        .to_string(index=False)
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--cv_seed", type=int, default=42)

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()