"""
Train BERT-CLS+Prompt baseline.

This model uses the same prompt-wrapped input format as DAPF, but it does
NOT predict through the MLM/verbalizer at [MASK]. Instead, it uses a standard
sequence-classification head over [CLS].

Input:
    Participant's narration on {domain}. {transcript} Patient has diagnosis [MASK].

Prediction:
    [CLS] -> classification head -> healthy/dementia

Saved representations:
    - final [CLS]
    - final auxiliary [MASK]
    - all-layer [CLS]
    - all-layer auxiliary [MASK]

The [MASK] representation here is auxiliary. It is not the supervised
prediction site, unlike DAPF.
"""

import argparse
import ast
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)


CLASS_NAMES = ["healthy", "dementia"]
LABEL_TO_ID = {"healthy": 0, "dementia": 1}
ID_TO_LABEL = {0: "healthy", 1: "dementia"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_num: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_num}")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def normalize_label(value) -> Optional[int]:
    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        if int(value) in [0, 1]:
            return int(value)

    if isinstance(value, float):
        if int(value) in [0, 1] and float(int(value)) == float(value):
            return int(value)

    text = str(value).strip().lower()

    healthy_values = {
        "0",
        "healthy",
        "control",
        "cn",
        "normal",
        "non-ad",
        "non ad",
        "nonad",
        "no dementia",
        "nondementia",
        "non-dementia",
    }

    dementia_values = {
        "1",
        "dementia",
        "ad",
        "alzheimer",
        "alzheimers",
        "alzheimer's",
        "probablead",
        "probable ad",
        "probable_ad",
    }

    if text in healthy_values:
        return 0

    if text in dementia_values:
        return 1

    raise ValueError(f"Could not normalize label value: {value!r}")


def find_text_column(df: pd.DataFrame) -> str:
    candidates = [
        "text",
        "joined_all_par_trans",
        "transcript",
        "transcription",
        "utterance",
        "content",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find transcript column. Expected one of "
        f"{candidates}. Available columns: {df.columns.tolist()}"
    )


def find_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["filename", "id", "sample_id", "participant_id", "file"]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def load_one_csv(path: Path, label_col: str = "ad", domain_name: str = "") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_csv(path)

    if label_col not in df.columns:
        raise ValueError(
            f"Label column {label_col!r} not found in {path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    text_col = find_text_column(df)
    id_col = find_id_column(df)

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)
    out["labels"] = df[label_col].apply(normalize_label)

    if id_col is not None:
        out["id"] = df[id_col].astype(str)
    else:
        out["id"] = [f"{path.stem}_{i}" for i in range(len(df))]

    out["source_file"] = path.stem
    out["domain"] = domain_name if domain_name else path.stem

    out = out.dropna(subset=["labels"])
    out["labels"] = out["labels"].astype(int)

    out["text"] = out["text"].fillna("").astype(str)
    out = out[out["text"].str.strip().str.len() > 0].reset_index(drop=True)

    return out


def build_prompted_text(
    transcript: str,
    domain: str,
    prompt_style: str = "dapf_template7",
) -> str:
    """
    Build the prompt-as-input text.

    Main template mirrors DAPF template 7:
        Participant's narration on {domain}. {text} Patient has diagnosis [MASK].
    """
    transcript = str(transcript).strip()
    domain = str(domain).strip()

    if prompt_style == "dapf_template7":
        return f"Participant's narration on {domain}. {transcript} Patient has diagnosis [MASK]."

    if prompt_style == "no_domain":
        return f"{transcript} Patient has diagnosis [MASK]."

    if prompt_style == "neutral":
        return f"Participant's narration on {domain}. {transcript} The answer is [MASK]."

    raise ValueError(f"Unknown prompt_style: {prompt_style}")


def load_data(
    data_dir: Path,
    src_data: str,
    trg_data: str,
    trg_test_data: str,
    train_mode: str,
    label_col: str = "ad",
    prompt_style: str = "dapf_template7",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    src_df = load_one_csv(data_dir / f"{src_data}.csv", label_col=label_col, domain_name="source")
    trg_df = load_one_csv(data_dir / f"{trg_data}.csv", label_col=label_col, domain_name="target")
    test_df = load_one_csv(data_dir / f"{trg_test_data}.csv", label_col=label_col, domain_name="target")

    if train_mode == "pooled":
        train_df = pd.concat([src_df, trg_df], ignore_index=True)
    elif train_mode == "target_only":
        train_df = trg_df.copy()
    elif train_mode == "source_only":
        train_df = src_df.copy()
    else:
        raise ValueError("train_mode must be pooled, target_only, or source_only.")

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df["prompted_text"] = train_df.apply(
        lambda r: build_prompted_text(r["text"], r["domain"], prompt_style),
        axis=1,
    )
    test_df["prompted_text"] = test_df.apply(
        lambda r: build_prompted_text(r["text"], r["domain"], prompt_style),
        axis=1,
    )

    return train_df, test_df


class PromptedTranscriptDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.ids = df["id"].astype(str).tolist()
        self.texts = df["prompted_text"].astype(str).tolist()
        self.labels = df["labels"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            diagnosis_mask_position = torch.tensor(-1, dtype=torch.long)
        else:
            diagnosis_mask_position = mask_positions[0, 0].long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
            "diagnosis_mask_position": diagnosis_mask_position,
        }


def compute_class_weights(labels: List[int], device: torch.device) -> torch.Tensor:
    labels_arr = np.asarray(labels)
    n_total = len(labels_arr)
    n_classes = 2

    weights = []
    for c in range(n_classes):
        n_c = int((labels_arr == c).sum())
        if n_c == 0:
            raise ValueError(f"Class {c} has zero examples.")
        weights.append(n_total / (n_classes * n_c))

    return torch.tensor(weights, dtype=torch.float, device=device)


def compute_ece(y_true: np.ndarray, y_prob_dementia: np.ndarray, n_bins: int = 10) -> float:
    y_pred = (y_prob_dementia >= 0.5).astype(int)
    confidence = np.maximum(y_prob_dementia, 1.0 - y_prob_dementia)
    correctness = (y_pred == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        if end == 1.0:
            mask = (confidence >= start) & (confidence <= end)
        else:
            mask = (confidence >= start) & (confidence < end)

        if not np.any(mask):
            continue

        ece += mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean())

    return float(ece)


@torch.no_grad()
def evaluate(model, dataloader, df: pd.DataFrame, device: torch.device, save_hidden_states: bool = False) -> Dict:
    model.eval()

    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []
    all_indices = []
    all_input_ids = []
    all_attention_masks = []
    all_mask_positions = []

    cls_hidden_states = []
    mask_hidden_states = []
    all_layer_cls_hidden_states = []
    all_layer_mask_hidden_states = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        mask_pos = batch["diagnosis_mask_position"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=save_hidden_states,
            return_dict=True,
        )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        all_logits.append(logits.detach().cpu())
        all_probs.append(probs.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_indices.append(batch["idx"].detach().cpu())
        all_input_ids.append(batch["input_ids"].detach().cpu())
        all_attention_masks.append(batch["attention_mask"].detach().cpu())
        all_mask_positions.append(batch["diagnosis_mask_position"].detach().cpu())

        if save_hidden_states:
            hidden_states = outputs.hidden_states

            final_cls = hidden_states[-1][:, 0, :].detach().cpu()
            cls_hidden_states.append(final_cls)

            final_mask_rows = []
            layer_mask_rows = []

            layer_cls = torch.stack(
                [layer[:, 0, :] for layer in hidden_states[1:]],
                dim=1,
            ).detach().cpu()
            all_layer_cls_hidden_states.append(layer_cls)

            for b in range(input_ids.size(0)):
                pos = int(mask_pos[b].detach().cpu().item())

                if pos < 0:
                    final_mask_rows.append(torch.full_like(hidden_states[-1][b, 0, :].detach().cpu(), float("nan")))
                    layer_mask_rows.append(
                        torch.stack(
                            [
                                torch.full_like(layer[b, 0, :].detach().cpu(), float("nan"))
                                for layer in hidden_states[1:]
                            ],
                            dim=0,
                        )
                    )
                else:
                    final_mask_rows.append(hidden_states[-1][b, pos, :].detach().cpu())
                    layer_mask_rows.append(
                        torch.stack([layer[b, pos, :] for layer in hidden_states[1:]], dim=0)
                        .detach()
                        .cpu()
                    )

            mask_hidden_states.append(torch.stack(final_mask_rows, dim=0))
            all_layer_mask_hidden_states.append(torch.stack(layer_mask_rows, dim=0))

    logits_np = torch.cat(all_logits, dim=0).numpy()
    probs_np = torch.cat(all_probs, dim=0).numpy()
    preds_np = torch.cat(all_preds, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()
    indices_np = torch.cat(all_indices, dim=0).numpy()
    input_ids_np = torch.cat(all_input_ids, dim=0).numpy()
    attention_masks_np = torch.cat(all_attention_masks, dim=0).numpy()
    mask_positions_np = torch.cat(all_mask_positions, dim=0).numpy()

    order = np.argsort(indices_np)

    logits_np = logits_np[order]
    probs_np = probs_np[order]
    preds_np = preds_np[order]
    labels_np = labels_np[order]
    input_ids_np = input_ids_np[order]
    attention_masks_np = attention_masks_np[order]
    mask_positions_np = mask_positions_np[order]

    if save_hidden_states:
        cls_tensor = torch.cat(cls_hidden_states, dim=0)[order]
        mask_tensor = torch.cat(mask_hidden_states, dim=0)[order]
        layer_cls_tensor = torch.cat(all_layer_cls_hidden_states, dim=0)[order]
        layer_mask_tensor = torch.cat(all_layer_mask_hidden_states, dim=0)[order]
    else:
        cls_tensor = None
        mask_tensor = None
        layer_cls_tensor = None
        layer_mask_tensor = None

    y_prob = probs_np[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_np, preds_np)),
        "precision_macro": float(precision_score(labels_np, preds_np, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels_np, preds_np, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(labels_np, preds_np, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(labels_np, preds_np, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(labels_np, preds_np, average="weighted", zero_division=0)),
        "ece": compute_ece(labels_np, y_prob, n_bins=10),
    }

    try:
        metrics["auroc"] = float(roc_auc_score(labels_np, y_prob))
    except ValueError:
        metrics["auroc"] = float("nan")

    results_df = pd.DataFrame(
        {
            "id": df["id"].astype(str).tolist(),
            "labels": labels_np,
            "pred_labels": preds_np,
            "logits": [json.dumps(x.tolist()) for x in logits_np],
            "probas": [json.dumps(x.tolist()) for x in probs_np],
            "probability_dementia": y_prob,
            "input_ids": [json.dumps(x.tolist()) for x in input_ids_np],
            "attention_mask": [json.dumps(x.tolist()) for x in attention_masks_np],
            "diagnosis_mask_position": mask_positions_np,
            "text": df["text"].astype(str).tolist(),
            "prompted_text": df["prompted_text"].astype(str).tolist(),
            "domain": df["domain"].astype(str).tolist(),
        }
    )

    class_report = classification_report(
        labels_np,
        preds_np,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    conf_mat = confusion_matrix(labels_np, preds_np)

    return {
        "metrics": metrics,
        "results_df": results_df,
        "classification_report": class_report,
        "confusion_matrix": conf_mat,
        "cls_hidden_states": cls_tensor,
        "mask_hidden_states": mask_tensor,
        "all_layer_cls_hidden_states": layer_cls_tensor,
        "all_layer_mask_hidden_states": layer_mask_tensor,
    }


def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, grad_clip: float = 1.0) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += float(loss.detach().cpu().item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def save_outputs(output_dir, tokenizer, model, test_eval, test_df, args):
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(output_dir / "tokenizer")
    model.save_pretrained(output_dir / "hf_model")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "id_to_label": ID_TO_LABEL,
            "label_to_id": LABEL_TO_ID,
            "note": "BERT-CLS+Prompt: classification head over [CLS]; [MASK] is auxiliary input token.",
        },
        output_dir / "best-checkpoint.pt",
    )

    test_eval["results_df"].to_csv(output_dir / "test_results.csv", index=False)
    pd.DataFrame(test_eval["classification_report"]).T.to_csv(output_dir / "test_class_report.csv")
    pd.DataFrame([test_eval["metrics"]]).to_csv(output_dir / "prediction_metrics.csv", index=False)

    pd.DataFrame(
        test_eval["confusion_matrix"],
        index=["true_healthy", "true_dementia"],
        columns=["pred_healthy", "pred_dementia"],
    ).to_csv(output_dir / "confusion_matrix.csv")

    payload = {
        "ids": test_df["id"].astype(str).tolist(),
        "labels": test_df["labels"].astype(int).tolist(),
        "cls_hidden_states": test_eval["cls_hidden_states"],
        "mask_hidden_states": test_eval["mask_hidden_states"],
        "all_layer_cls_hidden_states": test_eval["all_layer_cls_hidden_states"],
        "all_layer_mask_hidden_states": test_eval["all_layer_mask_hidden_states"],
        "note": (
            "BERT-CLS+Prompt. [MASK] is present in input but is not the supervised "
            "prediction site; prediction is made by classification head over [CLS]."
        ),
    }

    torch.save(payload, output_dir / "test_results_interpretability.pt")


def main():
    parser = argparse.ArgumentParser(description="Train BERT-CLS+Prompt baseline.")

    parser.add_argument("--project_root", type=str, default="./")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--off_line_model_dir", type=str, default="./model/bert-base-uncased")
    parser.add_argument("--src_data", type=str, default="ccc_train_all")
    parser.add_argument("--trg_data", type=str, default="adress-train_all")
    parser.add_argument("--trg_test_data", type=str, default="adress-test_all")
    parser.add_argument("--train_mode", type=str, default="pooled", choices=["pooled", "target_only", "source_only"])
    parser.add_argument("--label_col", type=str, default="ad")
    parser.add_argument("--prompt_style", type=str, default="dapf_template7", choices=["dapf_template7", "no_domain", "neutral"])
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--ce_class_weights", action="store_true")
    parser.add_argument("--save_hidden_states", action="store_true")
    parser.add_argument("--save_last_instead_of_best", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.gpu_num)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BERT-CLS+Prompt training")
    print("=" * 80)
    print("Device:", device)
    print("Args:", args)

    train_df, test_df = load_data(
        data_dir=Path(args.data_dir),
        src_data=args.src_data,
        trg_data=args.trg_data,
        trg_test_data=args.trg_test_data,
        train_mode=args.train_mode,
        label_col=args.label_col,
        prompt_style=args.prompt_style,
    )

    print("\nTrain size:", len(train_df))
    print("Test size:", len(test_df))
    print("\nTrain labels:")
    print(train_df["labels"].value_counts().sort_index())
    print("\nTest labels:")
    print(test_df["labels"].value_counts().sort_index())

    train_df.to_csv(output_dir / "train_metadata.csv", index=False)
    test_df.to_csv(output_dir / "test_metadata.csv", index=False)

    if args.dev_ratio > 0:
        train_split_df, dev_df = train_test_split(
            train_df,
            test_size=args.dev_ratio,
            random_state=args.seed,
            stratify=train_df["labels"],
        )
        train_split_df = train_split_df.reset_index(drop=True)
        dev_df = dev_df.reset_index(drop=True)
    else:
        train_split_df = train_df.copy().reset_index(drop=True)
        dev_df = None

    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.off_line_model_dir)))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(Path(args.off_line_model_dir)),
        num_labels=2,
        id2label={0: "healthy", 1: "dementia"},
        label2id={"healthy": 0, "dementia": 1},
    )
    model.to(device)

    train_loader = DataLoader(
        PromptedTranscriptDataset(train_split_df, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )

    dev_loader = None
    if dev_df is not None:
        dev_loader = DataLoader(
            PromptedTranscriptDataset(dev_df, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=False,
        )

    test_loader = DataLoader(
        PromptedTranscriptDataset(test_df, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )

    if args.ce_class_weights:
        class_weights = compute_class_weights(train_split_df["labels"].astype(int).tolist(), device)
        print("\nUsing class weights:", class_weights.detach().cpu().tolist())
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, max(total_steps - 1, 0)),
    )

    best_dev_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(args.num_epochs):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print("=" * 80)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
        )

        row = {"epoch": epoch + 1, "train_loss": train_loss}
        print(f"Train loss: {train_loss:.6f}")

        if dev_loader is not None:
            dev_eval = evaluate(
                model,
                dev_loader,
                dev_df,
                device,
                save_hidden_states=False,
            )
            dev_f1 = dev_eval["metrics"]["f1_macro"]
            row.update({f"dev_{k}": v for k, v in dev_eval["metrics"].items()})

            print("Dev macro-F1:", dev_f1)

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    if not args.save_last_instead_of_best and best_state is not None:
        print(f"\nLoading best dev model with macro-F1={best_dev_f1:.6f}")
        model.load_state_dict(best_state)
    else:
        print("\nUsing last epoch model.")

    print("\nEvaluating on test...")
    test_eval = evaluate(
        model,
        test_loader,
        test_df,
        device,
        save_hidden_states=args.save_hidden_states,
    )

    print("\nTest metrics:")
    for k, v in test_eval["metrics"].items():
        print(f"  {k}: {v}")

    save_outputs(output_dir, tokenizer, model, test_eval, test_df, args)

    print("\nSaved to:", output_dir)


if __name__ == "__main__":
    main()
