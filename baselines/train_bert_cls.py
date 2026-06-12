"""
Train a standard BERT [CLS] sequence-classification baseline on the same
limited data regime used by DAPF.

Main intended comparison:

    DAPF-BERT:
        CCC + ADReSS train -> ADReSS test
        prompt-based masked-token prediction

    BERT-CLS baseline:
        CCC + ADReSS train -> ADReSS test
        standard sequence classification head over [CLS]

This script saves outputs in a format that is intentionally similar to the
DAPF interpretability outputs:

    output_dir/
    ├── best-checkpoint.pt
    ├── tokenizer/
    ├── test_results.csv
    ├── test_class_report.csv
    ├── prediction_metrics.csv
    └── test_results_interpretability.pt

The saved .pt file contains final-layer [CLS] states and all-layer [CLS]
states so that the same representation-probing logic can be reused later.
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
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism can slow training but is useful for small-data experiments.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_num: int = 0) -> torch.device:
    """Use CUDA if available, then MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_num}")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def normalize_label(value) -> Optional[int]:
    """
    Normalize labels to:
        0 = healthy
        1 = dementia

    Handles common label formats used in the DAPF data files.
    """
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
    """Find the transcript/text column in a DAPF-style CSV."""
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
        "Could not find a transcript column. Expected one of: "
        f"{candidates}. Available columns: {df.columns.tolist()}"
    )


def find_id_column(df: pd.DataFrame) -> Optional[str]:
    """Find a stable sample identifier column if available."""
    candidates = ["filename", "id", "sample_id", "participant_id", "file"]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def load_one_csv(path: Path, label_col: str = "ad", domain_name: str = "") -> pd.DataFrame:
    """Load one DAPF-style CSV and standardize to id/text/label/domain."""
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

    # Remove empty transcripts.
    out["text"] = out["text"].fillna("").astype(str)
    out = out[out["text"].str.strip().str.len() > 0].reset_index(drop=True)

    return out


def load_data(
    data_dir: Path,
    src_data: str,
    trg_data: str,
    trg_test_data: str,
    train_mode: str,
    label_col: str = "ad",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train/test data.

    train_mode:
        pooled      = source + target train
        target_only = target train only
        source_only = source only
    """
    src_path = data_dir / f"{src_data}.csv"
    trg_path = data_dir / f"{trg_data}.csv"
    test_path = data_dir / f"{trg_test_data}.csv"

    src_df = load_one_csv(src_path, label_col=label_col, domain_name="source")
    trg_df = load_one_csv(trg_path, label_col=label_col, domain_name="target_train")
    test_df = load_one_csv(test_path, label_col=label_col, domain_name="target_test")

    if train_mode == "pooled":
        train_df = pd.concat([src_df, trg_df], ignore_index=True)
    elif train_mode == "target_only":
        train_df = trg_df.copy()
    elif train_mode == "source_only":
        train_df = src_df.copy()
    else:
        raise ValueError(
            f"Unknown train_mode={train_mode!r}. "
            "Choose from: pooled, target_only, source_only."
        )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


class TranscriptDataset(Dataset):
    """Simple transcript classification dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
    ):
        self.ids = df["id"].astype(str).tolist()
        self.texts = df["text"].astype(str).tolist()
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

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }

        return item


def compute_class_weights(labels: List[int], device: torch.device) -> torch.Tensor:
    """
    Compute balanced class weights for CrossEntropyLoss.

    weight_c = n_samples / (n_classes * n_c)
    """
    labels_arr = np.asarray(labels)
    n_total = len(labels_arr)
    n_classes = 2

    weights = []
    for c in range(n_classes):
        n_c = int((labels_arr == c).sum())
        if n_c == 0:
            raise ValueError(f"Class {c} has zero examples in training data.")
        weights.append(n_total / (n_classes * n_c))

    return torch.tensor(weights, dtype=torch.float, device=device)


def parse_array_string(value):
    """Safely parse list-like strings if needed by downstream scripts."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def compute_ece(
    y_true: np.ndarray,
    y_prob_dementia: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error for binary classification."""
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

        bin_acc = correctness[mask].mean()
        bin_conf = confidence[mask].mean()
        bin_weight = mask.mean()
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    df: pd.DataFrame,
    device: torch.device,
    save_hidden_states: bool = False,
) -> Dict:
    """Evaluate model and optionally collect hidden states."""
    model.eval()

    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []
    all_indices = []
    all_input_ids = []
    all_attention_masks = []

    cls_hidden_states = []
    all_layer_cls_hidden_states = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
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

        if save_hidden_states:
            # outputs.hidden_states is tuple:
            # embeddings + layer 1 ... layer 12.
            hidden_states = outputs.hidden_states

            # Final-layer [CLS].
            final_cls = hidden_states[-1][:, 0, :].detach().cpu()
            cls_hidden_states.append(final_cls)

            # All transformer-layer [CLS] states, excluding embedding layer.
            # Shape per batch: [batch, num_layers, hidden_dim].
            layer_cls = torch.stack(
                [layer[:, 0, :] for layer in hidden_states[1:]],
                dim=1,
            ).detach().cpu()
            all_layer_cls_hidden_states.append(layer_cls)

    logits_np = torch.cat(all_logits, dim=0).numpy()
    probs_np = torch.cat(all_probs, dim=0).numpy()
    preds_np = torch.cat(all_preds, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()
    indices_np = torch.cat(all_indices, dim=0).numpy()
    input_ids_np = torch.cat(all_input_ids, dim=0).numpy()
    attention_masks_np = torch.cat(all_attention_masks, dim=0).numpy()

    # Restore original dataloader order if needed.
    order = np.argsort(indices_np)
    logits_np = logits_np[order]
    probs_np = probs_np[order]
    preds_np = preds_np[order]
    labels_np = labels_np[order]
    input_ids_np = input_ids_np[order]
    attention_masks_np = attention_masks_np[order]

    if save_hidden_states:
        cls_tensor = torch.cat(cls_hidden_states, dim=0)[order]
        layer_cls_tensor = torch.cat(all_layer_cls_hidden_states, dim=0)[order]
    else:
        cls_tensor = None
        layer_cls_tensor = None

    y_prob_dementia = probs_np[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_np, preds_np)),
        "precision_macro": float(
            precision_score(labels_np, preds_np, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels_np, preds_np, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(labels_np, preds_np, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(labels_np, preds_np, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(labels_np, preds_np, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(labels_np, preds_np, average="weighted", zero_division=0)
        ),
        "ece": compute_ece(labels_np, y_prob_dementia, n_bins=10),
    }

    try:
        metrics["auroc"] = float(roc_auc_score(labels_np, y_prob_dementia))
    except ValueError:
        metrics["auroc"] = float("nan")

    results_df = pd.DataFrame(
        {
            "id": df["id"].astype(str).tolist(),
            "labels": labels_np,
            "pred_labels": preds_np,
            "logits": [json.dumps(x.tolist()) for x in logits_np],
            "probas": [json.dumps(x.tolist()) for x in probs_np],
            "probability_dementia": y_prob_dementia,
            "input_ids": [json.dumps(x.tolist()) for x in input_ids_np],
            "attention_mask": [json.dumps(x.tolist()) for x in attention_masks_np],
            "text": df["text"].astype(str).tolist(),
            "domain": df["domain"].astype(str).tolist()
            if "domain" in df.columns
            else [""] * len(df),
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
        "all_layer_cls_hidden_states": layer_cls_tensor,
    }


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    loss_fn,
    device: torch.device,
    grad_clip: Optional[float] = 1.0,
) -> float:
    """Train for one epoch."""
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
            labels=None,
            return_dict=True,
        )

        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += float(loss.detach().cpu().item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def save_outputs(
    output_dir: Path,
    tokenizer,
    model,
    test_eval: Dict,
    test_df: pd.DataFrame,
    args,
) -> None:
    """Save final test outputs and interpretability artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    # Save model checkpoint in both HF format and plain .pt state dict.
    model.save_pretrained(output_dir / "hf_model")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "id_to_label": ID_TO_LABEL,
            "label_to_id": LABEL_TO_ID,
        },
        output_dir / "best-checkpoint.pt",
    )

    test_eval["results_df"].to_csv(output_dir / "test_results.csv", index=False)

    pd.DataFrame(test_eval["classification_report"]).T.to_csv(
        output_dir / "test_class_report.csv"
    )

    pd.DataFrame([test_eval["metrics"]]).to_csv(
        output_dir / "prediction_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        test_eval["confusion_matrix"],
        index=["true_healthy", "true_dementia"],
        columns=["pred_healthy", "pred_dementia"],
    ).to_csv(output_dir / "confusion_matrix.csv")

    interpretability_payload = {
        "ids": test_df["id"].astype(str).tolist(),
        "labels": test_df["labels"].astype(int).tolist(),
        "cls_hidden_states": test_eval["cls_hidden_states"],
        "all_layer_cls_hidden_states": test_eval["all_layer_cls_hidden_states"],
        "note": (
            "Standard BERT-CLS baseline. No [MASK] diagnosis representation exists. "
            "Use cls_hidden_states and all_layer_cls_hidden_states for probing."
        ),
    }

    torch.save(
        interpretability_payload,
        output_dir / "test_results_interpretability.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train standard BERT [CLS] classifier baseline for dementia detection."
    )

    parser.add_argument("--project_root", type=str, default="./")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--off_line_model_dir", type=str, default="./model/bert-base-uncased")
    parser.add_argument("--src_data", type=str, default="ccc_train_all")
    parser.add_argument("--trg_data", type=str, default="adress-train_all")
    parser.add_argument("--trg_test_data", type=str, default="adress-test_all")
    parser.add_argument(
        "--train_mode",
        type=str,
        default="pooled",
        choices=["pooled", "target_only", "source_only"],
        help=(
            "pooled = source + target train; target_only = target train only; "
            "source_only = source only."
        ),
    )
    parser.add_argument("--label_col", type=str, default="ad")
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
    parser.add_argument(
        "--ce_class_weights",
        action="store_true",
        help="Use balanced class weights in CrossEntropyLoss.",
    )
    parser.add_argument(
        "--save_hidden_states",
        action="store_true",
        help="Save final-layer and all-layer [CLS] hidden states on test set.",
    )
    parser.add_argument(
        "--save_last_instead_of_best",
        action="store_true",
        help="If set, ignore dev performance and save the last epoch.",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.gpu_num)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BERT-CLS baseline training")
    print("=" * 80)
    print("Device:", device)
    print("Args:", args)

    data_dir = Path(args.data_dir)

    train_df, test_df = load_data(
        data_dir=data_dir,
        src_data=args.src_data,
        trg_data=args.trg_data,
        trg_test_data=args.trg_test_data,
        train_mode=args.train_mode,
        label_col=args.label_col,
    )

    print("\nLoaded data")
    print("Train size:", len(train_df))
    print("Test size:", len(test_df))
    print("\nTrain label counts:")
    print(train_df["labels"].value_counts().sort_index())
    print("\nTest label counts:")
    print(test_df["labels"].value_counts().sort_index())

    # Save the exact train/test metadata for reproducibility.
    train_df.to_csv(output_dir / "train_metadata.csv", index=False)
    test_df.to_csv(output_dir / "test_metadata.csv", index=False)

    # Optional dev split for checkpoint selection.
    if args.dev_ratio > 0:
        train_split_df, dev_df = train_test_split(
            train_df,
            test_size=args.dev_ratio,
            random_state=args.seed,
            stratify=train_df["labels"],
        )
        train_split_df = train_split_df.reset_index(drop=True)
        dev_df = dev_df.reset_index(drop=True)
        print("\nUsing dev split for checkpoint selection")
        print("Train split size:", len(train_split_df))
        print("Dev size:", len(dev_df))
    else:
        train_split_df = train_df.copy().reset_index(drop=True)
        dev_df = None
        print("\nNo dev split. Will save last epoch unless test evaluation is manually inspected.")

    model_path = Path(args.off_line_model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=2,
        id2label={0: "healthy", 1: "dementia"},
        label2id={"healthy": 0, "dementia": 1},
    )

    model.to(device)

    train_dataset = TranscriptDataset(
        train_split_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    if dev_df is not None:
        dev_dataset = TranscriptDataset(
            dev_df,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
    else:
        dev_loader = None

    test_dataset = TranscriptDataset(
        test_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    if args.ce_class_weights:
        class_weights = compute_class_weights(
            train_split_df["labels"].astype(int).tolist(),
            device=device,
        )
        print("\nUsing CE class weights:", class_weights.detach().cpu().tolist())
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("\nUsing unweighted CE loss")
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
    training_history = []

    for epoch in range(args.num_epochs):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print("=" * 80)

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
        )

        row = {"epoch": epoch + 1, "train_loss": train_loss}
        print(f"Train loss: {train_loss:.6f}")

        if dev_loader is not None:
            dev_eval = evaluate(
                model=model,
                dataloader=dev_loader,
                df=dev_df,
                device=device,
                save_hidden_states=False,
            )
            dev_f1 = dev_eval["metrics"]["f1_macro"]
            row.update({f"dev_{k}": v for k, v in dev_eval["metrics"].items()})

            print("Dev metrics:")
            for k, v in dev_eval["metrics"].items():
                print(f"  {k}: {v}")

            if dev_f1 > best_dev_f1:
                print(f"New best dev macro-F1: {dev_f1:.6f}")
                best_dev_f1 = dev_f1
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        training_history.append(row)
        pd.DataFrame(training_history).to_csv(
            output_dir / "training_history.csv",
            index=False,
        )

    if args.save_last_instead_of_best or best_state is None:
        print("\nSaving/evaluating last epoch model.")
    else:
        print(f"\nLoading best dev model with macro-F1={best_dev_f1:.6f}")
        model.load_state_dict(best_state)

    print("\nRunning final evaluation on ADReSS test set...")
    test_eval = evaluate(
        model=model,
        dataloader=test_loader,
        df=test_df,
        device=device,
        save_hidden_states=args.save_hidden_states,
    )

    print("\nTest metrics:")
    for k, v in test_eval["metrics"].items():
        print(f"  {k}: {v}")

    print("\nTest confusion matrix:")
    print(test_eval["confusion_matrix"])

    save_outputs(
        output_dir=output_dir,
        tokenizer=tokenizer,
        model=model,
        test_eval=test_eval,
        test_df=test_df,
        args=args,
    )

    print("\nSaved outputs to:", output_dir)
    print("Important files:")
    print(" ", output_dir / "best-checkpoint.pt")
    print(" ", output_dir / "hf_model")
    print(" ", output_dir / "tokenizer")
    print(" ", output_dir / "test_results.csv")
    print(" ", output_dir / "test_class_report.csv")
    print(" ", output_dir / "prediction_metrics.csv")
    print(" ", output_dir / "test_results_interpretability.pt")


if __name__ == "__main__":
    main()
