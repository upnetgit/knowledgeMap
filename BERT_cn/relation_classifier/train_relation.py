#!/usr/bin/env python3
"""Train a multi-class relation classifier using BERT.

This script trains a BERT model for relation classification on the prepared dataset.
Supports the fixed relation set: RELATED, CONTAINS, EXTENDS, PREREQUISITE, COMPUTER_REFLECTS_IDEOLOGY.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

# Sanitize environment for threading
def _sanitize_thread_env() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            if int(value) <= 0:
                raise ValueError
        except Exception:
            os.environ[key] = "1"

_sanitize_thread_env()

# Offline mode for HuggingFace
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# ---------- Configuration ----------
ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "bert-base-chinese"
MODEL_DIR = Path(os.environ.get("BERT_MODEL_DIR", str(DEFAULT_LOCAL_MODEL_DIR))).expanduser().resolve()
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = ROOT / "bert_relation_multiclass"
SPECIAL_TOKENS = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}

RELATIONS = ["RELATED", "CONTAINS", "EXTENDS", "PREREQUISITE", "COMPUTER_REFLECTS_IDEOLOGY"]
LABEL2ID = {label: idx for idx, label in enumerate(RELATIONS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def _validate_local_model_dir(model_dir: Path) -> None:
    required_files = ("config.json", "pytorch_model.bin", "vocab.txt")
    if not model_dir.exists():
        raise FileNotFoundError(f"Local model directory not found: {model_dir}")
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Local model directory incomplete: {model_dir}; missing files: {', '.join(missing)}"
        )


_validate_local_model_dir(MODEL_DIR)
print(f"Using local BERT model: {MODEL_DIR}")


# ---------- Data Loading ----------
def load_data(file_path: str) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, int], Dict[int, str]]:
    samples = []  # (context_sentence, subject_name, object_name, relation_label)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            samples.append((
                item["context_sentence"],
                item["subject_name"],
                item["object_name"],
                item["relation_label"]
            ))
    return samples, LABEL2ID, ID2LABEL


# ---------- Input Construction (Mark Entities) ----------
def create_input(sentence: str, subj: str, obj: str) -> str:
    # Mark entities with special tokens
    sent_encoded = sentence.replace(subj, "[E1]" + subj + "[/E1]", 1)
    sent_encoded = sent_encoded.replace(obj, "[E2]" + obj + "[/E2]", 1)
    return sent_encoded


class RelationDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str, str, str]], tokenizer, label2id: Dict[str, int], max_len: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.inputs: List[Dict[str, torch.Tensor]] = []
        self.labels: List[int] = []
        for sent, subj, obj, label in samples:
            text = create_input(sent, subj, obj)
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_len,
                return_tensors='pt'
            )
            self.inputs.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            })
            self.labels.append(label2id[label])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'],
            'attention_mask': self.inputs[idx]['attention_mask'],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ---------- Model ----------
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
if tokenizer is None:
    raise RuntimeError(f"Failed to load local tokenizer: {MODEL_DIR}")
tokenizer.add_special_tokens(SPECIAL_TOKENS)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
    num_labels=len(LABEL2ID)
)
if model is None:
    raise RuntimeError(f"Failed to load local model: {MODEL_DIR}")
model.resize_token_embeddings(len(tokenizer))
model.to(DEVICE)


# ---------- Training ----------
def train_model(train_file: str, val_file: str = None) -> None:
    samples, _, _ = load_data(train_file)
    dataset = RelationDataset(samples, tokenizer, LABEL2ID, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.4f} | Macro F1: {f1:.4f}")

        if val_file:
            evaluate_model(val_file)

    # Save model and mappings
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w", encoding='utf-8') as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, ensure_ascii=False, indent=2)

    print(f"Model saved to {OUTPUT_DIR}")


def evaluate_model(val_file: str) -> None:
    samples, _, _ = load_data(val_file)
    dataset = RelationDataset(samples, tokenizer, LABEL2ID, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, target_names=RELATIONS)
    print(f"Validation | Acc: {acc:.4f} | Macro F1: {f1:.4f}")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-class relation classifier")
    parser.add_argument("--train-file", type=str, required=True, help="Training JSONL file")
    parser.add_argument("--val-file", type=str, default="", help="Validation JSONL file")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = Path(args.output_dir)
    train_model(args.train_file, args.val_file if args.val_file else None)


if __name__ == "__main__":
    main()
