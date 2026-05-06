import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "multirelation"
MODEL_DIR = Path(os.environ.get("BERT_MODEL_DIR", str(ROOT.parent / "bert-base-chinese"))).expanduser().resolve()
OUTPUT_DIR = ROOT / "bert_relation_multiclass"

MAX_LEN = 160
BATCH_SIZE = 12
EPOCHS = 4
LR = 2e-5
SPECIAL_TOKENS = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


RELATIONS = ["RELATED", "CONTAINS", "EXTENDS", "PREREQUISITE", "COMPUTER_REFLECTS_IDEOLOGY"]
LABEL2ID = {name: idx for idx, name in enumerate(RELATIONS)}
ID2LABEL = {idx: name for name, idx in LABEL2ID.items()}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
        return value if value > 0 else default
    except Exception:
        return default


MAX_LEN = _env_int("MR_MAX_LEN", MAX_LEN)
BATCH_SIZE = _env_int("MR_BATCH_SIZE", BATCH_SIZE)
EPOCHS = _env_int("MR_EPOCHS", EPOCHS)
LR = _env_float("MR_LR", LR)


def create_input(sentence: str, subject: str, obj: str) -> str:
    sentence = str(sentence or "")
    subject = str(subject or "").strip()
    obj = str(obj or "").strip()
    text = sentence.replace(subject, f"[E1]{subject}[/E1]", 1) if subject and subject in sentence else f"[E1]{subject}[/E1] {sentence}".strip()
    text = text.replace(obj, f"[E2]{obj}[/E2]", 1) if obj and obj in text else f"{text} [E2]{obj}[/E2]".strip()
    return text


def load_jsonl(path: Path) -> List[Tuple[str, str, str, int]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = str(row.get("label") or "RELATED")
            if label not in LABEL2ID:
                continue
            records.append((
                str(row.get("sentence") or ""),
                str(row.get("subject") or ""),
                str(row.get("object") or ""),
                LABEL2ID[label],
            ))
    return records


class RelationDataset(Dataset):
    def __init__(self, records: List[Tuple[str, str, str, int]], tokenizer):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sent, subj, obj, label = self.records[idx]
        text = create_input(sent, subj, obj)
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def evaluate(model, dataloader) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    micro_f1 = f1_score(all_labels, all_preds, average="micro")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return {"micro_f1": float(micro_f1), "macro_f1": float(macro_f1), "preds": all_preds, "labels": all_labels}


def _build_class_weights(records: List[Tuple[str, str, str, int]]) -> torch.Tensor:
    # 小样本长尾场景下做逆频率加权，减轻 RELATED/COMPUTER_REFLECTS_IDEOLOGY 对其余类别的覆盖。
    counts = {idx: 0 for idx in range(len(RELATIONS))}
    for _sent, _subj, _obj, label in records:
        counts[int(label)] = counts.get(int(label), 0) + 1

    total = float(sum(counts.values()) or 1.0)
    weights = []
    for idx in range(len(RELATIONS)):
        c = float(counts.get(idx, 0))
        w = total / max(c, 1.0)
        weights.append(w)

    tensor = torch.tensor(weights, dtype=torch.float32)
    tensor = tensor / tensor.mean()
    return tensor


def train() -> None:
    train_records = load_jsonl(DATA_DIR / "train.jsonl")
    val_records = load_jsonl(DATA_DIR / "val.jsonl")
    test_records = load_jsonl(DATA_DIR / "test.jsonl")
    if not train_records or not val_records or not test_records:
        raise RuntimeError("请先运行 prepare_multirelation_data.py 生成 train/val/test 数据")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    train_loader = DataLoader(RelationDataset(train_records, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RelationDataset(val_records, tokenizer), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(RelationDataset(test_records, tokenizer), batch_size=BATCH_SIZE, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        num_labels=len(RELATIONS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

    class_weights = _build_class_weights(train_records).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 20), num_training_steps=total_steps)

    best_val = -1.0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(out.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())

        val_metrics = evaluate(model, val_loader)
        print(f"Epoch {epoch + 1}: loss={total_loss / max(1, len(train_loader)):.4f}, val_micro_f1={val_metrics['micro_f1']:.4f}, val_macro_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["micro_f1"] > best_val:
            best_val = val_metrics["micro_f1"]
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            with (OUTPUT_DIR / "label_mapping.json").open("w", encoding="utf-8") as f:
                json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, ensure_ascii=False, indent=2)

    best_model = AutoModelForSequenceClassification.from_pretrained(
        str(OUTPUT_DIR),
        local_files_only=True,
        num_labels=len(RELATIONS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(DEVICE)
    test_metrics = evaluate(best_model, test_loader)
    print(f"Test micro_f1={test_metrics['micro_f1']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}")
    print(classification_report(test_metrics["labels"], test_metrics["preds"], target_names=RELATIONS, labels=list(range(len(RELATIONS))), digits=4))


if __name__ == "__main__":
    train()
