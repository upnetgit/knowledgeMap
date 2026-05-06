#!/usr/bin/env python3
"""Encapsulated model training and inference for relation classification.

This module provides a reusable interface for training and using the relation classifier.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RELATIONS = ["RELATED", "CONTAINS", "EXTENDS", "PREREQUISITE", "COMPUTER_REFLECTS_IDEOLOGY"]


class RelationClassifier:
    """Wrapper for BERT-based relation classification."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.tokenizer = None
        self.model = None
        self.label2id = None
        self.id2label = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        mapping_path = self.model_dir / "label_mapping.json"
        if mapping_path.exists():
            with mapping_path.open("r", encoding="utf-8") as f:
                mapping = json.load(f)
                self.label2id = mapping["label2id"]
                self.id2label = mapping["id2label"]
        else:
            # Fallback
            self.label2id = {label: idx for idx, label in enumerate(RELATIONS)}
            self.id2label = {idx: label for label, idx in self.label2id.items()}

    def predict(self, context: str, subject: str, obj: str) -> Dict[str, float]:
        """Predict relation probabilities."""
        # Mark entities
        text = context.replace(subject, "[E1]" + subject + "[/E1]", 1)
        text = text.replace(obj, "[E2]" + obj + "[/E2]", 1)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        result = {self.id2label[idx]: float(prob) for idx, prob in enumerate(probs)}
        return result

    def predict_label(self, context: str, subject: str, obj: str) -> str:
        """Predict the most likely relation label."""
        probs = self.predict(context, subject, obj)
        return max(probs, key=probs.get)


# Training utilities
def train_classifier(train_file: str, output_dir: str, val_file: Optional[str] = None) -> None:
    """Train the classifier using the provided script."""
    import subprocess
    cmd = [
        "python", str(Path(__file__).parent / "train_relation.py"),
        "--train-file", train_file,
        "--output-dir", output_dir
    ]
    if val_file:
        cmd.extend(["--val-file", val_file])
    subprocess.run(cmd, check=True)


__all__ = ["RelationClassifier", "train_classifier", "RELATIONS"]
