#!/usr/bin/env python3
"""Lightweight relation inference helper.

This module provides a small, dependency-tolerant interface that can be used by
KG builders or ad-hoc tools without forcing a hard runtime dependency on a
trained classifier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

RELATIONS = ["RELATED", "CONTAINS", "EXTENDS", "PREREQUISITE", "COMPUTER_REFLECTS_IDEOLOGY"]

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from weak_label_rules import infer_with_confidence


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "BERT_cn" / "实体抽取" / "bert_relation_multiclass"


class RelationInference:
    """Infer relation labels from subject/object/context.

    If a trained model exists, this class may be extended later to use it.
    For now it safely falls back to the weak-label rules so the KG pipeline can
    continue to run in small-data / no-gold settings.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else _default_model_dir()
        self.available = False
        self.tokenizer = None
        self.model = None
        self._try_load_model()

    def _try_load_model(self) -> None:
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            return
        if not self.model_dir.exists():
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir), local_files_only=True)
            self.available = True
        except Exception:
            self.tokenizer = None
            self.model = None
            self.available = False

    def infer(self, subject: str, obj: str, context: str, same_class: bool = True, semantic_score: float = 0.0) -> Dict[str, object]:
        # Weak-rule fallback is intentionally the default in low-resource settings.
        result = infer_with_confidence(subject, obj, context, same_class=same_class, semantic_score=semantic_score)
        result["available_model"] = bool(self.available)
        result["model_dir"] = str(self.model_dir)
        return result

    def save_probe(self, output_path: str, subject: str, obj: str, context: str) -> None:
        payload = self.infer(subject, obj, context)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["RelationInference", "RELATIONS"]


