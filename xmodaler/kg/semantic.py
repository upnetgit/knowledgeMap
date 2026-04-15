#!/usr/bin/env python3
"""
中文语义匹配与教学摘要辅助工具。

目标：
- 优先复用仓库里的 `BERT_cn/bert-base-chinese` 做语义相似度计算；
- 如果本地模型不可用，则自动退化为轻量词面相似度，保证演示链路不断；
- 为知识点、思政元素、教学案例、媒体节点提供统一的排序接口。
"""

from __future__ import annotations

import re
import json
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for fallback mode
    torch = None

try:
    _transformers = importlib.import_module("transformers")
except Exception:  # pragma: no cover - transformers is optional for fallback mode
    _transformers = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, dict):
        parts = []
        for key in ("name", "title", "summary", "caption", "description", "keywords", "source", "label", "type"):
            if key in value:
                parts.append(_flatten_text(value[key]))
        if not parts:
            parts.extend(_flatten_text(v) for v in value.values())
        return " ".join(part for part in parts if part)
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(v) for v in value)
    return str(value)


class SemanticScorer:
    """基于本地中文BERT或词面规则的语义打分器。"""

    def __init__(self, model_dir: Optional[str] = None, max_length: int = 64, model_name: str = "bert_cn_base"):
        self.max_length = max_length
        self.model_name = str(model_name or "bert_cn_base").strip().lower()
        self.model_dir = self._resolve_model_dir(model_dir)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.available = False
        self._load_model()

    def _resolve_model_dir(self, model_dir: Optional[str]) -> Optional[Path]:
        if model_dir:
            path = Path(model_dir)
            if not path.is_absolute():
                path = _project_root() / path
            if path.exists():
                return path

        if self.model_name in {"bert_cn_finetuned", "finetuned", "bert_finetuned"}:
            finetuned_candidates = [
                _project_root() / "BERT_cn" / "实体抽取" / "bert_relation_model",
                _project_root() / "BERT_cn" / "bert_relation_model",
            ]
            for candidate in finetuned_candidates:
                if candidate.exists():
                    return candidate

        try:
            local_model_module = importlib.import_module("LOCAL_MODEL_MANAGER")
            LocalModelManager = getattr(local_model_module, "LocalModelManager", None)
            if LocalModelManager is None:
                raise ImportError("LocalModelManager not found")
            managed = LocalModelManager(project_root=str(_project_root())).get_model_path("bert")
            if managed:
                managed_path = Path(managed)
                if managed_path.exists():
                    return managed_path
        except Exception:
            pass

        candidates = [
            _project_root() / "BERT_cn" / "bert-base-chinese",
            _project_root() / "pretrained_models" / "BERT" / "bert-base-uncased",
            _project_root() / "pretrain_models" / "BERT" / "bert-base-uncased",
            _project_root() / "pretrained_models" / "BERT",
            _project_root() / "pretrain_models" / "BERT",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_model(self) -> None:
        if not self.model_dir or _transformers is None:
            return

        try:
            if torch is None:
                return
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = _transformers.AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
            self.model = _transformers.AutoModel.from_pretrained(str(self.model_dir), local_files_only=True).to(self.device)
            self.model.eval()
            self.available = True
        except Exception:
            self.tokenizer = None
            self.model = None
            self.device = None
            self.available = False

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # 中文按字/词混合处理，英文按单词处理
        tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_.-]+", text.lower())
        if not tokens:
            return []
        expanded: List[str] = []
        for token in tokens:
            if len(token) > 1 and re.fullmatch(r"[\u4e00-\u9fff]+", token):
                expanded.extend(list(token))
            else:
                expanded.append(token)
        return expanded

    def _lexical_similarity(self, text_a: str, text_b: str) -> float:
        tokens_a = set(self._tokenize(text_a))
        tokens_b = set(self._tokenize(text_b))
        if not tokens_a or not tokens_b:
            a = self._normalize(text_a)
            b = self._normalize(text_b)
            if not a or not b:
                return 0.0
            # 基于字符重叠的保底值
            overlap = len(set(a) & set(b))
            denom = max(len(set(a)) + len(set(b)), 1)
            return overlap / denom

        jaccard = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)
        coverage = len(tokens_a & tokens_b) / max(min(len(tokens_a), len(tokens_b)), 1)
        return 0.6 * jaccard + 0.4 * coverage

    def encode(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        if not self.available or self.tokenizer is None or self.model is None or torch is None:
            return None

        batch = [self._normalize(text) for text in texts]
        try:
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
                last_hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                summed = torch.sum(last_hidden * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                pooled = summed / counts
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled.detach().cpu().numpy()
        except Exception:
            return None

    def similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0

        if self.available:
            embeddings = self.encode([text_a, text_b])
            if embeddings is not None and len(embeddings) == 2:
                return float(np.clip(np.dot(embeddings[0], embeddings[1]), 0.0, 1.0))

        return float(np.clip(self._lexical_similarity(text_a, text_b), 0.0, 1.0))

    def rank_candidates(
        self,
        query_text: str,
        candidates: Union[Dict[str, Any], Sequence[Tuple[str, Any]]],
        top_k: int = 5,
        extra_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not query_text:
            query_text = extra_query or ""
        if extra_query:
            query_text = f"{query_text} {extra_query}".strip()

        if isinstance(candidates, dict):
            candidate_items = list(candidates.items())
        else:
            candidate_items = list(candidates)

        if not candidate_items:
            return []

        texts = [query_text] + [_flatten_text(meta) or name for name, meta in candidate_items]
        embeddings = self.encode(texts)

        results: List[Dict[str, Any]] = []
        if embeddings is not None:
            query_vec = embeddings[0]
            for idx, (name, meta) in enumerate(candidate_items, start=1):
                score = float(np.clip(np.dot(query_vec, embeddings[idx]), 0.0, 1.0))
                score = 0.85 * score + 0.15 * self._lexical_similarity(query_text, f"{name} {_flatten_text(meta)}")
                results.append({"name": name, "score": round(score, 4), "meta": meta})
        else:
            for name, meta in candidate_items:
                score = self._lexical_similarity(query_text, f"{name} {_flatten_text(meta)}")
                results.append({"name": name, "score": round(float(score), 4), "meta": meta})

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]


class RelationReranker:
    """计算机知识点-思政元素关系重排器（本地BERT优先，缺失时降级）。"""

    def __init__(self, model_dir: Optional[str] = None, threshold: float = 0.55):
        self.model_dir = self._resolve_model_dir(model_dir)
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.available = False
        self.tokenizer = None
        self.model = None
        self.device = None
        self.id2label: Dict[int, str] = {}
        self.positive_label_ids: List[int] = []
        self._load_model()

    def _resolve_model_dir(self, model_dir: Optional[str]) -> Optional[Path]:
        if model_dir:
            path = Path(model_dir)
            if not path.is_absolute():
                path = _project_root() / path
            if path.exists():
                return path

        candidates = [
            _project_root() / "BERT_cn" / "实体抽取" / "bert_relation_model",
            _project_root() / "BERT_cn" / "bert_relation_model",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_model(self) -> None:
        if torch is None or _transformers is None or not self.model_dir:
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = _transformers.AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
            self.model = _transformers.AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
            ).to(self.device)
            self.model.eval()

            mapping_file = self.model_dir / "label_mapping.json"
            if mapping_file.exists():
                mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
                raw_id2label = mapping.get("id2label", {})
                self.id2label = {int(k): str(v) for k, v in raw_id2label.items()}

            if not self.id2label:
                self.id2label = {idx: str(idx) for idx in range(int(self.model.config.num_labels or 2))}

            for idx, label in self.id2label.items():
                norm = str(label).strip().lower()
                if ("体现" in norm and "不" not in norm) or norm in {"reflects", "positive", "1", "yes", "true"}:
                    self.positive_label_ids.append(int(idx))
            if not self.positive_label_ids and len(self.id2label) >= 2:
                self.positive_label_ids = [max(self.id2label.keys())]

            self.available = self.tokenizer is not None and self.model is not None and bool(self.positive_label_ids)
        except Exception:
            self.available = False
            self.tokenizer = None
            self.model = None
            self.device = None
            self.id2label = {}
            self.positive_label_ids = []

    @staticmethod
    def _mark_text(sentence: str, subject: str, obj: str) -> str:
        text = str(sentence or "")
        if subject and subject in text:
            text = text.replace(subject, f"[E1]{subject}[/E1]", 1)
        if obj and obj in text:
            text = text.replace(obj, f"[E2]{obj}[/E2]", 1)
        if "[E1]" not in text or "[E2]" not in text:
            # 文本里未完整命中时，拼接成可学习格式，避免全空特征。
            text = f"{subject} 与 {obj} 的关系：{sentence}"
        return text

    def score(self, sentence: str, subject: str, obj: str) -> float:
        """返回 subject->obj 为正向关系的概率分数。"""
        sentence = str(sentence or "")
        subject = str(subject or "").strip()
        obj = str(obj or "").strip()
        if not sentence or not subject or not obj:
            return 0.0

        if not self.available:
            lexical = 0.55 if (subject in sentence and obj in sentence) else 0.0
            return float(np.clip(lexical, 0.0, 1.0))

        try:
            encoded_text = self._mark_text(sentence, subject, obj)
            encoded = self.tokenizer(
                encoded_text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().tolist()
            score = max(float(probs[idx]) for idx in self.positive_label_ids if idx < len(probs))
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    def accept(self, sentence: str, subject: str, obj: str) -> Tuple[bool, float]:
        score = self.score(sentence, subject, obj)
        return score >= self.threshold, score


def summarize_text(text: str, max_sentences: int = 2, max_chars: int = 160) -> str:
    """把长文本压缩成用于教学案例展示的摘要。"""
    if not text:
        return ""

    cleaned = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[。！？!?.])\s+|\n+", cleaned)
    snippets = [sentence.strip() for sentence in sentences if sentence.strip()]
    if snippets:
        summary = " ".join(snippets[:max_sentences])
    else:
        summary = cleaned

    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"
    return summary


__all__ = ["SemanticScorer", "RelationReranker", "summarize_text"]

