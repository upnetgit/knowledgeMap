#!/usr/bin/env python3
"""Prepare weakly supervised relation dataset.

This script keeps a minimal, dependency-light workflow:
1. Read known computer / ideology entities from `BERT_cn/datamain.txt`.
2. Optionally scan a JSONL of manual annotations.
3. Emit a weak dataset in JSONL format compatible with `train_multirelation.py`.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
BERT_ROOT = ROOT / "BERT_cn"

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from weak_label_rules import infer_with_confidence, is_evidence_valid


def load_datamain_entities(datamain_path: Path) -> Tuple[List[str], List[str]]:
    if not datamain_path.exists():
        return [], []

    computer_entities: List[str] = []
    ideology_entities: List[str] = []
    for raw_line in datamain_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            continue
        if not isinstance(parsed, (list, tuple)):
            continue
        items = [str(item).strip() for item in parsed if str(item).strip()]
        if key == "COMPUTER_LABELS":
            computer_entities = items
        elif key == "IDEOLOGY_LABELS":
            ideology_entities = items
    return computer_entities, ideology_entities


def _iter_text_files(text_dir: Path) -> Iterable[Path]:
    if not text_dir.exists():
        return []
    return sorted(path for path in text_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".txt")


def _load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？!?；;\n\r])", str(text or ""))
    return [part.strip() for part in parts if part and part.strip()]


def _make_record(idx: int, subject: str, obj: str, context: str, label: str, confidence: float, source: str) -> Dict[str, object]:
    return {
        "id": f"sample_{idx}",
        "subject_name": subject,
        "object_name": obj,
        "context_sentence": context,
        "relation_label": label,
        "annotator_id": source,
        "confidence": round(float(confidence), 4),
    }


def build_dataset(data_dir: Path, output_dir: Path, seed: int = 42, limit: int = 0) -> Dict[str, int]:
    random.seed(seed)
    datamain = BERT_ROOT / "datamain.txt"
    computer_entities, ideology_entities = load_datamain_entities(datamain)
    ideology_set = set(ideology_entities)

    records: List[Dict[str, object]] = []
    idx = 0

    # 1) manual annotations if present
    annotation_paths = [data_dir / "annotations", ROOT / "kg_output" / "manual_video_annotations.jsonl"]
    for ann_path in annotation_paths:
        if ann_path.is_dir():
            jsonl_files = sorted(ann_path.rglob("*.jsonl"))
        else:
            jsonl_files = [ann_path] if ann_path.exists() else []
        for path in jsonl_files:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                subject = str(item.get("computer_entity") or item.get("subject_name") or item.get("subject") or "").strip()
                obj = str(item.get("ideology_entity") or item.get("object_name") or item.get("object") or "").strip()
                context = str(item.get("caption") or item.get("ocr_text") or item.get("summary") or "").strip()
                if not subject or not obj or not context:
                    continue
                label = str(item.get("relation_label") or item.get("relation") or "COMPUTER_REFLECTS_IDEOLOGY").strip()
                if label not in {"RELATED", "CONTAINS", "EXTENDS", "PREREQUISITE", "COMPUTER_REFLECTS_IDEOLOGY"}:
                    label = "COMPUTER_REFLECTS_IDEOLOGY"
                confidence = float(item.get("confidence") or 0.75)
                records.append(_make_record(idx, subject, obj, context, label, confidence, "manual"))
                idx += 1

    # 2) weak labels from text corpus
    text_dir = data_dir / "txt"
    for text_file in _iter_text_files(text_dir):
        text = _load_text(text_file)
        sentences = _split_sentences(text)
        if not sentences:
            continue

        for sent in sentences:
            if limit > 0 and len(records) >= limit:
                break
            if not is_evidence_valid(sent):
                continue

            lower = sent.lower()
            comp_hits = [term for term in computer_entities if term and term in sent]
            ideo_hits = [term for term in ideology_entities if term and term in sent]

            # cross-class samples
            for comp in comp_hits[:2]:
                for ideo in ideo_hits[:2]:
                    result = infer_with_confidence(comp, ideo, sent, same_class=False, semantic_score=0.5)
                    if result["relation"] != "RELATED":
                        records.append(_make_record(idx, comp, ideo, sent, result["relation"], result["confidence"], "weak_rule"))
                        idx += 1

            # same-class samples (sparse)
            for a, b in zip(comp_hits, comp_hits[1:]):
                result = infer_with_confidence(a, b, sent, same_class=True, semantic_score=0.4)
                if result["relation"] in {"RELATED", "CONTAINS", "EXTENDS", "PREREQUISITE"}:
                    records.append(_make_record(idx, a, b, sent, result["relation"], result["confidence"], "weak_rule"))
                    idx += 1

            if any(token in lower for token in ["包含", "包括", "由", "组成"]):
                for comp in comp_hits[:1]:
                    for other in comp_hits[1:2]:
                        records.append(_make_record(idx, comp, other, sent, "CONTAINS", 0.7, "pattern"))
                        idx += 1
            if any(token in lower for token in ["先", "前置", "基础", "之后"]):
                for comp in comp_hits[:1]:
                    for other in comp_hits[1:2]:
                        records.append(_make_record(idx, comp, other, sent, "PREREQUISITE", 0.72, "pattern"))
                        idx += 1
            if any(token in sent for token in ["扩展", "拓展", "进阶", "高级"]):
                for comp in comp_hits[:1]:
                    for other in comp_hits[1:2]:
                        records.append(_make_record(idx, comp, other, sent, "EXTENDS", 0.68, "pattern"))
                        idx += 1

    # 3) fallback synthetic samples if the corpus is too small
    if not records and computer_entities and ideology_entities:
        for comp in computer_entities[:10]:
            for ideo in ideology_entities[:5]:
                context = f"{comp} 的学习体现 {ideo}。"
                records.append(_make_record(idx, comp, ideo, context, "COMPUTER_REFLECTS_IDEOLOGY", 0.72, "synthetic"))
                idx += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "relation_pairs.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # also save a weak copy for debugging/inspection
    weak_path = output_dir / "relation_pairs_weak.jsonl"
    with weak_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stat = defaultdict(int)
    for row in records:
        stat[str(row["relation_label"])] += 1
    print(json.dumps({"output": str(output_path), "count": len(records), "dist": dict(stat)}, ensure_ascii=False, indent=2))
    return dict(stat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare weak relation dataset")
    parser.add_argument("--data-dir", type=str, default=str(DATA_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "datasets" / "relation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    build_dataset(Path(args.data_dir), Path(args.output_dir), seed=int(args.seed), limit=int(args.limit))


if __name__ == "__main__":
    main()


