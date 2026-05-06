#!/usr/bin/env python3
"""Lightweight relation evaluation utilities.

This module stays dependency-light and can be used both in offline analysis and
as a simple CLI for JSONL outputs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

RELATIONS = ["RELATED", "CONTAINS", "EXTENDS", "PREREQUISITE", "COMPUTER_REFLECTS_IDEOLOGY"]


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _label_of(row: Dict, keys: Sequence[str] = ("label", "relation", "relation_label")) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "RELATED"


def compute_metrics(pred_rows: Iterable[Dict], gold_rows: Iterable[Dict]) -> Dict[str, object]:
    pred_map = {}
    for row in pred_rows:
        key = (str(row.get("subject") or row.get("subject_name") or ""), str(row.get("object") or row.get("object_name") or ""))
        pred_map[key] = _label_of(row, ("pred_label", "label", "relation", "relation_label"))

    gold_map = {}
    for row in gold_rows:
        key = (str(row.get("subject") or row.get("subject_name") or ""), str(row.get("object") or row.get("object_name") or ""))
        gold_map[key] = _label_of(row, ("gold_label", "label", "relation", "relation_label"))

    labels = list(dict.fromkeys(RELATIONS + sorted(set(pred_map.values()) | set(gold_map.values()))))
    tp = Counter()
    fp = Counter()
    fn = Counter()

    all_keys = sorted(set(pred_map) | set(gold_map))
    for key in all_keys:
        pred = pred_map.get(key, "RELATED")
        gold = gold_map.get(key, "RELATED")
        if pred == gold:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1

    def _prf(label: str) -> Tuple[float, float, float, int]:
        p = tp[label] / max(tp[label] + fp[label], 1)
        r = tp[label] / max(tp[label] + fn[label], 1)
        f1 = 2 * p * r / max(p + r, 1e-12) if (p + r) > 0 else 0.0
        support = tp[label] + fn[label]
        return p, r, f1, support

    per_class = {label: {"precision": _prf(label)[0], "recall": _prf(label)[1], "f1": _prf(label)[2], "support": _prf(label)[3]} for label in labels}
    micro_tp = sum(tp.values())
    micro_fp = sum(fp.values())
    micro_fn = sum(fn.values())
    micro_p = micro_tp / max(micro_tp + micro_fp, 1)
    micro_r = micro_tp / max(micro_tp + micro_fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-12) if (micro_p + micro_r) > 0 else 0.0
    macro_f1 = sum(v["f1"] for v in per_class.values()) / max(len(per_class), 1)

    all_entities = {k[0] for k in all_keys if k[0]} | {k[1] for k in all_keys if k[1]}
    edge_count = len({k: v for k, v in pred_map.items() if v})
    node_count = max(len(all_entities), 1)
    density = edge_count / max(node_count * max(node_count - 1, 1), 1)
    sparsity = 1.0 - density

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"f1": macro_f1},
        "per_class": per_class,
        "support": {label: per_class[label]["support"] for label in per_class},
        "graph": {
            "node_count": node_count,
            "edge_count": edge_count,
            "density": density,
            "sparsity": sparsity,
        },
    }


def save_report(report: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate relation predictions from JSONL files")
    parser.add_argument("--pred", type=str, required=True, help="Prediction JSONL")
    parser.add_argument("--gold", type=str, required=True, help="Gold JSONL")
    parser.add_argument("--output", type=str, default="", help="Optional report JSON path")
    args = parser.parse_args()

    pred_rows = load_jsonl(Path(args.pred))
    gold_rows = load_jsonl(Path(args.gold))
    report = compute_metrics(pred_rows, gold_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        save_report(report, Path(args.output))


if __name__ == "__main__":
    main()

