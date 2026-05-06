#!/usr/bin/env python3
import argparse
import re
from typing import Dict, Optional, Tuple

RELATIONS = {
    "RELATED",
    "CONTAINS",
    "EXTENDS",
    "PREREQUISITE",
    "COMPUTER_REFLECTS_IDEOLOGY",
}

_CONTAINS_TRIGGERS = ["包含", "包括", "由", "组成", "涵盖"]
_EXTENDS_TRIGGERS = ["拓展", "扩展", "进阶", "高级", "应用"]
_PREREQ_TRIGGERS = ["前置", "先学", "先", "基础", "再", "之后"]
_IDEOLOGY_HINTS = ["精神", "意识", "价值", "家国", "责任", "工匠", "规范"]


def is_evidence_valid(text: str, min_chars: int = 12) -> bool:
    cleaned = re.sub(r"\s+", "", str(text or ""))
    core = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", cleaned)
    return len(core) >= int(min_chars)


def infer_relation(subject: str, obj: str, context: str, same_class: bool = True) -> str:
    subject = str(subject or "")
    obj = str(obj or "")
    context = str(context or "")

    if not is_evidence_valid(context):
        return "RELATED"

    compact = re.sub(r"\s+", "", context)

    if same_class:
        if any(t in compact for t in _PREREQ_TRIGGERS) and (subject in compact and obj in compact):
            return "PREREQUISITE"
        if any(t in compact for t in _CONTAINS_TRIGGERS) and (subject in compact and obj in compact):
            return "CONTAINS"
        if any(t in compact for t in _EXTENDS_TRIGGERS):
            return "EXTENDS"
        return "RELATED"

    # cross-class heuristic
    if any(t in compact for t in _IDEOLOGY_HINTS):
        return "COMPUTER_REFLECTS_IDEOLOGY"
    return "RELATED"


def score_confidence(relation: str, context: str, semantic_score: float = 0.0) -> float:
    relation = str(relation or "RELATED")
    base = {
        "RELATED": 0.52,
        "CONTAINS": 0.68,
        "EXTENDS": 0.66,
        "PREREQUISITE": 0.70,
        "COMPUTER_REFLECTS_IDEOLOGY": 0.72,
    }.get(relation, 0.5)
    bonus = 0.12 if is_evidence_valid(context) else -0.10
    score = 0.6 * float(semantic_score) + 0.4 * base + bonus
    return max(0.0, min(1.0, round(score, 4)))


def infer_with_confidence(subject: str, obj: str, context: str, same_class: bool = True, semantic_score: float = 0.0) -> Dict[str, object]:
    relation = infer_relation(subject, obj, context, same_class=same_class)
    confidence = score_confidence(relation, context, semantic_score=semantic_score)
    return {
        "subject": subject,
        "object": obj,
        "relation": relation,
        "confidence": confidence,
        "evidence_valid": is_evidence_valid(context),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Weak-label relation inference rules")
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--context", type=str, required=True)
    parser.add_argument("--same-class", action="store_true", help="Whether subject/object are same class entities")
    parser.add_argument("--semantic-score", type=float, default=0.0)
    args = parser.parse_args()

    result = infer_with_confidence(
        subject=args.subject,
        obj=args.object,
        context=args.context,
        same_class=bool(args.same_class),
        semantic_score=float(args.semantic_score),
    )
    print(result)


if __name__ == "__main__":
    _cli()

