#!/usr/bin/env python3
"""Teaching-aware token helpers.

This module is intentionally lightweight and has no hard dependency on the
project's tokenizer implementation, so it can be imported safely in scripts,
evaluation, or preprocessing code.
"""

from __future__ import annotations

import re
from typing import List

_STAGE_ALIASES = {
    "大一": ["大一", "大1", "一年级", "基础", "初级", "入门", "导论", "概论", "零基础", "启蒙", "课程基础", "初学"],
    "大二": ["大二", "大2", "二年级", "进阶", "提高", "中级"],
    "大三": ["大三", "大3", "三年级", "高级", "综合", "工程化"],
    "大四": ["大四", "大4", "四年级", "毕业设计", "实习", "论文"],
    "大学": ["大学", "本科", "高等教育", "高校"],
    "高职": ["高职", "职业教育", "职教", "技能"],
    "高中": ["高中", "高考"],
    "初中": ["初中"],
    "小学": ["小学"],
}


def normalize_teaching_terms(text: str) -> str:
    """Normalize whitespace/punctuation for teaching-stage matching."""
    text = str(text or "")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[\u3000\t\r\n]+", "", text)
    return text


def detect_stage_tags(text: str) -> List[str]:
    """Detect teaching-stage keywords such as 大一/大二.

    Returns a de-duplicated list sorted by a stable priority that prefers
    narrower educational stages before broader ones.
    """
    normalized = normalize_teaching_terms(text)
    if not normalized:
        return []

    matched: List[str] = []
    for stage, aliases in _STAGE_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            matched.append(stage)

    if "大一" in matched:
        matched = ["大一"] + [item for item in matched if item != "大一"]
    elif "大二" in matched:
        matched = ["大二"] + [item for item in matched if item != "大二"]

    seen = set()
    deduped = []
    for item in matched:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


class TeachingTokenizer:
    """Compatibility helper exposing stage detection for teaching-aware flows."""

    @staticmethod
    def detect_stage_tags(text: str) -> List[str]:
        return detect_stage_tags(text)

    @staticmethod
    def normalize_teaching_terms(text: str) -> str:
        return normalize_teaching_terms(text)


__all__ = ["TeachingTokenizer", "detect_stage_tags", "normalize_teaching_terms"]

