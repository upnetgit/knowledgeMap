#!/usr/bin/env python3
"""CLI wrapper for relation evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from xmodaler.evaluation.relation_eval import compute_metrics, load_jsonl, save_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate relation JSONL predictions")
    parser.add_argument("--pred", type=str, required=True, help="Prediction JSONL path")
    parser.add_argument("--gold", type=str, required=True, help="Gold JSONL path")
    parser.add_argument("--output", type=str, default="", help="Optional JSON report path")
    args = parser.parse_args()

    pred_rows = load_jsonl(Path(args.pred))
    gold_rows = load_jsonl(Path(args.gold))
    report = compute_metrics(pred_rows, gold_rows)
    if args.output:
        save_report(report, Path(args.output))
    print(report)


if __name__ == "__main__":
    main()

