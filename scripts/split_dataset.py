#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_stratified(rows: List[Dict], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    random.seed(seed)
    by_label: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("relation_label") or row.get("label") or "RELATED")].append(row)

    train, val, test = [], [], []
    for label_rows in by_label.values():
        random.shuffle(label_rows)
        n = len(label_rows)
        tr_end = int(n * train_ratio)
        va_end = int(n * (train_ratio + val_ratio))
        train.extend(label_rows[:tr_end])
        val.extend(label_rows[tr_end:va_end])
        test.extend(label_rows[va_end:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Split relation dataset into train/val/test")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", type=str, default="stratified", choices=["stratified", "simple"], help="Split strategy")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    if not rows:
        raise RuntimeError("Input dataset is empty")

    if args.strategy == "simple":
        random.seed(int(args.seed))
        random.shuffle(rows)
        n = len(rows)
        tr_end = int(n * float(args.train_ratio))
        va_end = int(n * (float(args.train_ratio) + float(args.val_ratio)))
        train, val, test = rows[:tr_end], rows[tr_end:va_end], rows[va_end:]
    else:
        train, val, test = split_stratified(rows, train_ratio=float(args.train_ratio), val_ratio=float(args.val_ratio), seed=int(args.seed))
    out_dir = Path(args.output_dir)
    save_jsonl(out_dir / "train.jsonl", train)
    save_jsonl(out_dir / "val.jsonl", val)
    save_jsonl(out_dir / "test.jsonl", test)

    print(f"Saved train={len(train)}, val={len(val)}, test={len(test)} to {out_dir}")


if __name__ == "__main__":
    main()

