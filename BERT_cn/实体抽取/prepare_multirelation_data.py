import json
import random
import re
import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

random.seed(42)

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "computer_ideology_data.txt"
OUT_DIR = ROOT / "multirelation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RELATIONS = ["RELATED", "CONTAINS", "EXTENDS", "PREREQUISITE", "COMPUTER_REFLECTS_IDEOLOGY"]


def _load_datamain_entities() -> Tuple[List[str], List[str]]:
    datamain = ROOT.parent / "datamain.txt"
    computer_whitelist = ROOT.parent.parent / "data" / "computer_terms_whitelist.txt"
    ideology_whitelist = ROOT.parent.parent / "data" / "ideology_terms_whitelist.txt"

    computer = []
    ideology = []

    # 从datamain.txt加载
    if datamain.exists():
        for raw_line in datamain.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            if key == "COMPUTER_LABELS":
                computer.extend([str(item).strip() for item in ast.literal_eval(value) if str(item).strip()])
            elif key == "IDEOLOGY_LABELS":
                ideology.extend([str(item).strip() for item in ast.literal_eval(value) if str(item).strip()])

    # 从whitelist文件加载并合并
    if computer_whitelist.exists():
        for line in computer_whitelist.read_text(encoding="utf-8").splitlines():
            term = line.strip()
            if term and term not in computer:
                computer.append(term)

    if ideology_whitelist.exists():
        for line in ideology_whitelist.read_text(encoding="utf-8").splitlines():
            term = line.strip()
            if term and term not in ideology:
                ideology.append(term)

    return list(set(computer)), list(set(ideology))


def _safe_relation_from_pair(left: str, right: str) -> str:
    compact_l = re.sub(r"\s+", "", left)
    compact_r = re.sub(r"\s+", "", right)
    basic_tokens = ["基础", "入门", "初级", "导论", "概论", "原理"]
    advanced_tokens = ["进阶", "高级", "实践", "应用", "工程", "优化", "综合"]

    if compact_l and compact_l in compact_r and len(compact_l) < len(compact_r):
        return "EXTENDS"
    if compact_r and compact_r in compact_l and len(compact_r) < len(compact_l):
        return "CONTAINS"
    if any(token in left for token in basic_tokens) and any(token in right for token in advanced_tokens):
        return "PREREQUISITE"
    return "RELATED"


def _split_dataset(records: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1):
    random.shuffle(records)
    n = len(records)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return records[:train_end], records[train_end:val_end], records[val_end:]


def build_dataset() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"raw file not found: {RAW_FILE}")

    computer_entities, ideology_entities = _load_datamain_entities()
    ideology_set = set(ideology_entities)

    records: List[Dict] = []
    computer_to_ideologies: Dict[str, set] = defaultdict(set)

    # 1) 跨类正样本：COMPUTER_REFLECTS_IDEOLOGY
    with RAW_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = str(item.get("text") or "")
            computer = str(item.get("computer_label") or "").strip()
            ideologies = [str(name).strip() for name in item.get("ideology_labels") or [] if str(name).strip()]
            if not computer or not ideologies:
                continue
            for ideology in ideologies:
                records.append(
                    {
                        "sentence": text,
                        "subject": computer,
                        "object": ideology,
                        "label": "COMPUTER_REFLECTS_IDEOLOGY",
                    }
                )
                computer_to_ideologies[computer].add(ideology)

    # 2) 同类样本：弱监督构造 RELATED/CONTAINS/EXTENDS/PREREQUISITE
    for i, left in enumerate(computer_entities):
        for j, right in enumerate(computer_entities):
            if i == j:
                continue
            relation = _safe_relation_from_pair(left, right)
            sentence = f"教学知识图谱中，{left} 与 {right} 存在{relation}关系。"
            records.append({"sentence": sentence, "subject": left, "object": right, "label": relation})
            # 增加变体句子以扩大量
            if random.random() < 0.5:
                sentence2 = f"在计算机科学中，{left} 和 {right} 有{relation}联系。"
                records.append({"sentence": sentence2, "subject": left, "object": right, "label": relation})

    # 2.1) 小数据集补齐：显式构造三类关系样本，避免训练时类别缺失。
    for base in computer_entities[: min(20, len(computer_entities))]:  # 增加到20
        intro = f"{base}基础"
        practice = f"{base}实践"
        advanced = f"{base}高级"
        records.append({
            "sentence": f"{intro}是学习{base}实践的前提知识。",
            "subject": intro,
            "object": practice,
            "label": "PREREQUISITE",
        })
        records.append({
            "sentence": f"{base}课程包含{intro}等核心内容。",
            "subject": base,
            "object": intro,
            "label": "CONTAINS",
        })
        records.append({
            "sentence": f"{practice}是{base}在工程场景中的扩展应用。",
            "subject": practice,
            "object": base,
            "label": "EXTENDS",
        })
        records.append({
            "sentence": f"{advanced}是{base}的进阶版本。",
            "subject": advanced,
            "object": base,
            "label": "EXTENDS",
        })
        # 增加干扰数据：随机RELATED
        if random.random() < 0.3:
            random_other = random.choice(computer_entities)
            if random_other != base:
                sentence_noise = f"{base} 和 {random_other} 在某些方面相关。"
                records.append({"sentence": sentence_noise, "subject": base, "object": random_other, "label": "RELATED"})

    # 2.2) 额外生成PREREQUISITE样本，确保类别平衡
    prerequisite_pairs = [
        ("数据结构", "算法"),
        ("计算机组成原理", "操作系统"),
        ("算法", "计算机网络"),
        ("数据库", "软件工程"),
        ("编译原理", "人工智能"),
        ("机器学习", "深度学习"),
        ("计算机视觉", "自然语言处理"),
        ("分布式系统", "云计算"),
        ("大数据技术", "信息安全"),
        ("操作系统", "计算机网络"),
        ("软件工程", "编译原理"),
        ("人工智能", "机器学习"),
    ]
    for subj, obj in prerequisite_pairs:
        if subj in computer_entities and obj in computer_entities:
            sentence = f"学习{obj}之前需要掌握{subj}的基础知识。"
            records.append({"sentence": sentence, "subject": subj, "object": obj, "label": "PREREQUISITE"})
            # 变体
            sentence2 = f"要学{obj}，先学{subj}。"
            records.append({"sentence": sentence2, "subject": subj, "object": obj, "label": "PREREQUISITE"})

    # 3) 跨类负采样：用 RELATED 作为弱负类，帮助模型区分“跨类非直接映射”
    for comp in computer_entities:
        positives = computer_to_ideologies.get(comp, set())
        negatives = [name for name in ideology_set if name not in positives]
        random.shuffle(negatives)
        for ideology in negatives[:3]:  # 增加到3
            sentence = f"课程场景提到{comp}，但未直接体现{ideology}。"
            records.append({"sentence": sentence, "subject": comp, "object": ideology, "label": "RELATED"})
            # 变体
            sentence2 = f"{comp} 与 {ideology} 无直接关联。"
            records.append({"sentence": sentence2, "subject": comp, "object": ideology, "label": "RELATED"})

    # 4) 保存全集和拆分
    all_path = OUT_DIR / "all.jsonl"
    with all_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    train, val, test = _split_dataset(records)
    for name, split in [("train.jsonl", train), ("val.jsonl", val), ("test.jsonl", test)]:
        path = OUT_DIR / name
        with path.open("w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stat = defaultdict(int)
    for row in records:
        stat[row["label"]] += 1
    print(f"Total={len(records)}, train={len(train)}, val={len(val)}, test={len(test)}")
    print("Label dist:", dict(sorted(stat.items())))


if __name__ == "__main__":
    build_dataset()
