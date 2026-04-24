import json
import importlib
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========== 配置 ==========
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    semantic_module = importlib.import_module("xmodaler.kg.semantic")
    SemanticScorer = getattr(semantic_module, "SemanticScorer", None)
except Exception:
    SemanticScorer = None

MODEL_DIR = ROOT / "bert_relation_model"           # 训练好的关系分类器目录
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPECIAL_TOKENS = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}

tokenizer_obj: Optional[Any] = None
relation_model_obj: Optional[Any] = None

label2id = {}
id2label = {}


def _load_relation_assets():
    """按需加载模型与标签映射，避免导入时就失败。"""
    global tokenizer_obj, relation_model_obj, label2id, id2label
    if tokenizer_obj is not None and relation_model_obj is not None and id2label:
        return

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    if tokenizer is None:
        raise RuntimeError(f"无法加载本地 tokenizer: {MODEL_DIR}")
    tokenizer = cast(Any, tokenizer)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    relation_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
    if relation_model is None:
        raise RuntimeError(f"无法加载本地关系模型: {MODEL_DIR}")
    relation_model = cast(Any, relation_model)
    relation_model.resize_token_embeddings(len(tokenizer))
    relation_model.to(DEVICE)
    relation_model.eval()

    label_map_path = MODEL_DIR / "label_mapping.json"
    with label_map_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    tokenizer_obj = tokenizer
    relation_model_obj = relation_model
    label2id = mapping.get("label2id", {})
    id2label = {int(k): v for k, v in mapping.get("id2label", {}).items()}
    if not id2label:
        raise RuntimeError(f"标签映射为空: {label_map_path}")

# 加载预定义的思政标签列表（从原始数据中提取，或手动定义）
# 这里我们从原始数据文件中提取所有不重复的 ideology_labels
def load_ideology_labels(data_file="computer_ideology_data.txt"):
    labels = set()
    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    if not data_path.exists():
        return []

    with data_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            for lab in data.get("ideology_labels", []):
                if lab:
                    labels.add(str(lab).strip())
    return sorted(list(labels))

IDEOLOGY_LABELS = load_ideology_labels()
print(f"已加载 {len(IDEOLOGY_LABELS)} 个思政标签: {IDEOLOGY_LABELS[:5]}...")

SEMANTIC_SCORER = None
scorer_cls = SemanticScorer if callable(SemanticScorer) else None
if scorer_cls is not None:
    try:
        SEMANTIC_SCORER = scorer_cls(
            model_dir=str(ROOT.parent / "bert-base-chinese"),
            model_name="bert_cn_base",
            device_mode="cpu",
        )
    except Exception:
        SEMANTIC_SCORER = None

# ========== 辅助函数 ==========
def create_input(sentence, subj, obj):
    """与训练时一致的标记方式"""
    sentence = str(sentence or "")
    subj = str(subj or "").strip()
    obj = str(obj or "").strip()

    sent_encoded = sentence
    if subj and subj in sent_encoded:
        sent_encoded = sent_encoded.replace(subj, "[E1]" + subj + "[/E1]", 1)
    elif subj:
        sent_encoded = f"[E1]{subj}[/E1] {sent_encoded}".strip()

    if obj and obj in sent_encoded:
        sent_encoded = sent_encoded.replace(obj, "[E2]" + obj + "[/E2]", 1)
    elif obj:
        sent_encoded = f"{sent_encoded} [E2]{obj}[/E2]".strip()

    return sent_encoded


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _similarity_score(left: str, right: str) -> float:
    if SEMANTIC_SCORER is not None:
        try:
            scorer = cast(Any, SEMANTIC_SCORER)
            return float(scorer.similarity(left, right))
        except Exception:
            pass
    left_c = _compact(left)
    right_c = _compact(right)
    if not left_c or not right_c:
        return 0.0
    left_set = set(left_c)
    right_set = set(right_c)
    return float(len(left_set & right_set) / max(len(left_set | right_set), 1))

def predict_relation(sentence, subject, object_text):
    """返回 (label, confidence)，label 为 '体现' 或 '不体现'"""
    _load_relation_assets()
    tok = tokenizer_obj
    mdl = relation_model_obj
    if tok is None or mdl is None:
        raise RuntimeError("关系抽取模型未成功初始化")
    tok = cast(Any, tok)
    mdl = cast(Any, mdl)
    text = create_input(sentence, subject, object_text)
    encoding = tok(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    with torch.no_grad():
        logits = mdl(input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()
    label = id2label.get(pred_id, "不体现")
    return label, confidence

def extract_candidate_objects(sentence, subject, use_similarity=False, similarity_threshold=0.7):
    """
    从句子中提取候选客体（思政元素）
    方法1（默认）：直接匹配预定义思政标签在句子中的出现
    方法2（use_similarity=True）：使用语义相似度匹配，对句子中的每个词/短语计算与预定义标签的相似度
    """
    sentence = str(sentence or "")
    subject = str(subject or "")
    threshold = min(max(float(similarity_threshold), 0.0), 1.0)

    candidates: List[str] = []
    normalized_sentence = _compact(sentence)
    if not use_similarity:
        # 简单字符串匹配（精确匹配）
        for label in IDEOLOGY_LABELS:
            if label in sentence or _compact(label) in normalized_sentence:
                # 避免匹配到自身（如果主体也是思政标签？但主体是计算机概念，一般不会冲突）
                candidates.append(label)
        # 去重且保持顺序，保证结果稳定
        candidates = list(dict.fromkeys(candidates))
    else:
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', sentence)
        phrases = list(dict.fromkeys(words + [sentence]))
        for label in IDEOLOGY_LABELS:
            if label in candidates:
                continue
            label_score = max(_similarity_score(sentence, label), _similarity_score(subject, label))
            phrase_score = max(_similarity_score(item, label) for item in phrases) if phrases else 0.0
            if max(label_score, phrase_score) >= threshold:
                candidates.append(label)
    return candidates

def infer_objects(sentence, subject, use_similarity=False, similarity_threshold=0.7):
    """
    主函数：给定句子和主体，返回所有存在 '体现' 关系的客体及其置信度
    """
    if not str(sentence or "").strip() or not str(subject or "").strip():
        return []

    candidates = extract_candidate_objects(sentence, subject, use_similarity, similarity_threshold)
    results: List[Tuple[str, float]] = []
    for obj in candidates:
        label, conf = predict_relation(sentence, subject, obj)
        if label == "体现":
            results.append((obj, conf))
    return results

# ========== 示例使用 ==========
if __name__ == "__main__":
    # 测试句子
    test_sentence = "学习数据结构中的栈和队列，理解先进后出和先进先出的原理，培养严谨的逻辑思维和工匠精神，为开发高效软件打下基础。"
    subject = "数据结构"
    
    print(f"句子: {test_sentence}")
    print(f"主体: {subject}")
    
    # 方式1：精确匹配
    print("\n【精确匹配】推断出的客体及置信度：")
    results = infer_objects(test_sentence, subject, use_similarity=False)
    for obj, conf in results:
        print(f"  - {obj}: 置信度 {conf:.4f}")
    
    # 方式2：相似度匹配（如果句子中的词不是精确的思政标签，比如“严谨逻辑思维”可能匹配“逻辑思维”）
    test_sentence2 = "学习图结构的广度优先搜索，理解其在社交网络中的应用，培养合作精神和家国情怀（如分析疫情传播）。"
    subject2 = "数据结构"
    print(f"\n句子2: {test_sentence2}")
    print(f"主体: {subject2}")
    print("【相似度匹配】推断出的客体及置信度：")
    results2 = infer_objects(test_sentence2, subject2, use_similarity=True, similarity_threshold=0.6)
    for obj, conf in results2:
        print(f"  - {obj}: 置信度 {conf:.4f}")
    
    # 也可以测试一个不包含任何思政标签的句子
    test_sentence3 = "今天天气真好。"
    subject3 = "算法"
    print(f"\n句子3: {test_sentence3}")
    print(f"主体: {subject3}")
    results3 = infer_objects(test_sentence3, subject3)
    print(f"推断结果: {results3} (应为空列表)")
