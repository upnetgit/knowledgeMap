import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util  # 用于相似度计算
import re

# ========== 配置 ==========
MODEL_DIR = "./bert_relation_model"           # 训练好的关系分类器目录
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载关系分类模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
relation_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
relation_model.to(DEVICE)
relation_model.eval()

# 加载标签映射
with open(f"{MODEL_DIR}/label_mapping.json", "r", encoding='utf-8') as f:
    mapping = json.load(f)
label2id = mapping["label2id"]
id2label = {int(k): v for k, v in mapping["id2label"].items()}  # 确保 key 为 int

# 加载预定义的思政标签列表（从原始数据中提取，或手动定义）
# 这里我们从原始数据文件中提取所有不重复的 ideology_labels
def load_ideology_labels(data_file="computer_ideology_data.txt"):
    labels = set()
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            for lab in data["ideology_labels"]:
                labels.add(lab)
    return sorted(list(labels))

IDEOLOGY_LABELS = load_ideology_labels()
print(f"已加载 {len(IDEOLOGY_LABELS)} 个思政标签: {IDEOLOGY_LABELS[:5]}...")

# 可选：加载语义相似度模型（用于模糊匹配）
# 需要安装 sentence-transformers: pip install sentence-transformers
SIM_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=DEVICE)

# ========== 辅助函数 ==========
def create_input(sentence, subj, obj):
    """与训练时一致的标记方式"""
    sent_encoded = sentence.replace(subj, "[E1]" + subj + "[/E1]", 1)
    sent_encoded = sent_encoded.replace(obj, "[E2]" + obj + "[/E2]", 1)
    return sent_encoded

def predict_relation(sentence, subject, object_text):
    """返回 (label, confidence)，label 为 '体现' 或 '不体现'"""
    text = create_input(sentence, subject, object_text)
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    with torch.no_grad():
        logits = relation_model(input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()
    label = id2label[pred_id]   # '体现' 或 '不体现'
    return label, confidence

def extract_candidate_objects(sentence, subject, use_similarity=False, similarity_threshold=0.7):
    """
    从句子中提取候选客体（思政元素）
    方法1（默认）：直接匹配预定义思政标签在句子中的出现
    方法2（use_similarity=True）：使用语义相似度匹配，对句子中的每个词/短语计算与预定义标签的相似度
    """
    candidates = []
    if not use_similarity:
        # 简单字符串匹配（精确匹配）
        for label in IDEOLOGY_LABELS:
            if label in sentence:
                # 避免匹配到自身（如果主体也是思政标签？但主体是计算机概念，一般不会冲突）
                candidates.append(label)
        # 去重
        candidates = list(set(candidates))
    else:
        # 语义相似度匹配：将句子分词，对每个词（或连续词）与所有思政标签计算相似度
        # 更高效的方法：用 sentence-transformers 对句子和所有标签编码，然后计算相似度矩阵
        # 这里简化：只匹配与某个思政标签相似度超过阈值的词（需要预先分词）
        # 实际生产中可以更精细，比如提取名词短语
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', sentence)  # 简单分词
        # 对每个词，计算与每个标签的相似度
        word_emb = SIM_MODEL.encode(words, convert_to_tensor=True)
        label_emb = SIM_MODEL.encode(IDEOLOGY_LABELS, convert_to_tensor=True)
        similarities = util.cos_sim(word_emb, label_emb)  # (len(words), len(labels))
        for i, word in enumerate(words):
            max_sim, max_idx = torch.max(similarities[i], dim=0)
            if max_sim >= similarity_threshold:
                matched_label = IDEOLOGY_LABELS[max_idx]
                # 避免重复添加
                if matched_label not in candidates:
                    candidates.append(matched_label)
    return candidates

def infer_objects(sentence, subject, use_similarity=False, similarity_threshold=0.7):
    """
    主函数：给定句子和主体，返回所有存在 '体现' 关系的客体及其置信度
    """
    candidates = extract_candidate_objects(sentence, subject, use_similarity, similarity_threshold)
    results = []
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