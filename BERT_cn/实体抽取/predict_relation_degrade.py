import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载 spaCy 中文模型
nlp_spacy = spacy.load("zh_core_web_sm")
#或加载本地模型
#nlp_spacy = spacy.load("zh_core_web_trf")

# 加载微调好的 BERT 关系分类器
relation_model_path = "./bert_relation_model"
tokenizer_bert = AutoTokenizer.from_pretrained(relation_model_path)
model_bert = AutoModelForSequenceClassification.from_pretrained(relation_model_path)
model_bert.eval()
model_bert.to(DEVICE)

# 关系标签映射（需要与训练时一致，可从保存的配置中读取）
# 这里假设训练时我们保存了 id2rel，实际可以保存到文件，此处为示例手动定义
id2rel = {0: "创始人", 1: "首都", 2: "就读", 3: "作者"}  # 需与实际一致
rel2id = {v: k for k, v in id2rel.items()}

def create_input_for_relation(sentence, subj, obj):
    """为关系分类构造输入文本（与训练时一致）"""
    sent_encoded = sentence.replace(subj, "[E1]" + subj + "[/E1]", 1)
    sent_encoded = sent_encoded.replace(obj, "[E2]" + obj + "[/E2]", 1)
    return sent_encoded

def predict_relation(sentence, subj, obj):
    """预测两个实体之间的关系"""
    text = create_input_for_relation(sentence, subj, obj)
    encoding = tokenizer_bert(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    with torch.no_grad():
        logits = model_bert(input_ids, attention_mask=attention_mask).logits
        pred_id = torch.argmax(logits, dim=1).item()
    return id2rel[pred_id]

def extract_relations(sentence):
    """主函数：给定句子，返回所有可能的关系三元组"""
    doc = nlp_spacy(sentence)
    # 获取所有实体（文本，标签）
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    relations = []
    # 对每一对实体尝试预测关系
    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j:
                continue
            subj_text, subj_type = entities[i]
            obj_text, obj_type = entities[j]
            # 可以添加启发式规则：只考虑特定类型的实体对，比如人物-组织
            # 这里简单对所有对都预测
            rel = predict_relation(sentence, subj_text, obj_text)
            # 仅当关系不是 "无关系" 时才输出（如果模型有 "无关系" 类，需要过滤）
            # 本例中没有 "无关系" 类，所以默认都会输出一个类
            relations.append((subj_text, rel, obj_text))
    return relations

# 测试
test_sentence = "马云创立了阿里巴巴集团。"
result = extract_relations(test_sentence)
print("抽取的关系：")
for subj, rel, obj in result:
    print(f"({subj}, {rel}, {obj})")