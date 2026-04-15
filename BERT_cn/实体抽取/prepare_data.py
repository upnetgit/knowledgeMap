import json
import random
from collections import defaultdict

input_file = "computer_ideology_data.txt"
output_file = "train.jsonl"

# 读取所有数据
all_data = []
computer_to_ideologies = defaultdict(list)   # 每个计算机概念对应的思政标签集
ideology_set = set()

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        all_data.append(data)
        computer = data["computer_label"]
        ideologies = data["ideology_labels"]
        computer_to_ideologies[computer].extend(ideologies)
        for ideo in ideologies:
            ideology_set.add(ideo)

ideology_list = list(ideology_set)

# 生成正负样本
samples = []   # 每个元素: (sentence, subject, object, label)
label_map = {"不体现": 0, "体现": 1}

for data in all_data:
    sentence = data["text"]
    computer = data["computer_label"]
    ideologies = data["ideology_labels"]
    
    # 正样本：每个 ideology 都是一个正例
    for ideo in ideologies:
        samples.append((sentence, computer, ideo, 1))
    
    # 负样本：随机抽取一个不在当前 ideologies 中的 ideology
    # 为了保证负样本数量与正样本大致平衡，每个句子生成与正样本相同数量的负样本（或者固定数量）
    # 这里每个正样本对应生成一个负样本（随机选不相关的 ideology）
    for _ in range(len(ideologies)):
        # 从全局 ideology 中选一个不在当前 ideologies 中的
        candidates = [i for i in ideology_list if i not in ideologies]
        if not candidates:
            continue
        neg_ideo = random.choice(candidates)
        samples.append((sentence, computer, neg_ideo, 0))

# 打乱样本
random.shuffle(samples)

# 写入 jsonl
with open(output_file, 'w', encoding='utf-8') as f:
    for sent, subj, obj, label in samples:
        record = {
            "sentence": sent,
            "subject": subj,
            "object": obj,
            "label": label
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"生成 {len(samples)} 条训练样本，保存至 {output_file}")