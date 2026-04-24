import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

NEGATIVE_RATIO = 2
ROOT = Path(__file__).resolve().parent
input_file = ROOT / "computer_ideology_data.txt"
output_file = ROOT / "train.jsonl"

TEACHING_TEMPLATES = [
    "在大一基础课程中，{sentence}",
    "面向初学者的{computer}教学中，{sentence}",
    "在{computer}入门教学场景里，{sentence}",
]


def augment_sentence(sentence: str, computer: str):
    variants = [sentence]
    for template in TEACHING_TEMPLATES:
        variants.append(template.format(sentence=sentence, computer=computer))
    return list(dict.fromkeys(item for item in variants if item))


# 读取所有数据
all_data = []
computer_to_ideologies = defaultdict(list)
ideology_set = set()

with input_file.open('r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        all_data.append(data)
        computer = data["computer_label"]
        ideologies = data["ideology_labels"]
        computer_to_ideologies[computer].extend(ideologies)
        for ideo in ideologies:
            ideology_set.add(ideo)

ideology_list = sorted(ideology_set)

# 生成正负样本
samples = []

for data in all_data:
    sentence = data["text"]
    computer = data["computer_label"]
    ideologies = data["ideology_labels"]

    augmented_sentences = augment_sentence(sentence, computer)
    for sent_variant in augmented_sentences:
        # 正样本：每个 ideology 都是一个正例
        for ideo in ideologies:
            samples.append((sent_variant, computer, ideo, 1))

        # 负样本：每个正样本生成 NEGATIVE_RATIO 个负例，增强“非关系”判别能力。
        neg_count = len(ideologies) * max(1, int(NEGATIVE_RATIO))
        candidates = [i for i in ideology_list if i not in ideologies]
        if not candidates:
            continue
        if neg_count <= len(candidates):
            neg_samples = random.sample(candidates, k=neg_count)
        else:
            neg_samples = [random.choice(candidates) for _ in range(neg_count)]
        for neg_ideo in neg_samples:
            samples.append((sent_variant, computer, neg_ideo, 0))

# 打乱样本
random.shuffle(samples)

# 写入 jsonl
with output_file.open('w', encoding='utf-8') as f:
    for sent, subj, obj, label in samples:
        record = {
            "sentence": sent,
            "subject": subj,
            "object": obj,
            "label": label,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"生成 {len(samples)} 条训练样本，保存至 {output_file}")
