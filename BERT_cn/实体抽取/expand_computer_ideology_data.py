import json
import random
from pathlib import Path

# 加载实体
def load_entities():
    datamain = Path(__file__).parent / "datamain.txt"
    computer_whitelist = Path(__file__).parent.parent.parent / "data" / "computer_terms_whitelist.txt"
    ideology_whitelist = Path(__file__).parent.parent.parent / "data" / "ideology_terms_whitelist.txt"

    computer = []
    ideology = []

    if datamain.exists():
        for line in datamain.read_text(encoding="utf-8").splitlines():
            if "COMPUTER_LABELS" in line:
                computer = eval(line.split("=", 1)[1].strip())
            elif "IDEOLOGY_LABELS" in line:
                ideology = eval(line.split("=", 1)[1].strip())

    if computer_whitelist.exists():
        computer.extend([line.strip() for line in computer_whitelist.read_text(encoding="utf-8").splitlines() if line.strip()])

    if ideology_whitelist.exists():
        ideology.extend([line.strip() for line in ideology_whitelist.read_text(encoding="utf-8").splitlines() if line.strip()])

    return list(set(computer)), list(set(ideology))

computer_entities, ideology_entities = load_entities()

# 模板生成更多数据
templates = [
    "学习{comp}的过程中，培养了{ideo}。",
    "在{comp}的教学中，强调{ideo}的重要性。",
    "{comp}与{ideo}紧密相关。",
    "通过{comp}的学习，学生树立了{ideo}的观念。",
    "{comp}的实践体现了{ideo}的精神。",
]

random.seed(42)
new_data = []

for _ in range(1000):  # 生成1000条新数据
    comp = random.choice(computer_entities)
    ideo = random.choice(ideology_entities)
    template = random.choice(templates)
    text = template.format(comp=comp, ideo=ideo)
    new_data.append({
        "id": len(new_data) + 1000,  # 假设原有id从1开始
        "text": text,
        "computer_label": comp,
        "ideology_labels": [ideo]
    })

# 追加到文件
with open(Path(__file__).parent / "computer_ideology_data.txt", "a", encoding="utf-8") as f:
    for item in new_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"追加了{len(new_data)}条新数据到computer_ideology_data.txt")
