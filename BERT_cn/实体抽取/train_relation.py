import json
import os
from pathlib import Path


def _sanitize_thread_env() -> None:
    """Ensure OpenMP-related env vars are valid positive integers."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            if int(value) <= 0:
                raise ValueError
        except Exception:
            os.environ[key] = "1"


_sanitize_thread_env()

# 在离线环境下避免触发 HuggingFace 网络请求
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ---------- 配置 ----------
DEFAULT_LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "bert-base-chinese"
MODEL_DIR = Path(os.environ.get("BERT_MODEL_DIR", str(DEFAULT_LOCAL_MODEL_DIR))).expanduser().resolve()
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./bert_relation_model"


def _validate_local_model_dir(model_dir: Path) -> None:
    required_files = ("config.json", "pytorch_model.bin", "vocab.txt")
    if not model_dir.exists():
        raise FileNotFoundError(f"本地模型目录不存在: {model_dir}")
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"本地模型目录不完整: {model_dir}; 缺少文件: {', '.join(missing)}"
        )


_validate_local_model_dir(MODEL_DIR)
print(f"使用本地BERT模型: {MODEL_DIR}")

# ---------- 数据加载 ----------
def load_data(file_path):
    samples = []  # (sentence, subject, object, label)
    label_set = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            samples.append((item["sentence"], item["subject"], item["object"], item["label"]))
            label_set.add(item["label"])
    # 标签映射：0->不体现，1->体现
    label2id = {label: idx for idx, label in enumerate(sorted(label_set))}
    id2label = {idx: label for label, idx in label2id.items()}
    return samples, label2id, id2label

samples, label2id, id2label = load_data("train.jsonl")
print("标签映射:", label2id)
print("正负样本数:", {id2label[k]: sum(1 for _,_,_,l in samples if l==k) for k in label2id})

# ---------- 构造输入（标记主体和客体）----------
def create_input(sentence, subj, obj):
    # 用特殊标记包裹实体（只替换第一次出现）
    # 注意：如果 subj 或 obj 包含特殊字符（如括号），可能需要转义，但这里简单处理
    sent_encoded = sentence.replace(subj, "[E1]" + subj + "[/E1]", 1)
    sent_encoded = sent_encoded.replace(obj, "[E2]" + obj + "[/E2]", 1)
    return sent_encoded

class RelationDataset(Dataset):
    def __init__(self, samples, tokenizer, label2id, max_len):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.inputs = []
        self.labels = []
        for sent, subj, obj, label in samples:
            text = create_input(sent, subj, obj)
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_len,
                return_tensors='pt'
            )
            self.inputs.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            })
            self.labels.append(label)   # label 已经是整数
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'],
            'attention_mask': self.inputs[idx]['attention_mask'],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
dataset = RelationDataset(samples, tokenizer, label2id, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------- 模型 ----------
model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
    num_labels=len(label2id)   # 2 类
)
model.to(DEVICE)

# ---------- 优化器与调度器 ----------
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# ---------- 训练循环 ----------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

# ---------- 保存模型及映射 ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 保存 label2id 和 id2label
with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w", encoding='utf-8') as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

print(f"模型已保存至 {OUTPUT_DIR}")