# batch_inference.py
import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
import csv
from xmodaler.config import get_cfg
from xmodaler.modeling import build_model
from xmodaler.functional import load_vocab, decode_sequence

def load_vocab_tdconved(vocab_path):
    """加载词汇表（X-modaler 格式：每行一个词，索引从0开始）"""
    with open(vocab_path, 'r') as f:
        vocab = [line.strip() for line in f]
    # 确保特殊 token 存在（TDConvED 期望 <BOS>, <EOS>, <PAD>, <UNK>）
    # 通常词汇表第一个词是 '.' 或 '<PAD>'，第二个可能是 '<BOS>'，根据实际情况调整
    # 这里假设词汇表已包含所需 token
    return vocab

def main():
    # 配置
    cfg_file = "configs/video_caption/msrvtt/tdconved/tdconved.yaml"
    weight_file = "./pretrained_models/Video Captioning on MSR-VTT_TDConvED.pth"
    vocab_file = "./data/annotations/vocabulary.txt"
    feat_dir = "./data/features"          # 存放 .npy 特征的目录
    output_csv = "./results.csv"          # 输出结果文件

    # 加载配置
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = weight_file
    cfg.freeze()

    # 构建模型
    model = build_model(cfg)
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model

    # 加载词汇表
    vocab = load_vocab_tdconved(vocab_file)
    # X-modaler 的解码函数 decode_sequence 要求 vocab 是 list，索引即 token id
    # 注意：模型输出 token id 范围 [0, len(vocab)-1]

    # 获取所有特征文件
    feat_files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
    results = []

    for feat_file in tqdm(feat_files, desc="Inference"):
        video_id = os.path.splitext(feat_file)[0]
        feat_path = os.path.join(feat_dir, feat_file)
        feat = np.load(feat_path)          # shape (50, 2048)
        feat_tensor = torch.from_numpy(feat).float().unsqueeze(0)  # (1, 50, 2048)
        mask = torch.ones(1, 50).float()   # 所有帧有效

        # 构造输入字典
        batched_inputs = {
            'att_feats': feat_tensor,
            'att_masks': mask,
            'ids': [video_id]
        }

        # 推理
        with torch.no_grad():
            outputs = model(batched_inputs, use_beam_search=True, output_sents=True)
            # outputs['output'] 是 list of strings
            caption = outputs['output'][0]

        results.append({'video_id': video_id, 'caption': caption})
        print(f"{video_id}: {caption}")

    # 保存结果到 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'caption'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_csv}")

if __name__ == '__main__':
    main()