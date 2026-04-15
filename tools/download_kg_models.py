#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载知识图谱所需的预训练模型
"""

import sys
import argparse
import subprocess
from pathlib import Path

def download_spacy_model():
    """下载spaCy英文模型"""
    print("下载spaCy en_core_web_sm模型...")
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        check=True,
    )
    print("spaCy模型下载完成")

def download_blip_model():
    """下载BLIP图像字幕模型"""
    print("下载BLIP图像字幕模型...")
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration

        # 创建目录
        model_dir = Path("pretrained_models")
        model_dir.mkdir(exist_ok=True)

        # 下载模型
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # 保存到本地
        processor.save_pretrained(str(model_dir / "blip-image-captioning-base"))
        model.save_pretrained(str(model_dir / "blip-image-captioning-base"))

        print("BLIP模型下载并保存完成")

    except Exception as e:
        print(f"BLIP下载失败: {e}")
        print("请手动运行: from transformers import BlipProcessor, BlipForConditionalGeneration; processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base'); processor.save_pretrained('./pretrained_models/blip-image-captioning-base'); model.save_pretrained('./pretrained_models/blip-image-captioning-base')")

def parse_args():
    parser = argparse.ArgumentParser(description="下载知识图谱所需的预训练模型")
    parser.add_argument("--skip-spacy", action="store_true", help="跳过 spaCy 模型下载")
    parser.add_argument("--skip-blip", action="store_true", help="跳过 BLIP 模型下载")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("开始下载知识图谱预训练模型...")

    try:
        if not args.skip_spacy:
            download_spacy_model()

        if not args.skip_blip:
            download_blip_model()

    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        return 1
    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        return 1

    print("所有模型下载完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
