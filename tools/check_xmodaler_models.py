#!/usr/bin/env python3
"""
检查 XModaler 框架中所有预训练模型的可用性
列出每个模型的位置、大小和加载状态
"""

from pathlib import Path
from typing import Dict, List
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    # Avoid Windows GBK console crashes when printing Unicode symbols.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 模型清单
MODELS_INVENTORY = {
    "视频字幕生成 (Video Captioning)": [
        {
            "name": "TDConvED (MSR-VTT)",
            "paths": [
                "pretrained_models/MSR-VTT_TDConvED.pth",
                "configs/video_caption/msrvtt/tdconved/tdconved.yaml"
            ],
            "type": "video_captioning",
            "priority": "⭐⭐⭐⭐⭐ 推荐",
            "usage": "python build_kg.py --use-xmodaler-video --xmodaler-model-type tdconved"
        },
        {
            "name": "TA (MSVD)",
            "paths": [
                "configs/pretrain/MSVD_TA.pth",
                "configs/video_caption/msvd/ta/ta.yaml"
            ],
            "type": "video_captioning",
            "priority": "⭐⭐⭐ 备选",
            "usage": "python build_kg.py --use-xmodaler-video --xmodaler-model-type ta"
        }
    ],
    "图像字幕生成 (Image Captioning)": [
        {
            "name": "TDEN (CIDEr Score Optimization)",
            "paths": [
                "configs/pretrain/CIDEr Score Optimization_TDEN.pth",
                "configs/pretrain/tden/tden.yaml"
            ],
            "type": "image_captioning",
            "priority": "⭐⭐⭐⭐ 推荐替代BLIP",
        },
        {
            "name": "Attention (Cross-Entropy Loss)",
            "paths": [
                "configs/pretrain/Cross-Entropy Loss_Attention.pth",
                "configs/pretrain/tden/tden.yaml"
            ],
            "type": "image_captioning",
            "priority": "⭐⭐⭐ 轻量化",
        },
        {
            "name": "BLIP (Image Captioning Base)",
            "paths": [
                "pretrained_models/blip-image-captioning-base/",
            ],
            "type": "image_captioning",
            "priority": "⭐⭐⭐ 当前使用",
        }
    ],
    "图像检索 (Image Retrieval)": [
        {
            "name": "TDEN (Caption-based Image Retrieval on Flickr30k)",
            "paths": [
                "pretrained_models/TDEN_RETRIEVAL/TDEN_RETRIEVAL.pth",
                "configs/pretrain/tden/tden.yaml"
            ],
            "type": "image_retrieval",
            "priority": "⭐⭐⭐⭐ 多模态链接增强",
        }
    ],
    "视觉问答 (Visual Question Answering)": [
        {
            "name": "UNITER (VQA)",
            "paths": [
                "pretrained_models/Uniter/UNITER.pth",
                "configs/pretrain/uniter/uniter.yaml"
            ],
            "type": "vqa",
            "priority": "⭐⭐⭐ 需要微调",
        }
    ],
    "视觉常识推理 (Visual Commonsense Reasoning)": [
        {
            "name": "TDEN (Visual Commonsense Reasoning)",
            "paths": [
                "pretrained_models/TDEN/TDEN.pth",
                "configs/pretrain/tden/tden.yaml"
            ],
            "type": "visual_reasoning",
            "priority": "⭐⭐⭐⭐ 现象理解",
        }
    ],
    "视觉-语言预训练 (Vision-Language Pretraining)": [
        {
            "name": "TDEN (Pretrain)",
            "paths": [
                "pretrained_models/TDEN/TDEN.pth",
                "configs/pretrain/tden/tden.yaml"
            ],
            "type": "pretrain",
            "priority": "⭐⭐⭐⭐⭐ SOTA微调基础",
        },
        {
            "name": "UNITER (Pretrain)",
            "paths": [
                "pretrained_models/Uniter/UNITER.pth",
                "configs/pretrain/uniter/uniter.yaml"
            ],
            "type": "pretrain",
            "priority": "⭐⭐⭐⭐ 通用基础",
        }
    ],
    "特征提取 (Feature Extraction)": [
        {
            "name": "ResNet152",
            "paths": [
                "pretrained_models/resnet152-394f9c45.pth",
            ],
            "type": "feature_extractor",
            "priority": "⭐⭐⭐⭐⭐ 必需",
        }
    ],
    "语义匹配 (Semantic Matching)": [
        {
            "name": "BERT-base-chinese",
            "paths": [
                "BERT_cn/bert-base-chinese/",
            ],
            "type": "semantic",
            "priority": "⭐⭐⭐⭐⭐ 必需",
        }
    ]
}


def get_file_size(path: Path) -> str:
    """获取文件大小，返回格式化字符串"""
    if not path.exists():
        return "❌ 不存在"
    if path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
        return f"✅ {size_mb:.1f}MB"
    elif path.is_dir():
        total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        return f"✅ {size_mb:.1f}MB"
    return "⚠️  无法确定"


def check_model_availability():
    """检查所有模型的可用性"""
    results = {}

    for category, models in MODELS_INVENTORY.items():
        bucket = []
        results[category] = bucket

        for model in models:
            if not isinstance(model, dict):
                continue

            model_dict = model
            name_value = model_dict.get("name", "")
            type_value = model_dict.get("type", "")
            priority_value = model_dict.get("priority", "")
            model_name = name_value if isinstance(name_value, str) else ""
            model_type = type_value if isinstance(type_value, str) else ""
            model_priority = priority_value if isinstance(priority_value, str) else ""
            files = []
            model_info = {
                "name": model_name,
                "type": model_type,
                "priority": model_priority,
                "files": files
            }

            paths_value = model_dict.get("paths", [])
            if not isinstance(paths_value, list):
                paths_value = []

            for raw_path in paths_value:
                path_str = str(raw_path)
                path = PROJECT_ROOT / path_str
                size = get_file_size(path)
                files.append({
                    "path": path_str,
                    "status": size
                })

            bucket.append(model_info)

    return results


def print_report(results):
    """打印模型检查报告"""
    print("\n" + "="*100)
    print("XModaler 框架预训练模型可用性检查报告".center(100))
    print("="*100 + "\n")

    total_models = 0
    available_models = 0
    total_size_mb = 0

    for category, models in results.items():
        print(f"\n📚 {category}")
        print("-" * 100)

        for model in models:
            total_models += 1

            # 检查是否所有文件都可用
            all_available = all(
                "✅" in file["status"]
                for file in model["files"]
            )
            if all_available:
                available_models += 1

            status_icon = "✅" if all_available else "⚠️ "
            print(f"\n{status_icon} {model['name']}")
            print(f"   优先级: {model['priority']}")
            print(f"   类型: {model['type']}")

            for file in model["files"]:
                size_info = file["status"]
                if "MB" in size_info:
                    try:
                        size_mb = float(size_info.split()[1])
                        total_size_mb += size_mb
                    except:
                        pass
                print(f"   - {file['path']:50s} {size_info:>15s}")

    # 打印统计信息
    print("\n" + "="*100)
    print("📊 统计信息".center(100))
    print("="*100)
    print(f"总模型数: {total_models}")
    print(f"已下载模型: {available_models} ({100*available_models//total_models}%)")
    print(f"总大小: ~{total_size_mb:.1f}MB (~{total_size_mb/1024:.1f}GB)")
    print(f"可用显存需求: ~4-10GB (取决于并行加载)")

    print("\n" + "="*100)
    print("🎯 推荐使用优先级".center(100))
    print("="*100)
    print("""
第 1 优先级 (必需):
  ✅ ResNet152           - 视觉特征提取主干
  ✅ BERT-base-chinese   - 语义匹配与实体抽取
  ✅ TDConvED (MSR-VTT)  - 视频字幕生成

第 2 优先级 (推荐):
  ✅ TDEN (CIDEr)        - 图像字幕 (替代BLIP)
  ✅ TDEN (Pretrain)     - 预训练基础 (微调用)
  ✅ UNITER (Pretrain)   - 通用基础

第 3 优先级 (可选):
  ✅ TA (MSVD)           - 备选视频字幕
  ✅ TDEN (Retrieval)    - 多模态链接增强
  ✅ UNITER (VQA)        - 视觉问答
  ✅ TDEN (VCR)          - 常识推理
  ✅ Attention           - 轻量图像字幕
    """)

    print("\n" + "="*100)
    print("💡 快速开始命令".center(100))
    print("="*100)
    print("""
# 使用 TDConvED 视频字幕（当前已实现 ✅）
python build_kg.py --use-xmodaler-video --xmodaler-model-type tdconved

# 检查模型加载
python LOCAL_MODEL_MANAGER.py --status

# 测试 TDConvED 模型
python -c "
from xmodaler.kg.processors import CaptionGenerator
cap = CaptionGenerator(use_xmodaler_video=True)
print('✅ TDConvED 模型加载成功' if cap.xmodaler_model else '❌ 加载失败')
"
    """)

    print("="*100 + "\n")


def generate_json_report(results: Dict, output_path: str = "model_availability_report.json"):
    """生成 JSON 格式的报告"""
    report = {
        "timestamp": str(Path(__file__).resolve().parent),
        "total_categories": len(results),
        "categories": results
    }

    output_file = PROJECT_ROOT / output_path
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON 报告已保存到: {output_file}\n")


if __name__ == "__main__":
    print("\n正在检查模型可用性...\n")
    results = check_model_availability()
    print_report(results)
    generate_json_report(results)

