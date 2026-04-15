#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地预训练模型管理工具
优先使用项目内的本地模型，避免自动下载外部依赖
支持模型路径配置、完整性检查、缓存管理
"""

import os
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalModelManager:
    """本地模型管理器"""

    SEARCH_ROOTS = (
        Path("."),
        Path("pretrained_models"),
        Path("pretrain_models"),
        Path("configs") / "pretrain",
    )
    
    # 本地模型配置
    LOCAL_MODELS = {
        'resnet152': {
            'name': 'ResNet152',
            'paths': [
                'pretrained_models/resnet152-394f9c45.pth',
                'pretrained_models/image_feature_resnet152.pth'
            ],
            'size_mb': 230,
            'type': 'image_feature_extractor',
            'required': True,
            'description': '图像特征提取模型'
        },
        'msrvtt_tdconved': {
            'name': 'TDConvED (MSR-VTT)',
            'paths': [
                'pretrained_models/MSR-VTT_TDConvED.pth',
                'configs/video_caption/msrvtt/tdconved/'
            ],
            'size_mb': 1000,
            'type': 'video_captioning',
            'required': True,
            'description': '视频字幕生成模型 (MSR-VTT)'
        },
        'image_caption_cider_tden': {
            'name': 'TDEN (CIDEr Score Optimization)',
            'paths': [
                'configs/pretrain/CIDEr Score Optimization_TDEN.pth',
                'pretrained_models/image_caption_cider_tden.pth'
            ],
            'size_mb': 1000,
            'type': 'image_captioning',
            'required': False,
            'description': '图像字幕优化模型'
        },
        'image_caption_attention_ce': {
            'name': 'Attention (Cross-Entropy Loss)',
            'paths': [
                'configs/pretrain/Cross-Entropy Loss_Attention.pth',
                'pretrained_models/image_caption_attention_ce.pth'
            ],
            'size_mb': 1000,
            'type': 'image_captioning',
            'required': False,
            'description': '注意力机制图像字幕'
        },
        'video_caption_msvd_ta': {
            'name': 'TA (Video Captioning on MSVD)',
            'paths': [
                'configs/pretrain/MSVD_TA.pth',
                'pretrained_models/video_caption_msvd_ta.pth'
            ],
            'size_mb': 1000,
            'type': 'video_captioning',
            'required': False,
            'description': 'MSVD视频字幕模型'
        },
        'image_retrieval_flickr_tden': {
            'name': 'TDEN (Caption-based Image Retrieval)',
            'paths': [
                'configs/pretrain/Caption-based image retrieval on Flickr30k_TDEN.pth',
                'pretrained_models/image_retrieval_flickr_tden.pth'
            ],
            'size_mb': 1000,
            'type': 'image_retrieval',
            'required': False,
            'description': '基于字幕的图像检索'
        },
        'visual_reasoning_tden': {
            'name': 'TDEN (Visual Commonsense Reasoning)',
            'paths': [
                'configs/pretrain/Visual commonsense reasoning_TDEN.pth',
                'pretrained_models/visual_reasoning_tden.pth'
            ],
            'size_mb': 1000,
            'type': 'visual_reasoning',
            'required': False,
            'description': '视觉常识推理模型'
        },
        'vqa_uniter': {
            'name': 'Uniter (Visual Question Answering)',
            'paths': [
                'configs/pretrain/Visual Question Answering_Uniter.pth',
                'pretrained_models/vqa_uniter.pth'
            ],
            'size_mb': 1000,
            'type': 'visual_qa',
            'required': False,
            'description': '视觉问答模型'
        }
    }
    
    # HuggingFace模型配置（需要下载）
    HUGGINGFACE_MODELS = {
        'blip': {
            'name': 'BLIP Image Captioning Base',
            'model_id': 'Salesforce/blip-image-captioning-base',
            'size_mb': 360,
            'type': 'image_video_captioning',
            'required': True,
            'description': '图像/视频自动标注模型',
            'local_path': 'pretrained_models/blip-image-captioning-base',
            'local_paths': [
                'pretrained_models/blip-image-captioning-base',
                'pretrain_models/blip-image-captioning-base',
            ],
        },
        'bert': {
            'name': 'BERT Base (Local)',
            'model_id': 'bert-base-chinese',
            'size_mb': 400,
            'type': 'semantic_matching',
            'required': False,
            'description': '中文语义匹配模型',
            'local_path': 'BERT_cn/bert-base-chinese',
            'local_paths': [
                'BERT_cn/bert-base-chinese',
                'pretrained_models/BERT',
                'pretrain_models/BERT',
            ],
        },
        'spacy': {
            'name': 'spaCy zh_core_web_sm',
            'model_id': 'zh_core_web_sm',
            'size_mb': 40,
            'type': 'nlp_ner',
            'required': True,
            'description': '中文命名实体识别模型',
            'local_path': 'BERT_cn/zh_core_web_sm-3.8.0-py3-none-any.whl',
            'local_paths': [
                'BERT_cn/zh_core_web_sm-3.8.0-py3-none-any.whl',
            ],
            'command': 'python -m pip install ./BERT_cn/zh_core_web_sm-3.8.0-py3-none-any.whl'
        }
    }
    
    def __init__(self, project_root: str = '.'):
        """
        初始化模型管理器
        
        Args:
            project_root: 项目根目录
        """
        self.project_root = Path(project_root)
        self.model_cache = {}

    def _resolve_existing_path(self, rel_or_abs: str) -> Optional[Path]:
        """在多个候选根目录中解析并返回首个存在路径。"""
        raw = Path(rel_or_abs)
        candidates = [raw] if raw.is_absolute() else [self.project_root / raw]

        if not raw.is_absolute():
            for root in self.SEARCH_ROOTS:
                candidates.append(self.project_root / root / raw)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None
    
    def find_local_model(self, model_key: str) -> Optional[Path]:
        """
        查找本地模型
        
        Args:
            model_key: 模型键（如'resnet152'）
            
        Returns:
            模型路径，如果未找到返回None
        """
        if model_key not in self.LOCAL_MODELS:
            logger.warning(f"未知模型: {model_key}")
            return None
        
        model_info = self.LOCAL_MODELS[model_key]
        
        for rel_path in model_info['paths']:
            full_path = self._resolve_existing_path(rel_path)
            if full_path:
                logger.info(f"✅ 找到本地模型 {model_key}: {full_path}")
                return full_path
        
        logger.warning(f"❌ 未找到本地模型 {model_key}")
        return None
    
    def list_available_models(self) -> Dict[str, Dict]:
        """
        列出所有可用模型（本地+HuggingFace）
        
        Returns:
            模型字典
        """
        available = {}
        
        # 列出本地模型
        for key, info in self.LOCAL_MODELS.items():
            status = "✅" if self.find_local_model(key) else "❌"
            available[f"local_{key}"] = {
                'name': info['name'],
                'status': status,
                'type': info['type'],
                'required': info['required'],
                'size_mb': info['size_mb'],
                'description': info['description']
            }
        
        # 列出HuggingFace模型
        for key, info in self.HUGGINGFACE_MODELS.items():
            model_path = self.get_model_path(key)
            available[f"hf_{key}"] = {
                'name': info['name'],
                'status': '✅' if model_path else '❓',
                'type': info['type'],
                'required': info['required'],
                'size_mb': info['size_mb'],
                'description': info['description']
            }
        
        return available
    
    def print_model_status(self):
        """打印模型状态总结"""
        print("\n" + "="*80)
        print("本地预训练模型状态")
        print("="*80)
        
        # 本地模型状态
        print("\n【本地模型】")
        print("-" * 80)
        required_count = 0
        available_count = 0
        
        for key, info in self.LOCAL_MODELS.items():
            path = self.find_local_model(key)
            status = "✅ 已有" if path else "❌ 缺失"
            required = "必需" if info['required'] else "可选"
            print(f"  {status} | {info['name']:30s} | {required:4s} | {info['size_mb']:5d}MB")
            print(f"     描述: {info['description']}")
            if info['required']:
                required_count += 1
                if path:
                    available_count += 1
        
        # HuggingFace模型状态
        print("\n【需要下载的模型】")
        print("-" * 80)
        for key, info in self.HUGGINGFACE_MODELS.items():
            required = "必需" if info['required'] else "可选"
            status = "✅ 已有" if self.get_model_path(key) else "❓ 待处理"
            print(f"  {status} | {info['name']:30s} | {required:4s} | {info['size_mb']:5d}MB")
            print(f"     描述: {info['description']}")
            if 'command' in info:
                print(f"     命令: {info['command']}")
            else:
                print(f"     模型: {info['model_id']}")
        
        # 统计
        print("\n" + "="*80)
        print("统计信息")
        print("="*80)
        print(f"✅ 已有本地模型: {available_count}/{required_count} (必需模型)")
        print(f"❌ 缺失模型: {required_count - available_count}")
        print(f"❓ HuggingFace模型: {len(self.HUGGINGFACE_MODELS)}")
        
        # 存储需求
        local_size = sum(info['size_mb'] for info in self.LOCAL_MODELS.values())
        hf_size = sum(info['size_mb'] for info in self.HUGGINGFACE_MODELS.values())
        required_size = sum(
            info['size_mb'] for info in self.LOCAL_MODELS.values() 
            if info['required']
        ) + sum(
            info['size_mb'] for info in self.HUGGINGFACE_MODELS.values() 
            if info['required']
        )
        
        print(f"\n📊 存储需求:")
        print(f"   本地模型: ~{local_size}MB")
        print(f"   HuggingFace模型: ~{hf_size}MB")
        print(f"   必需总计: ~{required_size}MB")
        print(f"   全部: ~{local_size + hf_size}MB")
        print("="*80 + "\n")
    
    def get_model_path(self, model_key: str) -> Optional[str]:
        """
        获取模型路径（优先本地）
        
        Args:
            model_key: 模型键
            
        Returns:
            模型路径字符串，如果未找到返回None
        """
        if model_key == 'spacy' and self._spacy_package_installed('zh_core_web_sm'):
            return 'zh_core_web_sm'

        if model_key in self.LOCAL_MODELS:
            path = self.find_local_model(model_key)
            if path:
                return str(path)

        hf_info = self.HUGGINGFACE_MODELS.get(model_key)
        if hf_info:
            local_paths = hf_info.get('local_paths') or [hf_info.get('local_path')]
            for rel_path in [p for p in local_paths if p]:
                resolved = self._resolve_existing_path(rel_path)
                if resolved:
                    logger.info(f"✅ 找到本地HuggingFace模型 {model_key}: {resolved}")
                    return str(resolved)
        return None

    @staticmethod
    def _spacy_package_installed(package_name: str) -> bool:
        try:
            from spacy.util import is_package
            return bool(is_package(package_name))
        except Exception:
            return False

    def _install_python_package(self, package_path: Path) -> bool:
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--no-deps', str(package_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                logger.info(f"已安装本地Python包: {package_path}")
                return True
            logger.warning(f"本地包安装失败: {package_path}\n{result.stderr.strip()}")
            return False
        except Exception as e:
            logger.warning(f"安装本地包异常: {package_path} -> {e}")
            return False

    def ensure_spacy_model_installed(self, language: str = 'zh', auto_install: bool = True) -> bool:
        """确保spaCy中文模型可用；优先使用本地 wheel。"""
        package_name = 'zh_core_web_sm' if language == 'zh' else 'en_core_web_sm'
        if self._spacy_package_installed(package_name):
            return True

        if language != 'zh':
            return False

        wheel_path = self.get_model_path('spacy')
        if not wheel_path or not auto_install:
            return False

        if self._install_python_package(Path(wheel_path)):
            return self._spacy_package_installed(package_name)
        return False

    def get_preferred_spacy_model(self, language: str = 'zh', auto_install: bool = True) -> Optional[str]:
        """返回最适合加载的spaCy模型名；中文优先，英文回退。"""
        if language == 'zh':
            if self._spacy_package_installed('zh_core_web_sm'):
                return 'zh_core_web_sm'

            wheel_path = self.get_model_path('spacy')
            if wheel_path and auto_install and self._install_python_package(Path(wheel_path)):
                if self._spacy_package_installed('zh_core_web_sm'):
                    return 'zh_core_web_sm'

            # 中文模式下，只有在中文包不可用且未能安装时，才允许英文回退
            if self._spacy_package_installed('en_core_web_sm'):
                return 'en_core_web_sm'
            return None

        return 'en_core_web_sm' if self._spacy_package_installed('en_core_web_sm') else None
    
    def export_model_config(self, output_path: str = 'model_config.json'):
        """
        导出模型配置文件
        
        Args:
            output_path: 输出配置文件路径
        """
        config = {
            'local_models': {},
            'huggingface_models': {}
        }
        
        # 导出本地模型配置
        for key, info in self.LOCAL_MODELS.items():
            path = self.find_local_model(key)
            config['local_models'][key] = {
                'name': info['name'],
                'found': path is not None,
                'path': str(path) if path else None,
                'type': info['type'],
                'required': info['required'],
                'size_mb': info['size_mb']
            }
        
        # 导出HuggingFace模型配置
        for key, info in self.HUGGINGFACE_MODELS.items():
            config['huggingface_models'][key] = {
                'name': info['name'],
                'model_id': info['model_id'],
                'required': info['required'],
                'size_mb': info['size_mb'],
                'local_path': info.get('local_path')
            }
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型配置已导出到: {output_path}")
    
    def check_required_models(self) -> Tuple[bool, Dict[str, str]]:
        """
        检查必需模型是否完整
        
        Returns:
            (是否完整, 缺失模型字典)
        """
        missing = {}
        
        for key, info in self.LOCAL_MODELS.items():
            if info['required']:
                path = self.find_local_model(key)
                if not path:
                    missing[key] = info['name']
        
        for key, info in self.HUGGINGFACE_MODELS.items():
            if info['required']:
                if not self.get_model_path(key):
                    missing[f"hf_{key}"] = info['name']
        
        return len(missing) == 0, missing


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='本地模型管理工具')
    parser.add_argument('--status', action='store_true', help='显示模型状态')
    parser.add_argument('--list', action='store_true', help='列出所有模型')
    parser.add_argument('--find', type=str, help='查找指定模型')
    parser.add_argument('--export', type=str, default='model_config.json', 
                        help='导出模型配置')
    parser.add_argument('--check', action='store_true', help='检查必需模型')
    
    args = parser.parse_args()
    
    manager = LocalModelManager()
    
    if not any(vars(args).values()):
        args.status = True
    
    if args.status:
        manager.print_model_status()
    
    if args.list:
        models = manager.list_available_models()
        print("\n所有模型:")
        for key, info in models.items():
            print(f"  {info['status']} {key:30s} | {info['name']}")
    
    if args.find:
        path = manager.find_local_model(args.find)
        if path:
            print(f"✅ 找到: {path}")
        else:
            print(f"❌ 未找到: {args.find}")
    
    if args.export:
        manager.export_model_config(args.export)
    
    if args.check:
        complete, missing = manager.check_required_models()
        if complete:
            print("✅ 所有必需模型已完整")
        else:
            print("❌ 缺失以下模型:")
            for key, name in missing.items():
                print(f"   - {key}: {name}")


if __name__ == '__main__':
    main()

