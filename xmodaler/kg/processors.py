#!/usr/bin/env python3
"""
多模态数据处理器模块
处理文本、图像、视频数据的特征提取和描述生成

优先使用项目内的本地预训练模型，避免自动下载外部依赖
支持ResNet152、BLIP、spaCy等模型的本地加载
"""

from __future__ import annotations

import os
import json
import cv2
import logging
import tempfile
import subprocess
import threading
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_zh_text(text: Optional[str]) -> str:
    """轻量中文文本规范化，便于后续实体匹配与排序。"""
    if not text:
        return ""
    compact = " ".join(str(text).strip().split())
    return compact.replace("，", ",").replace("。", ".").strip()


def get_local_model_path(model_key: str) -> Optional[Path]:
    """
    获取本地模型路径（优先使用本地模型）

    Args:
        model_key: 模型键 (如'resnet152', 'blip')

    Returns:
        模型路径，如果未找到返回None
    """
    from LOCAL_MODEL_MANAGER import LocalModelManager

    try:
        manager = LocalModelManager()
        return manager.find_local_model(model_key)
    except Exception as e:
        logger.warning(f"无法加载本地模型管理器: {e}")
        return None


def resolve_torch_device(device_mode: str = "auto"):
    """根据 device_mode 解析 torch 设备。"""
    import torch
    mode = str(device_mode or "auto").strip().lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextProcessor:
    """文本数据处理器"""
    
    def __init__(self, nlp_model=None):
        """
        初始化文本处理器
        
        Args:
            nlp_model: spaCy模型对象
        """
        self.nlp = nlp_model
        self.entity_types = {
            'PERSON': '人物',
            'ORG': '组织',
            'GPE': '地点',
            'PRODUCT': '产品',
            'EVENT': '事件',
            'LAW': '法律',
            'LANGUAGE': '语言',
            'WORK_OF_ART': '艺术作品'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        从文本提取命名实体
        
        Args:
            text: 输入文本
            
        Returns:
            按类型分类的实体字典
        """
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            ent_type = self.entity_types.get(ent.label_, ent.label_)
            if ent_type not in entities:
                entities[ent_type] = []
            entities[ent_type].append(ent.text)
        
        return entities
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        从文本提取关键词（基于TF-IDF）
        
        Args:
            text: 输入文本
            top_k: 返回前k个关键词
            
        Returns:
            关键词列表
        """
        from collections import Counter
        
        # 简单实现：词频统计
        if not self.nlp:
            words = text.lower().split()
        else:
            doc = self.nlp(text)
            words = [token.text.lower() for token in doc if not token.is_stop]
        
        word_freq = Counter(words)
        keywords = [word for word, _ in word_freq.most_common(top_k)]
        
        return keywords
    
    def load_txt_files(self, txt_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        批量加载txt文件
        
        Args:
            txt_dir: txt文件目录
            
        Returns:
            文件内容字典 {filename: {entities, keywords, raw_text}}
        """
        result = {}
        
        if not os.path.exists(txt_dir):
            logger.warning(f"目录不存在: {txt_dir}")
            return result
        
        for filename in os.listdir(txt_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(txt_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    result[filename] = {
                        'raw_text': text,
                        'entities': self.extract_entities(text),
                        'keywords': self.extract_keywords(text),
                        'length': len(text)
                    }
                    logger.info(f"已加载: {filename}")
                except Exception as e:
                    logger.error(f"加载失败 {filename}: {str(e)}")
        
        return result


class ImageProcessor:
    """图像数据处理器"""
    
    def __init__(self, model_path: Optional[str] = None, device_mode: str = "auto"):
        """
        初始化图像处理器
        
        Args:
            model_path: 预训练模型路径
        """
        self.model_path = model_path
        self.device_mode = str(device_mode or "auto").strip().lower()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            import torch
            import torchvision.models as models
            
            device = resolve_torch_device(self.device_mode)
            base_model = models.resnet152(weights=None)
            local_weight = get_local_model_path('resnet152')
            if local_weight and local_weight.exists():
                state_dict = torch.load(str(local_weight), map_location='cpu')
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                base_model.load_state_dict(state_dict, strict=False)
                logger.info(f"已加载本地ResNet152权重: {local_weight}")
            else:
                # 若本地权重不存在，尽量保持无下载模式运行
                logger.info("未找到本地ResNet152权重，使用随机初始化模型占位")

            self.model = torch.nn.Sequential(*list(base_model.children())[:-1]).to(device)
            self.model.eval()
            self.device = device
            self.feature_backend = "xmodaler_resnet152"
            logger.info("已加载ResNet152特征提取器")
        except Exception as e:
            logger.warning(f"无法加载预训练模型: {str(e)}")
            self.model = None
            self.feature_backend = "unavailable"
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        从图像提取特征向量
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            特征向量
        """
        if not self.model:
            return None
        
        try:
            from PIL import Image
            import torch
            import torchvision.transforms as transforms
            
            # 加载并预处理图像
            image = Image.open(image_path).convert('RGB')
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                features = self.model(input_batch)
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"特征提取失败 {image_path}: {str(e)}")
            return None
    
    def get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        获取图像元数据
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            元数据字典
        """
        try:
            from PIL import Image
            
            image = Image.open(image_path)
            metadata = {
                'size': image.size,
                'mode': image.mode,
                'format': image.format,
                'path': image_path,
                'filename': os.path.basename(image_path)
            }
            
            return metadata
        except Exception as e:
            logger.error(f"获取元数据失败: {str(e)}")
            return {}
    
    def load_images(self, img_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        批量加载图像
        
        Args:
            img_dir: 图像目录
            
        Returns:
            图像数据字典 {filename: {metadata, features}}
        """
        result = {}
        
        if not os.path.exists(img_dir):
            logger.warning(f"目录不存在: {img_dir}")
            return result
        
        for root, dirs, files in os.walk(img_dir):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(root, filename)
                    try:
                        category = os.path.basename(root)
                        result[filename] = {
                            'path': filepath,
                            'category': category,
                            'metadata': self.get_image_metadata(filepath),
                            'features': self.extract_features(filepath)
                        }
                        logger.info(f"已加载图像: {filename}")
                    except Exception as e:
                        logger.error(f"加载失败 {filename}: {str(e)}")
        
        return result


class VideoProcessor:
    """视频数据处理器"""
    
    def __init__(self, frames_per_video: int = 8, device_mode: str = "auto"):
        """
        初始化视频处理器
        
        Args:
            frames_per_video: 每个视频采样帧数
        """
        self.frames_per_video = frames_per_video
        self.device_mode = str(device_mode or "auto").strip().lower()
        self.image_processor = ImageProcessor(device_mode=self.device_mode)

    def _extract_frame_feature(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """直接复用图像主干提取单帧特征，避免临时落盘。"""
        if not self.image_processor.model:
            return None

        try:
            from PIL import Image
            import torch
            import torchvision.transforms as transforms

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            input_tensor = preprocess(image).unsqueeze(0).to(self.image_processor.device)
            with torch.no_grad():
                features = self.image_processor.model(input_tensor)
            return features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"帧特征提取失败: {str(e)}")
            return None
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        从视频均匀采样帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            帧列表
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                return []
            
            frame_indices = np.linspace(0, total_frames - 1, 
                                       self.frames_per_video, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            
            cap.release()
            return frames
        except Exception as e:
            logger.error(f"帧提取失败: {str(e)}")
            return []

    @staticmethod
    def _frame_ocr_score(frame_bgr: np.ndarray) -> float:
        """为单帧估算 OCR 可读性，优先保留文字密度高、清晰的 PPT/公开课画面。"""
        if frame_bgr is None or frame_bgr.size == 0:
            return float("-inf")

        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if max(gray.shape[:2]) > 1280:
                scale = 1280.0 / float(max(gray.shape[:2]))
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            sharpness = float(cv2.Laplacian(blur, cv2.CV_64F).var())
            mean_intensity = float(np.mean(blur))

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blur)

            _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )

            def _binary_score(binary_img: np.ndarray) -> float:
                black_ratio = float(np.mean(binary_img == 0))
                # 文本区域通常占比不高，但也不能太低；公开课 PPT 往往在 8%~25% 之间更可读。
                density_score = max(0.0, 1.0 - abs(black_ratio - 0.16) / 0.16)

                inv = 255 - binary_img
                components = 0
                try:
                    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
                    for stat in stats[1:num_labels]:
                        area = int(stat[cv2.CC_STAT_AREA])
                        if 12 <= area <= 1800:
                            components += 1
                except Exception:
                    components = 0

                component_score = min(components / 60.0, 1.0)
                return density_score * 2.0 + component_score * 1.1

            score = max(_binary_score(otsu), _binary_score(adaptive))
            score += min(sharpness / 350.0, 2.0)

            # 过暗/过亮的页面通常更难识别
            if mean_intensity < 35 or mean_intensity > 235:
                score -= 0.8

            return score
        except Exception:
            return float("-inf")

    @classmethod
    def select_ocr_frames(cls, frames: List[np.ndarray], target_count: int = 6) -> List[np.ndarray]:
        """从候选帧中挑选更适合 OCR 的帧，并尽量保持时间分散。"""
        if not frames or target_count <= 0:
            return []

        target_count = min(target_count, len(frames))
        scored = [(idx, cls._frame_ocr_score(frame)) for idx, frame in enumerate(frames)]
        scored.sort(key=lambda item: item[1], reverse=True)

        # 先做时间分桶，确保整段视频有覆盖。
        bucket_count = min(target_count, max(3, len(frames) // 4))
        boundaries = np.linspace(0, len(frames), num=bucket_count + 1, dtype=int)
        selected_indices: List[int] = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            window = [item for item in scored if start <= item[0] < end]
            if not window:
                continue
            center = (start + end - 1) / 2.0
            best_idx, _ = max(window, key=lambda item: (item[1], -abs(item[0] - center)))
            if best_idx not in selected_indices:
                selected_indices.append(best_idx)

        # 再用全局高分帧补齐，并保持一定间隔，尽量避免抽到相邻页面。
        min_gap = max(1, len(frames) // max(target_count * 2, 1))
        for idx, _score in scored:
            if idx in selected_indices:
                continue
            if all(abs(idx - picked) >= min_gap for picked in selected_indices):
                selected_indices.append(idx)
            if len(selected_indices) >= target_count:
                break

        if len(selected_indices) < target_count:
            for idx, _score in scored:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                if len(selected_indices) >= target_count:
                    break

        selected_indices = sorted(selected_indices[:target_count])
        return [frames[idx] for idx in selected_indices]
    
    def extract_video_features(self, video_path: str) -> Optional[List[np.ndarray]]:
        """
        从视频帧提取特征
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            特征列表
        """
        frames = self.extract_frames(video_path)
        if not frames:
            return None
        
        features = []
        for frame in frames:
            feature = self._extract_frame_feature(frame)
            if feature is not None:
                features.append(feature)
        
        return features if features else None
    
    def get_video_metadata(self, video_path: str) -> Dict[str, any]:
        """
        获取视频元数据
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            元数据字典
        """
        try:
            cap = cv2.VideoCapture(video_path)
            metadata = {
                'path': video_path,
                'filename': os.path.basename(video_path),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            }
            
            cap.release()
            return metadata
        except Exception as e:
            logger.error(f"获取元数据失败: {str(e)}")
            return {}
    
    def load_videos(self, video_dir: str) -> Dict[str, Dict[str, any]]:
        """
        批量加载视频
        
        Args:
            video_dir: 视频目录
            
        Returns:
            视频数据字典
        """
        result = {}
        
        if not os.path.exists(video_dir):
            logger.warning(f"目录不存在: {video_dir}")
            return result
        
        for filename in os.listdir(video_dir):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                filepath = os.path.join(video_dir, filename)
                try:
                    result[filename] = {
                        'path': filepath,
                        'metadata': self.get_video_metadata(filepath),
                        'features': self.extract_video_features(filepath)
                    }
                    logger.info(f"已加载视频: {filename}")
                except Exception as e:
                    logger.error(f"加载失败 {filename}: {str(e)}")
        
        return result


class CaptionGenerator:
    """图像和视频描述生成器"""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        use_ocr: bool = True,
        ocr_lang: str = "chi_sim+eng",
        use_asr: bool = True,
        asr_model_size: str = "small",
        use_xmodaler_video: bool = True,
        xmodaler_model_type: str = "tdconved",
        use_tden_retrieval: bool = False,
        device_mode: str = "auto",
    ):
        """
        初始化描述生成器

        Args:
            model_name: HuggingFace模型名称
            use_ocr: 是否启用OCR识别（可选；默认开启，且不依赖 PaddlePaddle）
            ocr_lang: Tesseract OCR 语言包配置（默认中文+英文）
            use_asr: 是否启用音频ASR（可选，默认开启）
            asr_model_size: faster-whisper 模型尺寸（tiny/base/small/medium/large）
            use_xmodaler_video: 是否使用 xmodaler 专业视频字幕模型（默认开启，优先于BLIP）
            xmodaler_model_type: xmodaler 模型类型（tdconved/ta，默认 tdconved）
            use_tden_retrieval: 是否加载 TDEN 检索模型（默认关闭；当前主流程未直接使用）
        """
        self.model_name = model_name
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        self.use_asr = use_asr
        self.asr_model_size = asr_model_size
        self.use_xmodaler_video = use_xmodaler_video
        self.xmodaler_model_type = str(xmodaler_model_type or "tdconved").strip().lower()
        self.use_tden_retrieval = bool(use_tden_retrieval)
        self.device_mode = str(device_mode or "auto").strip().lower()
        self.model = None
        self.processor = None
        self.ocr_model = None
        self.ocr_backend = None
        self.asr_model = None
        self.asr_backend = None
        self.xmodaler_model = None
        self.xmodaler_config = None
        self.xmodaler_vocab = None
        self.tden_image_model = None
        self.tden_image_config = None
        self.tden_retrieval_model = None
        self.tden_retrieval_config = None
        self.video_processor = None
        self._video_processor_lock = threading.Lock()
        self.device = None
        self.xmodaler_load_error = ""
        self._load_model()
        if self.use_ocr:
            self._load_ocr_model()
        if self.use_asr:
            self._load_asr_model()
        if self.use_xmodaler_video:
            self._load_xmodaler_video_model()
        self._load_tden_image_caption_model()
        if self.use_tden_retrieval:
            self._load_tden_retrieval_model()

    def _load_model(self):
        """加载预训练的图像描述模型"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            from LOCAL_MODEL_MANAGER import LocalModelManager

            device = resolve_torch_device(self.device_mode)
            manager = LocalModelManager()
            local_blip = manager.get_model_path('blip')
            load_path = local_blip if local_blip else self.model_name
            if local_blip:
                logger.info(f"优先使用本地BLIP模型: {local_blip}")

            try:
                self.processor = BlipProcessor.from_pretrained(str(load_path), local_files_only=bool(local_blip))
                self.model = BlipForConditionalGeneration.from_pretrained(
                    str(load_path), local_files_only=bool(local_blip)
                ).to(device)
            except Exception as local_err:
                if local_blip:
                    # 本地目录损坏时回退到模型名加载（有网可自动恢复，无网则继续降级到其他分支）
                    logger.warning(f"本地BLIP加载失败，尝试远程模型ID回退: {local_err}")
                    self.processor = BlipProcessor.from_pretrained(str(self.model_name), local_files_only=False)
                    self.model = BlipForConditionalGeneration.from_pretrained(
                        str(self.model_name), local_files_only=False
                    ).to(device)
                    load_path = self.model_name
                else:
                    raise
            self.device = device
            logger.info(f"已加载模型: {load_path}")
        except Exception as e:
            logger.warning(f"无法加载图像描述模型: {str(e)}")

    def _load_xmodaler_video_model(self):
        """加载 xmodaler 视频字幕生成模型（TDConvED or TA）。"""
        try:
            import torch
            from xmodaler.config import get_cfg
            from xmodaler.modeling import build_model, add_config
            from xmodaler.checkpoint import XmodalerCheckpointer
            from LOCAL_MODEL_MANAGER import LocalModelManager

            device = resolve_torch_device(self.device_mode)
            manager = LocalModelManager()

            if self.xmodaler_model_type == "tdconved":
                model_key = "msrvtt_tdconved"
                config_candidates = [
                    "configs/video_caption/msrvtt/tdconved/tdconved.yaml",
                    "configs/video_captioning/msrvtt/tdconved/tdconved.yaml",
                    "configs/video_captioning/tdconved.yaml",
                ]
            elif self.xmodaler_model_type == "ta":
                model_key = "video_caption_msvd_ta"
                config_candidates = [
                    "configs/video_caption/msvd/ta/ta.yaml",
                    "configs/video_captioning/msvd/ta/ta.yaml",
                    "configs/video_captioning/ta.yaml",
                ]
            else:
                logger.warning(f"未知的 xmodaler 模型类型: {self.xmodaler_model_type}")
                return

            # 查找模型权重
            model_path = manager.find_local_model(model_key)
            if not model_path:
                logger.warning(f"未找到 xmodaler 模型权重: {model_key}")
                return

            # 查找配置文件
            project_root = Path(__file__).resolve().parents[2]
            config_path = None
            for rel_path in config_candidates:
                candidate = project_root / rel_path
                if candidate.exists():
                    config_path = candidate
                    break

            if not config_path:
                logger.warning(
                    "未找到 xmodaler 配置文件，已尝试: %s",
                    ", ".join(str(project_root / rel_path) for rel_path in config_candidates),
                )
                return

            # 加载配置
            cfg = get_cfg()
            # 先根据目标 yaml 注册模型专属配置节点（如 MODEL.TDCONVED）
            tmp_cfg = cfg.load_from_file_tmp(str(config_path))
            add_config(cfg, tmp_cfg)
            cfg.merge_from_file(str(config_path))
            # 推理场景下优先使用与模型权重匹配的词表。
            # 对于 msrvtt_tdconved（预训练权重来自 MSR-VTT），必须优先使用 repository 中的 msrvtt_dataset 词表（7001 words）。
            # 这样能确保 token embedding / predictor 与 checkpoint 形状对齐，避免加载成 2075 词表。
            vocab_candidates = []
            if model_key == "msrvtt_tdconved":
                vocab_candidates.append(project_root / "xmodaler" / "datasets" / "msrvtt_dataset" / "vocabulary.txt")
            vocab_candidates.extend([
                project_root / "data" / "annotations" / "vocabulary.txt",
                project_root / "configs" / "video_captioning" / "vocabulary.txt",
            ])

            selected_vocab = None
            for vocab_path in vocab_candidates:
                if vocab_path.exists():
                    selected_vocab = vocab_path
                    break

            if selected_vocab is not None:
                cfg.INFERENCE.VOCAB = str(selected_vocab)
                # 直接按选定词表长度设置模型词表大小，避免默认值 2075 覆盖。
                try:
                    vocab_list = self._load_vocab_file(str(selected_vocab))
                    if vocab_list is not None:
                        cfg.MODEL.VOCAB_SIZE = len(vocab_list)
                        logger.info(
                            "xmodaler 视频字幕使用词表: %s (VOCAB_SIZE=%s)",
                            selected_vocab,
                            cfg.MODEL.VOCAB_SIZE,
                        )
                except Exception as e:
                    logger.warning("读取词表失败，保留原始 VOCAB_SIZE=%s: %s", cfg.MODEL.VOCAB_SIZE, e)
            # 与当前运行设备保持一致，避免 CPU 环境被默认 CUDA 配置拉起
            cfg.MODEL.DEVICE = str(device)
            cfg.MODEL.WEIGHTS = str(model_path)
            cfg.freeze()

            # 构建模型
            model = build_model(cfg)
            checkpointer = XmodalerCheckpointer(model)
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
            model.eval()
            model = model.to(device)

            self.xmodaler_model = model
            self.xmodaler_config = cfg
            self.device = device
            self.xmodaler_load_error = ""
            logger.info(f"已加载 xmodaler 视频字幕模型: {model_key} from {model_path}")
        except ModuleNotFoundError as e:
            missing_module = getattr(e, "name", "") or str(e)
            install_hint = "请先安装 transformers（推荐），或安装兼容包 pytorch-transformers 以兼容旧代码。"
            logger.warning(
                "无法加载 xmodaler 视频字幕模型，缺少依赖: %s。%s",
                missing_module,
                install_hint,
            )
            self.xmodaler_model = None
            self.xmodaler_load_error = str(e)
        except Exception as e:
            logger.exception(f"无法加载 xmodaler 视频字幕模型: {str(e)}")
            self.xmodaler_model = None
            self.xmodaler_load_error = str(e)

    @staticmethod
    def _load_vocab_file(vocab_path: Optional[str]) -> Optional[List[str]]:
        if not vocab_path:
            return None
        path = Path(str(vocab_path))
        if not path.exists() or not path.is_file():
            return None
        vocab = ['.']
        try:
            # 与 xmodaler.functional.load_vocab 保持一致：保留空行占位，避免词表长度少 1。
            for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
                vocab.append(line.strip())
            return vocab
        except Exception:
            return None

    def _get_video_processor(self) -> VideoProcessor:
        if self.video_processor is None:
            with self._video_processor_lock:
                if self.video_processor is None:
                    self.video_processor = VideoProcessor(frames_per_video=50, device_mode=self.device_mode)
        return self.video_processor

    def _load_tden_image_caption_model(self):
        """加载 TDEN 图像字幕模型（替代 BLIP）。"""
        try:
            import torch
            from xmodaler.config import get_cfg
            from xmodaler.modeling import build_model, add_config
            from xmodaler.checkpoint import XmodalerCheckpointer
            from LOCAL_MODEL_MANAGER import LocalModelManager

            device = resolve_torch_device(self.device_mode)
            manager = LocalModelManager()

            # 查找 TDEN 模型权重
            model_key = "image_caption_cider_tden"  # CIDEr 优化的版本最优
            model_path = manager.find_local_model(model_key)

            if not model_path:
                logger.warning(f"未找到 TDEN 图像字幕模型权重: {model_key}")
                return

            # 查找配置文件
            project_root = Path(__file__).resolve().parents[2]
            config_path = project_root / "configs/pretrain/tden/tden.yaml"

            if not config_path.exists():
                logger.warning(f"未找到 TDEN 配置文件: {config_path}")
                return

            # 加载配置
            cfg = get_cfg()
            tmp_cfg = cfg.load_from_file_tmp(str(config_path))
            add_config(cfg, tmp_cfg)
            cfg.merge_from_file(str(config_path))
            cfg.MODEL.DEVICE = str(device)
            cfg.MODEL.WEIGHTS = str(model_path)
            cfg.freeze()

            # 构建模型
            model = build_model(cfg)
            checkpointer = XmodalerCheckpointer(model)
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
            model.eval()
            model = model.to(device)

            self.tden_image_model = model
            self.tden_image_config = cfg
            self.device = device
            logger.info(f"已加载 TDEN 图像字幕模型（CIDEr优化）from {model_path}")
        except Exception as e:
            logger.exception(f"无法加载 TDEN 图像字幕模型: {str(e)}")
            self.tden_image_model = None

    def _load_tden_retrieval_model(self):
        """加载 TDEN 检索模型（图像-文本多模态匹配）。"""
        try:
            import torch
            from xmodaler.config import get_cfg
            from xmodaler.modeling import build_model, add_config
            from xmodaler.checkpoint import XmodalerCheckpointer
            from LOCAL_MODEL_MANAGER import LocalModelManager

            device = resolve_torch_device(self.device_mode)
            manager = LocalModelManager()

            # 查找 TDEN 检索模型权重
            model_key = "image_retrieval_flickr_tden"
            model_path = manager.find_local_model(model_key)

            if not model_path:
                logger.warning(f"未找到 TDEN 检索模型权重: {model_key}")
                return

            # 查找配置文件
            project_root = Path(__file__).resolve().parents[2]
            config_path = project_root / "configs/pretrain/tden/tden.yaml"

            if not config_path.exists():
                logger.warning(f"未找到 TDEN 检索配置文件: {config_path}")
                return

            # 加载配置
            cfg = get_cfg()
            tmp_cfg = cfg.load_from_file_tmp(str(config_path))
            add_config(cfg, tmp_cfg)
            cfg.merge_from_file(str(config_path))
            cfg.MODEL.DEVICE = str(device)
            cfg.MODEL.WEIGHTS = str(model_path)
            cfg.freeze()

            # 构建模型
            model = build_model(cfg)
            checkpointer = XmodalerCheckpointer(model)
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
            model.eval()
            model = model.to(device)

            self.tden_retrieval_model = model
            self.tden_retrieval_config = cfg
            self.device = device
            logger.info(f"已加载 TDEN 检索模型（Flickr30K）from {model_path}")
        except Exception as e:
            logger.exception(f"无法加载 TDEN 检索模型: {str(e)}")
            self.tden_retrieval_model = None

    def _load_ocr_model(self):
        """加载可选OCR后端（非 Paddle 方案）。"""
        try:
            from shutil import which

            if which("tesseract") is None:
                logger.warning("未检测到 tesseract 可执行文件，OCR 将被禁用")
                self.ocr_model = None
                self.ocr_backend = None
                return

            import pytesseract

            self.ocr_model = pytesseract
            self.ocr_backend = "pytesseract"
            logger.info("已加载 pytesseract OCR 后端")
        except Exception as e:
            logger.warning(f"无法加载OCR后端: {str(e)}")
            self.ocr_model = None
            self.ocr_backend = None

    def _load_asr_model(self):
        """加载可选 ASR 后端（A方案：faster-whisper，缺失时自动降级）。"""
        try:
            from faster_whisper import WhisperModel

            try:
                import torch
                if self.device_mode == "cpu":
                    use_cuda = False
                elif self.device_mode == "cuda":
                    use_cuda = bool(torch.cuda.is_available())
                else:
                    use_cuda = bool(torch.cuda.is_available())
            except Exception:
                use_cuda = False

            device = "cuda" if use_cuda else "cpu"
            compute_type = "float16" if use_cuda else "int8"
            self.asr_model = WhisperModel(self.asr_model_size, device=device, compute_type=compute_type)
            self.asr_backend = "faster-whisper"
            logger.info(f"已加载 ASR 后端: {self.asr_backend} ({self.asr_model_size}, {device})")
        except Exception as e:
            logger.warning(f"无法加载 ASR 后端，将跳过语音转写: {str(e)}")
            self.asr_model = None
            self.asr_backend = None

    def transcribe_video_audio(self, video_path: str) -> Optional[str]:
        """将视频音频转写为文本（教学讲解语音）。"""
        if not self.asr_model:
            return None

        audio_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                audio_path = tmp_file.name

            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-ac', '1', '-ar', '16000',
                audio_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"音频提取失败，跳过ASR: {result.stderr[:200]}")
                return None

            segments, _ = self.asr_model.transcribe(
                audio_path,
                language='zh',
                beam_size=3,
                vad_filter=True,
            )
            texts = [normalize_zh_text(seg.text) for seg in segments if normalize_zh_text(seg.text)]
            if not texts:
                return None

            transcript = normalize_zh_text(' '.join(texts))
            logger.info(f"ASR转写文本: {transcript[:100]}...")
            return transcript
        except Exception as e:
            logger.warning(f"ASR转写失败 {video_path}: {str(e)}")
            return None
        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

    def recognize_text_from_image(self, image_path: str) -> Optional[str]:
        """
        从图像中识别文字（用于教学课件；非 Paddle 后端可选）

        Args:
            image_path: 图像文件路径

        Returns:
            识别出的文字
        """
        if not self.ocr_model:
            return None

        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            frame_np = np.array(image)
            prepared_images = self._prepare_ocr_images_for_ppt(frame_np)

            best_text = None
            best_score = float("-inf")
            # psm=6 对段落文本稳定，11 对稀疏文本更稳，4 对多列课件更友好
            for prepared in prepared_images:
                for config in ("--oem 3 --psm 6", "--oem 3 --psm 11", "--oem 3 --psm 4"):
                    raw_text = self.ocr_model.image_to_string(
                        prepared,
                        lang=self.ocr_lang,
                        config=config,
                    )
                    candidate = normalize_zh_text(raw_text)
                    if not candidate:
                        continue
                    score = self._score_ocr_text(candidate)
                    if score > best_score:
                        best_score = score
                        best_text = candidate

            if best_text:
                # 中文教学场景下过滤典型乱码，避免污染实体匹配。
                if self._is_likely_garbled_ocr(best_text):
                    logger.info("OCR结果疑似乱码，已忽略")
                    return None
                logger.info(f"OCR识别文字: {best_text[:100]}...")
                return best_text
            return None
        except Exception as e:
            logger.error(f"OCR识别失败 {image_path}: {str(e)}")
            return None

    @staticmethod
    def _prepare_ocr_images_for_ppt(frame_np: np.ndarray) -> List["Image.Image"]:
        """针对PPT页面生成多份OCR输入：灰度、放大、去噪、二值化（优化版）。"""
        from PIL import Image

        gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
        # 小字号课件普遍存在压缩模糊，2x 放大可显著提升识别稳定性。
        upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # 更强的去噪：先进行双边滤波（保留边界），再中值滤波
        bilateral = cv2.bilateralFilter(upscaled, 9, 75, 75)
        denoise = cv2.medianBlur(bilateral, 5)

        # 对比度增强：CLAHE（自适应直方图均衡化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoise)

        # 多种二值化方式以适应不同的PPT背景
        _, binary_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_adaptive = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            11,
        )
        # 三值化：对灰色文字有更好的保留
        _, binary_tri = cv2.threshold(denoise, 127, 255, cv2.THRESH_BINARY)

        return [
            Image.fromarray(binary_otsu),      # Otsu 自适应阈值
            Image.fromarray(binary_adaptive),  # 局部自适应阈值（最稳定）
            Image.fromarray(enhanced),         # 增强但未二值化（保留灰度信息）
            Image.fromarray(binary_tri),       # 三值化（适合灰色文字）
        ]

    @staticmethod
    def _contains_chinese(text: Optional[str]) -> bool:
        if not text:
            return False
        return any('\u4e00' <= ch <= '\u9fff' for ch in text)

    def _score_ocr_text(self, text: str) -> float:
        """为OCR候选打分，优先中文准确性。"""
        normalized = normalize_zh_text(text)
        if not normalized:
            return float("-inf")

        total_len = max(len(normalized), 1)
        zh_count = sum(1 for ch in normalized if '\u4e00' <= ch <= '\u9fff')
        digit_count = sum(1 for ch in normalized if ch.isdigit())
        ascii_alpha_count = sum(1 for ch in normalized if ch.isascii() and ch.isalpha())
        punctuation_count = sum(1 for ch in normalized if ch in '，。！？；：（）')
        tokens = [token for token in normalized.split(" ") if token]
        short_ascii_tokens = [token for token in tokens if token.isascii() and token.isalpha() and len(token) <= 3]
        long_ascii_tokens = [token for token in tokens if token.isascii() and token.isalpha() and len(token) >= 5]

        chinese_ratio = zh_count / total_len
        ascii_ratio = ascii_alpha_count / total_len

        # 计分策略：
        # - 中文字符权重最高（3.0）：教学课件主要信息
        # - 数字和标点符号有益（教学内容常含数字）
        # - 纯ASCII碎片词为噪声，大幅降权
        score = (zh_count * 4.0)
        score += (digit_count * 0.18)
        score += (punctuation_count * 0.16)
        score += (min(total_len, 240) * 0.02)

        if zh_count >= 4:
            score += min(chinese_ratio * 12.0, 8.0)

        if zh_count > 0 and ascii_alpha_count > 0:
            score += min(zh_count, 10) * 0.3

        # 惩罚纯ASCII垃圾
        score -= ascii_alpha_count * 0.18
        score -= len(short_ascii_tokens) * 0.7
        score -= len(long_ascii_tokens) * 0.25

        # 英文/数字占比过高时，进一步降权。
        if zh_count == 0 and ascii_alpha_count >= 8:
            score -= 12.0
        elif chinese_ratio < 0.08 and (ascii_ratio > 0.30 or digit_count > 8):
            score -= 5.0

        # 若几乎全是ASCII字母（乱码特征），大幅降低
        if zh_count == 0 and ascii_alpha_count > 20:
            score -= 15.0

        return score

    def _is_likely_garbled_ocr(self, text: str) -> bool:
        normalized = normalize_zh_text(text)
        if not normalized:
            return True

        zh_count = sum(1 for ch in normalized if '\u4e00' <= ch <= '\u9fff')
        total_len = max(len(normalized), 1)
        chinese_ratio = zh_count / total_len
        if zh_count >= 4 and chinese_ratio >= 0.08:
            return False

        ascii_alpha_count = sum(1 for ch in normalized if ch.isascii() and ch.isalpha())
        token_list = [token for token in normalized.split(" ") if token]
        token_count = len(token_list)
        short_ascii_tokens = len([
            token for token in normalized.split(" ")
            if token.isascii() and token.isalpha() and len(token) <= 3
        ])
        digit_count = sum(1 for ch in normalized if ch.isdigit())
        symbol_count = sum(1 for ch in normalized if ch in "<>{}[]()=+-*/\\|_~`^@#$%^&")

        # 典型乱码特征：大量短英文碎片词，且不含中文。
        if ascii_alpha_count >= 20 and token_count >= 6 and short_ascii_tokens / max(token_count, 1) >= 0.45:
            return True

        # 终端/代码页/网页控件片段，通常缺少中文且符号占比很高。
        if zh_count < 2 and (ascii_alpha_count + digit_count) >= 18 and symbol_count >= 4:
            return True

        # 英文主导且短 token 很多，通常是噪声而不是可读字幕。
        if zh_count < 2 and ascii_alpha_count >= 12 and token_count >= 5 and short_ascii_tokens >= 3:
            return True
        return False

    @staticmethod
    def _is_useful_video_ocr(text: Optional[str]) -> bool:
        """判断视频 OCR 是否值得拼接到最终字幕中，避免长串噪声污染结果。"""
        normalized = normalize_zh_text(text or "")
        if not normalized:
            return False

        zh_count = sum(1 for ch in normalized if '\u4e00' <= ch <= '\u9fff')
        ascii_alpha_count = sum(1 for ch in normalized if ch.isascii() and ch.isalpha())
        digit_count = sum(1 for ch in normalized if ch.isdigit())
        total_len = max(len(normalized), 1)

        # 至少要有一些中文主体；如果几乎全是英文/数字碎片，则不要拼接到最终字幕。
        if zh_count < 4 and ascii_alpha_count > 12:
            return False
        # 太长且中文很少，通常是终端/代码页/网页内容噪声。
        if total_len > 120 and zh_count < 12:
            return False
        # 中文占比太低，也认为不值得拼接。
        if zh_count / total_len < 0.12 and digit_count > 10:
            return False
        return True

    @staticmethod
    def _is_blip_caption_low_quality(caption: str) -> bool:
        """
        检测BLIP描述是否质量低（常见乱码或过度通用）
        
        Args:
            caption: BLIP生成的描述文本
            
        Returns:
            True 表示质量低，False 表示质量可接受
        """
        if not caption:
            return True
        
        text = caption.lower().strip()
        
        # 长度过短
        if len(text) < 10:
            return True
        
        # 常见乱码模式（已从日志中观察）
        garbled_patterns = [
            "sh re erarmi",
            "tear agar",
            "bra ee",
            "are ratan",
            "feta gor",
            "ere eer"
        ]
        for pattern in garbled_patterns:
            if pattern in text:
                return True
        
        # 大量单个字母分散排列
        single_chars = sum(1 for char in text if char.isalpha() and text.count(char) == 1)
        if len(text) > 20 and single_chars / len(text) > 0.4:
            return True
        
        # 无意义的重复单字母（如"e e e e"）
        import re
        repeated_singles = re.findall(r'(\b\w\b\s+)+', text)
        if len(repeated_singles) > 5:
            return True
        
        return False

    def generate_image_caption(self, image_path: str) -> Optional[str]:
        """
        为图像生成描述（优化版）

        Args:
            image_path: 图像文件路径
            
        Returns:
            图像描述
        """
        # 中文教学场景：OCR是主要信息源，英文模型仅作补充
        ocr_text = None
        blip_caption = None

        # 优先处理OCR，这是中文场景的核心
        if self.use_ocr:
            ocr_text = self.recognize_text_from_image(image_path)
            if ocr_text and len(ocr_text.strip()) >= 8:
                logger.info(f"[{Path(image_path).name}] OCR识别成功，主要文字: {ocr_text[:80]}...")
                # OCR优先，直接返回带标记的文字
                return normalize_zh_text(f"教学课件，主要文字：{ocr_text}")

        # 若OCR未获得有效文字，尝试BLIP描述
        if not self.model or not self.processor:
            if ocr_text:
                return normalize_zh_text(f"教学课件，文字：{ocr_text}")
            return None
        
        try:
            from PIL import Image
            import torch
            
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # 优化参数：增加max_length，使用beam search，过滤短结果
                out = self.model.generate(
                    **inputs,
                    max_length=100,
                    min_length=15,
                    num_beams=3,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    temperature=0.7
                )

            blip_caption = normalize_zh_text(self.processor.decode(out[0], skip_special_tokens=True))

            # 过滤质量差的BLIP描述（常见乱码或通用短语）
            if blip_caption and self._is_blip_caption_low_quality(blip_caption):
                logger.info(f"[{Path(image_path).name}] BLIP描述质量低，已忽略")
                if ocr_text:
                    return normalize_zh_text(f"教学课件，文字：{ocr_text}")
                return None

            # 融合结果
            if ocr_text and blip_caption:
                return normalize_zh_text(f"教学课件，主要文字：{ocr_text}。图像信息（英文）：{blip_caption}")
            elif ocr_text:
                return normalize_zh_text(f"教学课件，文字：{ocr_text}")
            elif blip_caption:
                return normalize_zh_text(f"图像：{blip_caption}")

            return None
        except Exception as e:
            logger.error(f"描述生成失败 {image_path}: {str(e)}")
            if ocr_text:
                return normalize_zh_text(f"教学课件，文字：{ocr_text}")
            return None

    def generate_video_caption_xmodaler(self, video_path: str) -> Optional[Dict[str, Optional[str]]]:
        """
        用 xmodaler 视频字幕模型生成描述（TDConvED 或 TA）。
        返回结构化结果：{xmodaler_caption, blip_fallback, ocr_text, asr_text}

        Args:
            video_path: 视频文件路径

        Returns:
            描述字典或 None
        """
        if not self.xmodaler_model:
            return None

        try:
            import torch
            from xmodaler.functional import decode_sequence
            from xmodaler.config import kfg

            # 从视频抽取特征（按需创建并复用，避免重复初始化图像主干）
            video_processor = self._get_video_processor()
            frames = video_processor.extract_frames(video_path)
            if not frames:
                return None

            # 提取特征（用 ResNet）
            features = []
            for frame in frames:
                feat = video_processor._extract_frame_feature(frame)
                if feat is not None:
                    features.append(feat)

            if not features:
                return None

            # 补齐到 50 帧（xmodaler 标准）
            while len(features) < 50:
                features.append(features[-1] if features else np.zeros(2048))
            features = features[:50]

            feat_array = np.array(features, dtype=np.float32)  # (50, 2048)
            feat_tensor = torch.from_numpy(feat_array).float().unsqueeze(0)  # (1, 50, 2048)
            att_masks = torch.ones((1, feat_tensor.size(1)), dtype=torch.float32)
            max_seq_len = int(getattr(getattr(self.xmodaler_config, 'MODEL', None), 'MAX_SEQ_LEN', 20))
            g_tokens_type = torch.ones((1, max_seq_len), dtype=torch.long)

            # xmodaler 的 beam search 入口期望的是单个 batch 字典，而不是 list[dict]
            batched_inputs = {
                kfg.ATT_FEATS: feat_tensor.to(self.device),
                kfg.ATT_MASKS: att_masks.to(self.device),
                kfg.G_TOKENS_TYPE: g_tokens_type.to(self.device),
                kfg.IDS: ['video'],
            }

            # 推理
            with torch.no_grad():
                outputs = self.xmodaler_model(batched_inputs, use_beam_search=True, output_sents=True)
                output_key = getattr(kfg, 'OUTPUT', 'OUTPUT')
                output_value = None
                if isinstance(outputs, dict):
                    output_value = outputs.get(output_key)
                    if output_value is None:
                        output_value = outputs.get(kfg.G_SENTS_IDS)
                    if output_value is None:
                        output_value = outputs.get('output')

                xmodaler_caption = None
                if isinstance(output_value, (list, tuple)) and len(output_value) > 0:
                    first_item = output_value[0]
                    if isinstance(first_item, str):
                        xmodaler_caption = first_item
                    elif torch.is_tensor(first_item):
                        vocab = self._load_vocab_file(getattr(getattr(self.xmodaler_config, 'INFERENCE', None), 'VOCAB', ''))
                        if vocab is not None:
                            try:
                                decoded = decode_sequence(vocab, first_item)
                                if decoded:
                                    xmodaler_caption = decoded[0]
                            except Exception:
                                pass
                elif torch.is_tensor(output_value):
                    vocab = self._load_vocab_file(getattr(getattr(self.xmodaler_config, 'INFERENCE', None), 'VOCAB', ''))
                    if vocab is not None:
                        try:
                            decoded = decode_sequence(vocab, output_value)
                            if decoded:
                                xmodaler_caption = decoded[0]
                        except Exception:
                            pass

            # 补充其他文本来源
            ocr_text = None
            asr_text = None
            blip_fallback = None

            # 抽样帧做 OCR
            if self.use_ocr:
                ocr_frames = VideoProcessor.select_ocr_frames(frames, target_count=min(len(frames), 8))
                logger.info(f"视频OCR选帧: {len(ocr_frames)}/{len(frames)} 帧")
                ocr_texts = []
                for frame in ocr_frames:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as frame_file:
                        frame_path = frame_file.name
                    try:
                        cv2.imwrite(frame_path, frame)
                        text = self.recognize_text_from_image(frame_path)
                        if text:
                            ocr_texts.append(text)
                    finally:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                if ocr_texts:
                    ocr_text = self._aggregate_ocr_texts(ocr_texts)

            # ASR
            if self.use_asr:
                asr_text = self.transcribe_video_audio(video_path)


            return {
                'xmodaler_caption': xmodaler_caption,
                'blip_fallback': blip_fallback,
                'ocr_text': ocr_text,
                'asr_text': asr_text,
            }
        except Exception as e:
            logger.warning(f"xmodaler 视频字幕生成失败: {str(e)}")
            return None

    def generate_video_caption(self, video_path: str) -> Optional[str]:
        """
        为视频生成描述（优先 xmodaler，回退 BLIP+OCR+ASR）

        Args:
            video_path: 视频文件路径

        Returns:
            视频描述
        """
        try:
            # 方案 A：优先使用 xmodaler 视频字幕模型
            if self.use_xmodaler_video and self.xmodaler_model:
                result = self.generate_video_caption_xmodaler(video_path)
                if result and result.get('xmodaler_caption'):
                    caption = self._compose_video_caption_parts(
                        result.get('ocr_text'),
                        result.get('asr_text'),
                        result['xmodaler_caption'],
                        fallback_visual_label="视觉补充",
                    )
                    if caption:
                        video_processor = self._get_video_processor()
                        video_meta = video_processor.get_video_metadata(video_path)
                        meta_hint = []
                        if video_meta.get('fps'):
                            meta_hint.append(f"{video_meta['fps']:.1f}fps")
                        if video_meta.get('frame_count'):
                            meta_hint.append(f"{video_meta['frame_count']}帧")
                        if meta_hint:
                            caption = f"{caption} | {' '.join(meta_hint)}"
                        logger.info(f"使用 xmodaler 生成视频描述: {caption[:100]}...")
                        return caption
            elif self.use_xmodaler_video and not self.xmodaler_model:
                logger.warning(f"xmodaler 模型未加载，原因: {self.xmodaler_load_error or 'unknown'}")

            # 方案 B：回退到 BLIP+OCR+ASR
            logger.info("xmodaler 不可用或生成失败，改用 BLIP+OCR+ASR")
            video_processor = self._get_video_processor()
            frames = video_processor.extract_frames(video_path)

            if not frames:
                return None

            # 用中间帧走 BLIP，用多帧聚合 OCR（更适合教学 PPT 视频）
            middle_frame_idx = len(frames) // 2
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                temp_path = tmp_file.name
            cv2.imwrite(temp_path, frames[middle_frame_idx])

            ocr_texts: List[str] = []
            # 公开课/PPT 场景：优先从整段视频中筛选文字密度更高的帧，再做 OCR
            ocr_frames = VideoProcessor.select_ocr_frames(frames, target_count=min(len(frames), 10))

            try:
                blip_caption = self.generate_image_caption(temp_path)
                if self.use_ocr:
                    logger.info(f"视频OCR选帧: {len(ocr_frames)}/{len(frames)} 帧")
                    for frame in ocr_frames:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as frame_file:
                            frame_path = frame_file.name
                        try:
                            cv2.imwrite(frame_path, frame)
                            text = self.recognize_text_from_image(frame_path)
                            if text:
                                ocr_texts.append(text)
                        finally:
                            if os.path.exists(frame_path):
                                os.remove(frame_path)

                ocr_text = self._aggregate_ocr_texts(ocr_texts) if ocr_texts else None
                asr_text = self.transcribe_video_audio(video_path) if self.use_asr else None
                caption = self._fusion_captions(blip_caption, ocr_text, asr_text)
                if caption:
                    video_meta = video_processor.get_video_metadata(video_path)
                    meta_hint = []
                    if video_meta.get('fps'):
                        meta_hint.append(f"{video_meta['fps']:.1f}fps")
                    if video_meta.get('frame_count'):
                        meta_hint.append(f"{video_meta['frame_count']}帧")
                    if meta_hint:
                        caption = f"{caption} | {' '.join(meta_hint)}"
                return normalize_zh_text(caption) or None
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            logger.error(f"视频描述生成失败: {str(e)}")
            return None

    def _aggregate_ocr_texts(self, texts: List[str]) -> Optional[str]:
        """聚合多帧 OCR 结果，去重去噪，优先保留高频文本。"""
        if not texts:
            return None
        
        # 规范化并过滤乱码
        cleaned = []
        for text in texts:
            normalized = normalize_zh_text(text)
            if normalized and not self._is_likely_garbled_ocr(normalized):
                cleaned.append(normalized)
        
        if not cleaned:
            return None

        from collections import Counter

        counter = Counter(cleaned)
        # 基于频率和长度排序，优先选择稳定且完整的文本
        ranked = sorted(
            counter.items(),
            key=lambda item: (-item[1], -sum(1 for ch in item[0] if '\u4e00' <= ch <= '\u9fff'), -len(item[0]))
        )

        selected: List[str] = []
        for candidate, _count in ranked:
            if any(candidate == item or candidate in item or item in candidate for item in selected):
                continue
            selected.append(candidate)
            if len(selected) >= 2:
                break

        if not selected:
            return None

        # 优先保留更完整的主体文本；若存在第二段明显不同且可读的文本，则轻量拼接。
        best_text = selected[0]
        if len(selected) >= 2:
            merged = f"{selected[0]}；{selected[1]}"
            if len(merged) <= 140:
                best_text = merged

        logger.info(f"聚合 {len(texts)} 帧OCR结果，{len(cleaned)} 个有效，最优: {best_text[:100] if best_text else 'None'}...")
        return normalize_zh_text(best_text) if best_text else None

    def _compose_video_caption_parts(
        self,
        ocr_text: Optional[str],
        asr_text: Optional[str],
        visual_text: Optional[str],
        *,
        fallback_visual_label: str = "视觉补充",
    ) -> Optional[str]:
        """将视频多模态结果统一拼成中文教学场景的三段式描述。"""
        parts: List[str] = []

        if ocr_text and len(ocr_text.strip()) >= 10 and self._is_useful_video_ocr(ocr_text):
            parts.append(f"课件：{normalize_zh_text(ocr_text)}")

        if asr_text and len(asr_text.strip()) >= 15:
            parts.append(f"讲解：{normalize_zh_text(asr_text[:200])}")

        visual_text = normalize_zh_text(visual_text) if visual_text else ""
        if visual_text:
            if not parts:
                parts.append(f"图像描述：{visual_text}")
            elif self._contains_chinese(visual_text):
                parts.append(f"{fallback_visual_label}：{visual_text}")
            else:
                parts.append(f"视觉参考：{visual_text}")

        if not parts:
            return None

        return normalize_zh_text(' | '.join(parts))

    def _fusion_captions(self,
                         blip_caption: Optional[str],
                         ocr_text: Optional[str],
                         asr_text: Optional[str] = None) -> Optional[str]:
        """
        融合视频多模态描述

        优先级：OCR（视觉文字）> ASR（讲解语音）> BLIP（英文描述）

        Args:
            blip_caption: BLIP生成的英文描述（可能含乱码，低优先级）
            ocr_text: OCR识别的中文文字（教学课件核心）
            asr_text: ASR识别的讲解语音文本（教学讲解核心）

        Returns:
            融合后的描述
        """
        if not blip_caption and not ocr_text and not asr_text:
            return None

        result = self._compose_video_caption_parts(
            ocr_text,
            asr_text,
            blip_caption,
            fallback_visual_label="视觉补充",
        )
        if not result:
            return None
        logger.info(f"融合多模态描述: {result[:120]}...")
        return result


class VideoEditor:
    """视频编辑器"""
    
    def __init__(self):
        """初始化视频编辑器"""
        pass
    
    def clip_video(self, input_path: str, output_path: str, start_time: float = 0, duration: float = 60):
        """
        剪辑视频片段
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            start_time: 开始时间（秒）
            duration: 持续时间（秒）
        """
        try:
            import subprocess
            
            # 使用ffmpeg剪辑视频
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-y',  # 覆盖输出文件
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"视频剪辑成功: {output_path}")
                return True
            else:
                logger.error(f"视频剪辑失败: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"视频剪辑异常: {str(e)}")
            return False
    
    def get_video_duration(self, video_path: str) -> float:
        """
        获取视频时长
        
        Args:
            video_path: 视频路径
            
        Returns:
            时长（秒）
        """
        try:
            import subprocess
            import json
            
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                duration = float(data['format']['duration'])
                return duration
            else:
                logger.error(f"获取视频时长失败: {result.stderr}")
                return 0
        except Exception as e:
            logger.error(f"获取视频时长异常: {str(e)}")
            return 0

