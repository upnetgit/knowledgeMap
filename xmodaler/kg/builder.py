#!/usr/bin/env python3
"""
改进的知识图谱构建器
支持文本、图像、视频的多模态处理
"""

from __future__ import annotations

import os
import json
import logging
import re
import importlib
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    import spacy
except ImportError:
    spacy = None

try:
    _neo4j_module = importlib.import_module("neo4j")
    GraphDatabase = _neo4j_module.GraphDatabase
except Exception:
    GraphDatabase = None
import networkx as nx

from .processors import TextProcessor, ImageProcessor, VideoProcessor, CaptionGenerator
from .semantic import SemanticScorer, RelationReranker, summarize_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KGBuilder:
    """知识图谱构建器"""
    
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 database: str = "neo4j",
                 language: str = "zh",
                 device_mode: str = "auto",
                 relation_model_dir: Optional[str] = None,
                 relation_threshold: float = 0.55,
                 enable_relation_rerank: bool = True,
                 use_xmodaler_video: bool = True,
                 xmodaler_model_type: str = "tdconved",
                 enable_selective_node_expansion: bool = False,
                 selective_expansion_terms: Optional[List[str]] = None,
                 selective_expansion_computer_terms: Optional[List[str]] = None,
                 selective_expansion_ideology_terms: Optional[List[str]] = None,
                 selective_expansion_source_scope: str = "both",
                 selective_expansion_min_support: int = 1,
                 selective_expansion_min_score: float = 0.55,
                 selective_expansion_max_new_total: int = 20,
                 selective_expansion_max_new_computer: int = 20,
                 selective_expansion_max_new_ideology: int = 20):
        """
        初始化知识图谱构建器

        Args:
            neo4j_uri: Neo4j数据库URI
            user: 数据库用户名
            password: 数据库密码
            database: Neo4j数据库名（Aura必需）
            language: 语言（zh或en）
            use_xmodaler_video: 是否使用 xmodaler 专业视频字幕模型（默认开启）
            xmodaler_model_type: xmodaler 模型类型（tdconved/ta，默认 tdconved）
        """
        self.neo4j_uri = neo4j_uri
        self.user = user
        self.password = password
        self.database = database or "neo4j"
        self.language = language
        self.device_mode = str(device_mode or "auto").strip().lower()
        
        # 初始化处理器
        try:
            from LOCAL_MODEL_MANAGER import LocalModelManager

            model_manager = LocalModelManager(project_root=str(Path(__file__).resolve().parents[2]))
            preferred_model = model_manager.get_preferred_spacy_model(language=language, auto_install=True)
            if preferred_model:
                if language == 'zh' and preferred_model != 'zh_core_web_sm':
                    logger.warning("中文模式下未找到可用中文spaCy模型，改用空白中文模型")
                    nlp = spacy.blank('zh') if spacy is not None else None
                else:
                    logger.info(f"使用spaCy模型: {preferred_model}")
                    nlp = spacy.load(preferred_model)
            elif spacy is not None:
                fallback_lang = 'zh' if language == 'zh' else 'en'
                logger.warning(f"未找到可用的spaCy模型，使用空白模型: {fallback_lang}")
                nlp = spacy.blank(fallback_lang)
            else:
                logger.warning("spaCy不可用，文本实体抽取将降级")
                nlp = None
        except Exception as e:
            logger.warning(f"无法加载spaCy模型，文本实体抽取将降级: {e}")
            nlp = None
        
        self.text_processor = TextProcessor(nlp_model=nlp)
        self.image_processor = ImageProcessor(device_mode=self.device_mode)
        self.video_processor = VideoProcessor(device_mode=self.device_mode)
        self.caption_generator = CaptionGenerator(
            use_ocr=True, 
            ocr_lang='chi_sim+eng', 
            use_asr=True,
            use_xmodaler_video=use_xmodaler_video,
            xmodaler_model_type=xmodaler_model_type,
            device_mode=self.device_mode,
        )
        self.semantic_scorer = SemanticScorer(device_mode=self.device_mode)
        if self.semantic_scorer.available:
            logger.info(f"语义模型已启用: {self.semantic_scorer.model_dir}")
        else:
            logger.warning("语义模型未启用，已降级为词面匹配（请检查 BERT_cn/bert-base-chinese）")
        self.enable_relation_rerank = bool(enable_relation_rerank)
        self.relation_threshold = max(0.0, min(1.0, float(relation_threshold)))
        self.relation_reranker = RelationReranker(
            model_dir=relation_model_dir,
            threshold=self.relation_threshold,
            device_mode=self.device_mode,
        ) if self.enable_relation_rerank else None
        if self.relation_reranker is not None:
            if self.relation_reranker.available:
                logger.info(f"关系重排模型已启用: {self.relation_reranker.model_dir}")
            else:
                logger.warning("关系重排模型未启用，已回退规则策略（请检查 relation_model_dir）")

        # 有选择的节点扩展：默认关闭，避免图谱无节制增长
        self.enable_selective_node_expansion = bool(enable_selective_node_expansion)
        self.selective_expansion_catalog = {
            'computer_science': self._normalize_term_list(selective_expansion_computer_terms),
            'ideology': self._normalize_term_list(selective_expansion_ideology_terms),
        }
        legacy_terms = self._normalize_term_list(selective_expansion_terms)
        if legacy_terms and not self.selective_expansion_catalog['ideology']:
            # 兼容旧参数：未显式分类时，沿用为思政扩展候选
            self.selective_expansion_catalog['ideology'] = legacy_terms
        self.selective_expansion_terms = legacy_terms  # 保留兼容字段，外部日志/调试可见
        self.selective_expansion_source_scope = str(selective_expansion_source_scope or 'both').strip().lower()
        self.selective_expansion_min_support = max(1, int(selective_expansion_min_support))
        self.selective_expansion_min_score = max(0.0, min(1.0, float(selective_expansion_min_score)))
        self.selective_expansion_max_new_total = max(0, int(selective_expansion_max_new_total))
        self.selective_expansion_max_new_computer = max(0, int(selective_expansion_max_new_computer))
        self.selective_expansion_max_new_ideology = max(0, int(selective_expansion_max_new_ideology))
        self._selective_expanded_entities = set()
        self._selective_expanded_counts = defaultdict(int)

        # 记录使用的图像字幕模型
        self.image_caption_model = "tden" if self.caption_generator.tden_image_model else "blip"
        logger.info(f"使用图像字幕模型: {self.image_caption_model}")
        
        # 检查检索模型可用性
        self.use_retrieval = bool(self.caption_generator.tden_retrieval_model)
        if self.use_retrieval:
            logger.info("✅ TDEN 检索模型已启用，将增强多模态链接")
        
        # 知识图谱数据结构
        self.graph = nx.DiGraph()  # 有向图
        self.entity_metadata = {}
        self.relations = defaultdict(list)
        self.case_records: Dict[str, Dict] = {}
        self.media_records: Dict[str, Dict] = {}
        self.invalid_media_records: List[Dict] = []
        
        # 预定义实体类型
        self.entity_types = {
            'computer_science': [],  # 计算机核心知识点
            'ideology': []  # 思政元素
        }

        # 连接Neo4j
        if GraphDatabase is None:
            logger.warning("neo4j驱动不可用，图谱将仅构建内存与JSON结果")
            self.driver = None
        else:
            try:
                self.driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))
                self._test_connection()
                logger.info("已连接到Neo4j数据库")
            except Exception as e:
                logger.error(f"连接Neo4j失败: {str(e)}")
                self.driver = None

    def _normalize_relation_type(self, relation_type: str) -> str:
        """将关系类型规范为Neo4j兼容的上层语义标签。"""
        relation_type = (relation_type or 'CONNECTED').strip()
        relation_type = re.sub(r'[^0-9A-Za-z_]+', '_', relation_type)
        relation_type = relation_type.upper().strip('_')
        return relation_type or 'CONNECTED'

    def _node_labels(self, node_name: str, attrs: Dict) -> List[str]:
        """
        获取节点的Neo4j标签列表
        优先使用存储在attrs中的labels，否则根据type推导
        """
        if 'labels' in attrs and attrs['labels']:
            labels = attrs['labels']
            if isinstance(labels, list):
                return labels
            elif isinstance(labels, str):
                return [labels]

        # 根据type推导标签
        node_type = attrs.get('type', attrs.get('node_type', 'entity'))
        base_label = 'Entity'

        type_to_labels = {
            'computer_science': ['Entity', 'KnowledgePoint'],
            'ideology': ['Entity', 'IdeologyElement'],
            'teaching_case': ['Entity', 'TeachingCase'],
            'media_image': ['Entity', 'Media', 'Image'],
            'media_video': ['Entity', 'Media', 'Video'],
        }

        return type_to_labels.get(str(node_type).strip(), [base_label])

    # ...existing code...

    def _relation_score(self, relation_type: str) -> float:
        relation_type = self._normalize_relation_type(relation_type)
        return {
            'SIMILAR': 0.95,
            'RELATED': 0.85,
            'MENTIONS': 0.9,
            'LINKS_TO_CASE': 0.8,
            'MEDIA_LINKED_IMAGE': 0.75,
            'MEDIA_LINKED_VIDEO': 0.75,
            'CONNECTED_BY_IMAGE': 0.75,
            'CONNECTED_BY_VIDEO': 0.75,
            'CASE_SUPPORTS': 0.85,
            'COMPUTER_REFLECTS_IDEOLOGY': 0.9,
        }.get(relation_type, 0.6)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        text = str(text or "")
        if not text.strip():
            return []
        chunks = re.split(r"[。！？!?\n\r]+", text)
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    def _load_relation_priors(self) -> Dict[Tuple[str, str], float]:
        """从 BERT_cn/实体抽取 的样本中读取先验关系强度。"""
        prior_file = Path(__file__).resolve().parents[2] / 'BERT_cn' / '实体抽取' / 'computer_ideology_data.txt'
        if not prior_file.exists():
            return {}

        positive = defaultdict(int)
        total = defaultdict(int)
        try:
            with prior_file.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    subject = str(data.get('computer_core') or data.get('subject') or '').strip()
                    if not subject:
                        continue
                    labels = data.get('ideology_labels') or []
                    relation = str(data.get('relation') or data.get('label') or '体现').strip()
                    is_positive = relation in {'体现', '相关', '正向'} or str(data.get('label', '')).strip() == '体现'
                    for ideology in labels:
                        ideology_name = str(ideology).strip()
                        if not ideology_name:
                            continue
                        key = (subject, ideology_name)
                        total[key] += 1
                        if is_positive:
                            positive[key] += 1
        except Exception:
            return {}

        priors = {}
        for key, cnt in total.items():
            priors[key] = positive.get(key, 0) / max(cnt, 1)
        return priors

    def _best_support_sentence(self, computer: str, ideology: str) -> Tuple[str, float]:
        """从案例文本中提取可解释证据，返回 (证据句, 规则分)。"""
        best_sentence = ''
        best_score = 0.0
        for case in self.case_records.values():
            raw_text = str(case.get('raw_text', '') or '')
            summary = str(case.get('summary', '') or '')
            text = f"{raw_text}\n{summary}".strip()
            if not text:
                continue
            doc_hit = 1.0 if (computer in text and ideology in text) else 0.0
            if doc_hit <= 0.0:
                continue
            sentence_score = 0.0
            sentence = ''
            for chunk in self._split_sentences(text):
                if computer in chunk and ideology in chunk:
                    density = min(1.0, len(chunk) / 80.0)
                    sentence_score = max(sentence_score, 0.75 + 0.2 * (1.0 - abs(0.5 - density)))
                    if not sentence:
                        sentence = chunk

            candidate_score = max(0.45 * doc_hit, sentence_score)
            if candidate_score > best_score:
                best_score = candidate_score
                best_sentence = sentence or summary or raw_text[:180]

        return best_sentence, float(np.clip(best_score, 0.0, 1.0))

    def _build_computer_ideology_relations(self, output_dir: str):
        """规则召回 + BERT重排，构建计算机知识点到思政元素关系。"""
        computers = [item for item in self.entity_types.get('computer_science', []) if item]
        ideologies = [item for item in self.entity_types.get('ideology', []) if item]
        if not computers or not ideologies:
            return

        relation_json = {}
        priors = self._load_relation_priors()
        for computer in computers:
            for ideology in ideologies:
                support_sentence, rule_score = self._best_support_sentence(computer, ideology)
                prior_score = float(priors.get((computer, ideology), 0.0))
                if prior_score > 0:
                    rule_score = max(rule_score, 0.35 + 0.55 * prior_score)

                if rule_score < 0.35:
                    continue

                bert_score = 0.0
                if self.relation_reranker and support_sentence:
                    bert_score = self.relation_reranker.score(support_sentence, computer, ideology)

                if self.relation_reranker and self.relation_reranker.available:
                    final_score = 0.5 * rule_score + 0.5 * bert_score
                else:
                    final_score = rule_score

                if final_score < self.relation_threshold:
                    continue

                caption = support_sentence or f"{computer} 与 {ideology} 在教学语料中存在关联"
                self.graph.add_edge(
                    computer,
                    ideology,
                    relation='COMPUTER_REFLECTS_IDEOLOGY',
                    similarity=float(np.clip(final_score, 0.0, 1.0)),
                    caption=caption,
                    media_type='text',
                )
                self.relations['COMPUTER_REFLECTS_IDEOLOGY'].append(
                    (computer, ideology, float(np.clip(final_score, 0.0, 1.0)))
                )
                relation_json[f'{computer}--{ideology}'] = {
                    'source_entity': computer,
                    'target_entity': ideology,
                    'relation': 'COMPUTER_REFLECTS_IDEOLOGY',
                    'rule_score': round(float(rule_score), 4),
                    'bert_score': round(float(bert_score), 4),
                    'similarity': round(float(final_score), 4),
                    'evidence': caption,
                }

        if relation_json:
            self._save_relations_to_json(
                relation_json,
                os.path.join(output_dir, 'computer_ideology_relations.json')
            )

    def _node_labels(self, node_name: str, attrs: Optional[Dict] = None) -> List[str]:
        attrs = attrs or {}
        labels = attrs.get('labels')
        if labels:
            return list(dict.fromkeys(labels if isinstance(labels, (list, tuple)) else [labels]))

        node_type = attrs.get('type') or attrs.get('node_type')
        if node_type == 'teaching_case':
            return ['Entity', 'TeachingCase']
        if node_type == 'media_image':
            return ['Entity', 'Media', 'Image']
        if node_type == 'media_video':
            return ['Entity', 'Media', 'Video']
        if node_type == 'ideology':
            return ['Entity', 'IdeologyElement']
        if node_type == 'computer_science':
            return ['Entity', 'KnowledgePoint']

        # 兜底：根据媒体类型和预置实体类型推断标签，减少 schema 漂移。
        media_type = str(attrs.get('media_type', '') or '').lower()
        if media_type == 'video':
            return ['Entity', 'Media', 'Video']
        if media_type == 'image':
            return ['Entity', 'Media', 'Image']

        if node_name in set(self.entity_types.get('computer_science', []) or []):
            return ['Entity', 'KnowledgePoint']
        if node_name in set(self.entity_types.get('ideology', []) or []):
            return ['Entity', 'IdeologyElement']
        return ['Entity']

    def _enrich_dimension_node_properties(self) -> None:
        """
        为知识点/思政节点补充教学案例与媒体属性，便于教学侧检索
        注意：图像已不再是节点，故不需要收集image_names，但保留videos收集
        """
        for node_name, attrs in list(self.graph.nodes(data=True)):
            labels = set(self._node_labels(node_name, attrs))
            if 'KnowledgePoint' not in labels and 'IdeologyElement' not in labels:
                continue

            case_names = set(str(item).strip() for item in attrs.get('teaching_cases', []) if str(item).strip())
            video_names = set(str(item).strip() for item in attrs.get('videos', []) if str(item).strip())

            neighbors = set(self.graph.successors(node_name)) | set(self.graph.predecessors(node_name))
            for neighbor in neighbors:
                neighbor_attrs = self.graph.nodes[neighbor]
                neighbor_labels = set(self._node_labels(neighbor, neighbor_attrs))
                if 'TeachingCase' in neighbor_labels:
                    case_names.add(str(neighbor))
                if 'Video' in neighbor_labels:
                    video_names.add(str(neighbor))

            attrs['teaching_cases'] = sorted(case_names)
            attrs['videos'] = sorted(video_names)
            attrs['teaching_case_count'] = len(attrs['teaching_cases'])
            attrs['video_count'] = len(attrs['videos'])
            # 图像数和媒体计数已从属性中获取
            attrs['media_count'] = attrs.get('image_count', 0) + attrs['video_count']

    def _get_synonym_dict(self) -> Dict[str, List[str]]:
        """获取中文同义词词典。"""
        synonyms = {
            "编程": ["编码", "程序设计", "软件开发"],
            "代码": ["程序", "源码", "脚本"],
            "规范": ["标准", "准则", "规程"],
            "质量": ["品质", "水平", "标准"],
            "优化": ["改进", "提升", "优化"],
            "算法": ["算法", "方法", "技巧"],
            "数据": ["信息", "资料", "数据"],
            "结构": ["架构", "组织", "结构"],
            "设计": ["设计", "规划", "构思"],
            "模式": ["模式", "样式", "范式"],
            "思维": ["思想", "观念", "理念"],
            "素养": ["修养", "素质", "涵养"],
            "精神": ["精神", "气质", "风貌"],
            "意识": ["意识", "观念", "认识"],
            "责任": ["责任", "义务", "担当"],
            "贡献": ["贡献", "奉献", "付出"],
            "发展": ["发展", "进步", "成长"],
            "课程": ["教程", "课件", "教学"],
            "学习": ["学习", "掌握", "理解"],
            "实践": ["实践", "应用", "操作"],
            "案例": ["例子", "实例", "范例"],
            "优秀": ["良好", "出色", "优秀"],
            "基础": ["基础", "根本", "基石"],
        }

        expanded = {}
        for key, values in synonyms.items():
            expanded[key] = values
            for value in values:
                if value not in expanded:
                    expanded[value] = []
                expanded[value].append(key)

        return expanded

    def _allowed_entity_set(self) -> set:
        """默认只允许预置的计算机核心知识点和思政元素。"""
        allowed = set()
        for key in ('computer_science', 'ideology'):
            for entity in self.entity_types.get(key, []) or []:
                name = str(entity).strip()
                if name:
                    allowed.add(name)
        return allowed

    def _is_allowed_entity(self, entity_name: str) -> bool:
        if not entity_name:
            return False
        allowed = self._allowed_entity_set()
        if not allowed:
            # 若未配置预置实体，保持兼容放行
            return True
        return entity_name.strip() in allowed

    @staticmethod
    def _normalize_term_text(term: str) -> str:
        text = re.sub(r'\s+', '', str(term or '').strip())
        return text.strip('，。；：,:;!?！？、（）()[]【】"\'“”')

    def _normalize_term_list(self, terms) -> List[str]:
        if not terms:
            return []
        if isinstance(terms, str):
            raw_terms = re.split(r'[,，;；\n]+', terms)
        elif isinstance(terms, (list, tuple, set)):
            raw_terms = list(terms)
        else:
            raw_terms = [terms]

        normalized = []
        seen = set()
        for term in raw_terms:
            cleaned = self._normalize_term_text(term)
            if cleaned and cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
        return normalized

    def _source_scope_allows(self, source_kind: str) -> bool:
        scope = self.selective_expansion_source_scope
        if scope in ('both', 'all', 'caption+ocr'):
            return True
        return scope == str(source_kind or '').strip().lower()

    @staticmethod
    def _compact_text(text: str) -> str:
        return re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]+', '', str(text or '').strip())

    def _selective_candidate_score(self, candidate: str, source_text: str, extra_ocr_text: str = '') -> float:
        candidate = self._normalize_term_text(candidate)
        if not candidate:
            return 0.0

        combined = self._normalize_term_text(f"{source_text} {extra_ocr_text}".strip())
        if not combined:
            return 0.0

        compact_candidate = self._compact_text(candidate)
        compact_source = self._compact_text(combined)

        if compact_candidate and compact_candidate in compact_source:
            return 1.0

        chinese_candidate = ''.join(ch for ch in candidate if '\u4e00' <= ch <= '\u9fff')
        chinese_source = ''.join(ch for ch in combined if '\u4e00' <= ch <= '\u9fff')
        if chinese_candidate and chinese_candidate in chinese_source:
            return 0.95

        ratio_score = SequenceMatcher(None, compact_candidate, compact_source).ratio()
        chinese_ratio = SequenceMatcher(None, chinese_candidate, chinese_source).ratio() if chinese_candidate and chinese_source else 0.0

        semantic_score = 0.0
        try:
            semantic_score = float(self.semantic_scorer.similarity(combined, candidate)) if self.semantic_scorer else 0.0
        except Exception:
            semantic_score = 0.0

        return max(ratio_score, chinese_ratio, semantic_score)

    def _can_expand_more(self, target_type: str) -> bool:
        if self.selective_expansion_max_new_total > 0 and len(self._selective_expanded_entities) >= self.selective_expansion_max_new_total:
            return False
        if target_type == 'computer_science' and self.selective_expansion_max_new_computer > 0:
            return self._selective_expanded_counts[target_type] < self.selective_expansion_max_new_computer
        if target_type == 'ideology' and self.selective_expansion_max_new_ideology > 0:
            return self._selective_expanded_counts[target_type] < self.selective_expansion_max_new_ideology
        return True

    def _promotion_target_type(self, target_type: str) -> Tuple[str, str]:
        if target_type == 'computer_science':
            return 'computer_science', 'KnowledgePoint'
        return 'ideology', 'IdeologyElement'

    def _expand_selective_entities(self,
                                   source_name: str,
                                   source_kind: str,
                                   source_text: str,
                                   known_entities: Dict[str, Dict],
                                   media_type: str,
                                   media_path: str,
                                   extra_ocr_text: str = '') -> List[str]:
        """将白名单中的短语按需提升为思政/计算机节点。"""
        if not self.enable_selective_node_expansion:
            return []

        if not self._source_scope_allows(source_kind):
            return []

        expanded = []

        for target_type, terms in self.selective_expansion_catalog.items():
            if not terms or not self._can_expand_more(target_type):
                continue

            for term in terms:
                if not self._can_expand_more(target_type):
                    break
                if not term or term in known_entities or term in self._allowed_entity_set() or term in self._selective_expanded_entities:
                    continue

                score = self._selective_candidate_score(term, source_text, extra_ocr_text)
                support = 0
                if self._normalize_term_text(term) in self._normalize_term_text(source_text):
                    support += 1
                if extra_ocr_text and self._normalize_term_text(term) in self._normalize_term_text(extra_ocr_text):
                    support += 1
                if score >= self.selective_expansion_min_score:
                    support += 1

                if support < self.selective_expansion_min_support:
                    continue

                node_type, node_label = self._promotion_target_type(target_type)
                node_attrs = {
                    'type': node_type,
                    'labels': ['Entity', node_label],
                    'source': f'{source_name}::selective_expansion',
                    'sources': {source_name},
                    'keywords': [term],
                    'count': support,
                    'selective_expanded': True,
                    'selective_source_kind': source_kind,
                    'selective_target_type': target_type,
                    'media_type': 'text',
                    'promotion_evidence': source_text[:500],
                }
                self.graph.add_node(term, **node_attrs)
                self.entity_types.setdefault(node_type, [])
                if term not in self.entity_types[node_type]:
                    self.entity_types[node_type].append(term)
                known_entities[term] = {
                    'type': node_type,
                    'source': f'{source_name}::selective_expansion',
                    'sources': {source_name},
                    'keywords': {term},
                    'count': support,
                    'promoted': True,
                }
                self._selective_expanded_entities.add(term)
                self._selective_expanded_counts[target_type] += 1

                relation_type = 'MEDIA_LINKED_IMAGE' if media_type == 'image' else 'MEDIA_LINKED_VIDEO'
                self.graph.add_edge(
                    term,
                    source_name,
                    relation=relation_type,
                    similarity=score if score > 0 else 1.0,
                    caption=source_text,
                    ocr_text=extra_ocr_text or source_text,
                    media_path=media_path,
                    media_type=media_type,
                )
                self.relations[relation_type].append((term, source_name, score if score > 0 else 1.0))
                expanded.append(term)

        if expanded:
            logger.info(f"[{source_name}] 有选择扩展节点: {expanded}")
        return expanded

    def _infer_stage_tags(self, text: str) -> List[str]:
        """从教学文本中推断学段标签。"""
        text = text or ""
        stage_alias = {
            '小学': ['小学', '小学生', '基础教育'],
            '初中': ['初中', '中学生'],
            '高中': ['高中', '高考'],
            '大学': ['大学', '本科', '高校', '高等教育'],
            '高职': ['高职', '职业教育', '职教'],
        }
        matched = [stage for stage, aliases in stage_alias.items() if any(token in text for token in aliases)]
        return matched or ['大学']

    def _extract_case_structured_fields(self, title: str, raw_text: str, keywords: List[str]) -> Dict:
        """抽取教学案例结构化字段，供教学适配检索使用。"""
        source_text = f"{title}\n{raw_text}"
        isbn_match = re.search(r'ISBN[:：]?\s*([0-9\-Xx]+)', source_text)
        chapter_titles = re.findall(r'(?:第[一二三四五六七八九十百\d]+章[^\n]{0,40})', source_text)

        discipline_lexicon = [
            '计算机', '软件工程', '人工智能', '机器学习', '大数据', '数据库', '网络', '程序设计', 'Python', 'Hadoop'
        ]
        disciplines = [token for token in discipline_lexicon if token.lower() in source_text.lower()]

        ideology_hits = [
            entity for entity in self.entity_types.get('ideology', [])
            if entity and entity in source_text
        ]

        # 目标字段用高频关键词生成，便于查询解释
        objective_terms = keywords[:5]

        return {
            'isbn': isbn_match.group(1) if isbn_match else '',
            'stage_tags': self._infer_stage_tags(source_text),
            'course_tags': sorted(set(disciplines + objective_terms))[:10],
            'ideology_tags': sorted(set(ideology_hits))[:10],
            'chapter_count': len(chapter_titles),
            'chapter_titles': chapter_titles[:20],
            'teaching_objectives': objective_terms,
        }
    
    def _test_connection(self):
        """测试Neo4j连接"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            return True
        except Exception as e:
            logger.error(f"数据库连接测试失败: {str(e)}")
            return False

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _data_root(self) -> Path:
        return self._project_root() / 'data'

    def _normalize_media_path(self, path_value: str) -> str:
        """将媒体路径规范为相对 data/ 的可移植路径。"""
        if not path_value:
            return ''

        raw = str(path_value).strip()
        if raw.startswith('/media/'):
            return raw

        p = Path(raw)
        data_root = self._data_root().resolve()

        candidates = []
        if p.is_absolute():
            candidates.append(p.resolve())
        else:
            candidates.append((self._project_root() / p).resolve())
            candidates.append((data_root / p).resolve())

        for candidate in candidates:
            try:
                relative = candidate.relative_to(data_root)
                return relative.as_posix()
            except Exception:
                continue

        return raw.replace('\\', '/')

    def _media_url_from_path(self, path_value: str) -> str:
        normalized = self._normalize_media_path(path_value)
        if not normalized:
            return ''
        if normalized.startswith('/media/'):
            return normalized
        return f"/media/{normalized}"

    def _resolve_persisted_media_fields(self, attrs: Dict) -> Dict[str, str]:
        """入库时统一补齐媒体路径字段；播放优先分段 clip。"""
        source_path = self._normalize_media_path(attrs.get('path', ''))
        clip_path = self._normalize_media_path(attrs.get('clip_path', ''))

        clip_url = attrs.get('clip_url', '') or (self._media_url_from_path(clip_path) if clip_path else '')
        source_url = attrs.get('media_url', '') or (self._media_url_from_path(source_path) if source_path else '')

        # 播放侧统一优先 clip
        preferred_relative = clip_path or source_path
        preferred_url = clip_url or source_url or (self._media_url_from_path(preferred_relative) if preferred_relative else '')

        return {
            'path': source_path,
            'clip_path': clip_path,
            'clip_url': clip_url,
            'relative_path': preferred_relative,
            'media_url': preferred_url,
        }

    def _resolve_existing_clip(self, video_path: str) -> Tuple[str, str]:
        """若已存在预处理分段视频，返回其相对路径和URL。"""
        normalized_video = self._normalize_media_path(video_path)
        if not normalized_video:
            return '', ''

        stem = Path(normalized_video).stem
        clip_dir = self._data_root() / 'clips'
        clip_candidates = [
            clip_dir / f"{stem}_clip.mp4",
            clip_dir / f"{stem}.mp4",
        ]

        for candidate in clip_candidates:
            if candidate.exists() and candidate.is_file():
                rel = self._normalize_media_path(str(candidate))
                return rel, self._media_url_from_path(rel)
        return '', ''

    @staticmethod
    def _semantic_text_from_name(name: str) -> str:
        stem = Path(str(name or '')).stem
        chunks = re.split(r'[\s_\-]+', stem)
        return ' '.join([item for item in chunks if len(item) >= 2])

    def _valid_image_item(self, img_name: str, img_data: Dict) -> bool:
        path = str(img_data.get('path') or '')
        metadata = img_data.get('metadata') or {}
        if not path or not os.path.exists(path):
            self.invalid_media_records.append({'name': img_name, 'type': 'image', 'reason': 'missing_path'})
            return False
        width, height = metadata.get('size', (0, 0)) if isinstance(metadata.get('size'), (tuple, list)) else (0, 0)
        if int(width or 0) <= 0 or int(height or 0) <= 0:
            self.invalid_media_records.append({'name': img_name, 'type': 'image', 'reason': 'invalid_size'})
            return False
        return True

    def _valid_video_item(self, video_name: str, video_data: Dict) -> bool:
        path = str(video_data.get('path') or '')
        metadata = video_data.get('metadata') or {}
        if not path or not os.path.exists(path):
            self.invalid_media_records.append({'name': video_name, 'type': 'video', 'reason': 'missing_path'})
            return False
        frame_count = int(metadata.get('frame_count') or 0)
        fps = float(metadata.get('fps') or 0.0)
        width = int(metadata.get('width') or 0)
        height = int(metadata.get('height') or 0)
        if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
            self.invalid_media_records.append({'name': video_name, 'type': 'video', 'reason': 'invalid_metadata'})
            return False
        return True
    
    def _load_known_entities(self, txt_dir: str) -> Dict[str, Dict]:
        """
        从文本文件加载已知实体
        
        Args:
            txt_dir: 文本目录
            
        Returns:
            实体字典 {entity_name: {type, source, keywords}}
        """
        logger.info("正在加载已知实体...")
        known_entities = {}
        
        txt_data = self.text_processor.load_txt_files(txt_dir)
        
        for filename, data in txt_data.items():
            case_name = Path(filename).stem
            flat_entities: List[str] = []
            for entity_list in data['entities'].values():
                for entity in entity_list:
                    if self._is_allowed_entity(entity):
                        flat_entities.append(entity)
            structured_fields = self._extract_case_structured_fields(
                title=filename,
                raw_text=data['raw_text'],
                keywords=data['keywords'],
            )
            self.case_records[case_name] = {
                'name': case_name,
                'title': filename,
                'summary': summarize_text(data['raw_text']),
                'raw_text': data['raw_text'],
                'keywords': data['keywords'][:12],
                'entities': sorted(set(flat_entities)),
                'source_file': filename,
                'media_type': 'text',
                **structured_fields,
            }

            # 处理所有实体
            for entity_type, entities in data['entities'].items():
                for entity in entities:
                    if not self._is_allowed_entity(entity):
                        continue
                    if entity not in known_entities:
                        known_entities[entity] = {
                            'type': entity_type,
                            'source': filename,
                            'sources': {filename},
                            'keywords': set(),
                            'count': 0
                        }
                    known_entities[entity]['count'] += 1
                    known_entities[entity]['sources'].add(filename)
                    known_entities[entity]['keywords'].update(data['keywords'])
        
        logger.info(f"已加载 {len(known_entities)} 个已知实体")
        return known_entities
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """
        从文本提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        entities_dict = self.text_processor.extract_entities(text)
        entities = []
        for entity_list in entities_dict.values():
            entities.extend(entity_list)
        return list(set(entities))
    
    def _match_entities(self,
                        caption_entities: List[str],
                        known_entities: Dict[str, Dict],
                        context_text: Optional[str] = None,
                        top_k: int = 12,
                        semantic_threshold: float = 0.45) -> List[Tuple[str, float]]:
        """
        将图像/视频描述中的实体与已知实体匹配

        支持多种匹配策略：
        1. 精确匹配 (similarity = 1.0)
        2. 模糊匹配 (子串包含, similarity = 0.8)
        3. 同义词匹配 (similarity = 0.9)
        4. 拼音匹配 (similarity = 0.6)

        Args:
            caption_entities: 描述中的实体
            known_entities: 已知实体字典

        Returns:
            匹配的实体及相似度列表
        """
        matches: List[Tuple[str, float]] = []
        matched_entities = set()
        query_text = (context_text or " ".join(caption_entities) or "").strip()

        # Phase3: 语义向量召回优先，减少对纯字符串包含的依赖
        candidate_texts = {
            entity: {
                'name': entity,
                'type': info.get('type', ''),
                'source': info.get('source', ''),
                'keywords': list(info.get('keywords', []))[:8],
            }
            for entity, info in known_entities.items()
        }
        if candidate_texts and query_text:
            ranked = self.semantic_scorer.rank_candidates(query_text, candidate_texts, top_k=max(top_k * 2, top_k))
            for item in ranked:
                score = float(item.get('score', 0.0))
                name = item.get('name')
                if not name or name in matched_entities:
                    continue
                if score >= semantic_threshold:
                    matches.append((name, score))
                    matched_entities.add(name)

        # exact-match 提权，确保同名实体稳定靠前
        for entity in caption_entities:
            if entity in known_entities and entity not in matched_entities:
                matches.append((entity, 1.0))
                matched_entities.add(entity)

        # 子串匹配兜底
        for caption_entity in caption_entities:
            for known_entity in known_entities:
                if known_entity in matched_entities:
                    continue
                if caption_entity.lower() in known_entity.lower() or known_entity.lower() in caption_entity.lower():
                    matches.append((known_entity, 0.8))
                    matched_entities.add(known_entity)
                if len(matches) >= top_k:
                    break
            if len(matches) >= top_k:
                break

        # 拼音匹配兜底（可选）
        if len(matches) < top_k:
            try:
                import pinyin
                for caption_entity in caption_entities:
                    caption_pinyin = pinyin.get(caption_entity, format="strip", delimiter="")
                    for known_entity in known_entities:
                        if known_entity in matched_entities:
                            continue
                        known_pinyin = pinyin.get(known_entity, format="strip", delimiter="")
                        if caption_pinyin and known_pinyin and caption_pinyin.lower() in known_pinyin.lower():
                            matches.append((known_entity, 0.6))
                            matched_entities.add(known_entity)
                        if len(matches) >= top_k:
                            break
                    if len(matches) >= top_k:
                        break
            except ImportError:
                logger.debug("pinyin库未安装，跳过拼音匹配")

        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:top_k]

    def _build_teaching_cases(self, known_entities: Dict[str, Dict], output_dir: str):
        """把文本语料建成教学案例节点，并与知识点/思政元素相连。"""
        case_relations = {}
        for case_name, case_info in self.case_records.items():
            self.graph.add_node(
                case_name,
                type='teaching_case',
                labels=['Entity', 'TeachingCase'],
                title=case_info.get('title', case_name),
                summary=case_info.get('summary', ''),
                keywords=case_info.get('keywords', []),
                source_file=case_info.get('source_file', ''),
                media_type='text',
                stage_tags=case_info.get('stage_tags', []),
                course_tags=case_info.get('course_tags', []),
                ideology_tags=case_info.get('ideology_tags', []),
                teaching_objectives=case_info.get('teaching_objectives', []),
                chapter_count=case_info.get('chapter_count', 0),
                chapter_titles=case_info.get('chapter_titles', []),
                isbn=case_info.get('isbn', ''),
            )

            ranked = self.semantic_scorer.rank_candidates(
                case_info.get('summary', '') or case_info.get('raw_text', ''),
                {
                    entity: {
                        'name': entity,
                        'type': info.get('type', ''),
                        'keywords': list(info.get('keywords', []))[:8],
                        'source': info.get('source', ''),
                    }
                    for entity, info in known_entities.items()
                },
                top_k=5,
            )

            for entity in case_info.get('entities', []):
                if entity in known_entities:
                    self.graph.add_edge(case_name, entity,
                                        relation='MENTIONS',
                                        similarity=1.0,
                                        caption=case_info.get('summary', ''),
                                        media_type='text')
                    self.relations['MENTIONS'].append((case_name, entity, 1.0))
                    case_relations[f'{case_name}--{entity}'] = {
                        'case': case_name,
                        'entity': entity,
                        'relation': 'MENTIONS',
                        'similarity': 1.0,
                        'summary': case_info.get('summary', ''),
                    }

            for item in ranked:
                entity = item['name']
                if item['score'] < 0.45 or entity not in known_entities:
                    continue
                self.graph.add_edge(case_name, entity,
                                    relation='RELATED',
                                    similarity=float(item['score']),
                                    caption=case_info.get('summary', ''),
                                    media_type='text')
                self.relations['RELATED'].append((case_name, entity, float(item['score'])))
                case_relations[f'{case_name}--{entity}--related'] = {
                    'case': case_name,
                    'entity': entity,
                    'relation': 'RELATED',
                    'similarity': float(item['score']),
                    'summary': case_info.get('summary', ''),
                }

        self._save_relations_to_json(case_relations, os.path.join(output_dir, 'teaching_case_relations.json'))

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """计算余弦相似度并截断到[0,1]。"""
        a_norm = np.linalg.norm(vec_a)
        b_norm = np.linalg.norm(vec_b)
        if a_norm <= 1e-12 or b_norm <= 1e-12:
            return 0.0
        score = float(np.dot(vec_a, vec_b) / (a_norm * b_norm))
        return max(0.0, min(1.0, score))

    def _build_cross_modal_links(self, output_dir: str, threshold: float = 0.55, top_k: int = 3):
        """
        基于视觉特征建立跨模态链接（仅视频）
        因为图像不再作为节点存储，所以只处理视频间的关联
        """
        video_items = []

        for node_name, attrs in self.graph.nodes(data=True):
            if attrs.get('media_type') != 'video':
                continue
            
            feats = attrs.get('features')
            if feats is None:
                continue
            
            try:
                frame_feats = np.asarray(feats, dtype=np.float32)
                pooled = frame_feats.mean(axis=0) if frame_feats.ndim > 1 else frame_feats.reshape(-1)
                video_items.append((node_name, pooled.reshape(-1)))
            except Exception:
                continue

        if len(video_items) < 2:
            return

        cross_relations = {}
        for i, (video_name_1, video_feat_1) in enumerate(video_items):
            for j, (video_name_2, video_feat_2) in enumerate(video_items):
                if i >= j:
                    continue
                
                sim = self._cosine_similarity(video_feat_1, video_feat_2)
                if sim >= threshold:
                    rel_key = f"{video_name_1}--{video_name_2}--cross"
                    cross_relations[rel_key] = {
                        'source_media': video_name_1,
                        'target_media': video_name_2,
                        'relation': 'CONNECTED_BY_VISUAL',
                        'similarity': round(float(sim), 4),
                        'source_type': 'video',
                        'target_type': 'video',
                    }
                    self.graph.add_edge(
                        video_name_1,
                        video_name_2,
                        relation='CONNECTED_BY_VISUAL',
                        similarity=float(sim),
                        media_type='cross_modal',
                    )
                    self.relations['CONNECTED_BY_VISUAL'].append((video_name_1, video_name_2, float(sim)))

        if cross_relations:
            self._save_relations_to_json(cross_relations, os.path.join(output_dir, 'cross_modal_relations.json'))
    
    def build_kg(self, data_dir: str, output_dir: str = 'kg_output',
                computer_entities: List[str] = None, ideology_entities: List[str] = None,
                custom_relations: List[Tuple[str, str, str]] = None,
                match_top_k: int = 12,
                semantic_threshold: float = 0.45):
        """
        构建完整的知识图谱
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
            computer_entities: 计算机核心知识点实体列表
            ideology_entities: 思政元素实体列表
            custom_relations: 自定义关系列表 [(entity1, entity2, relation_type), ...]
        """
        logger.info("=" * 50)

        self.match_top_k = max(1, int(match_top_k))
        self.semantic_threshold = max(0.0, min(1.0, float(semantic_threshold)))
        logger.info("开始构建知识图谱")
        logger.info("=" * 50)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 第一步：预先插入实体
        if computer_entities and ideology_entities:
            self.insert_predefined_entities(computer_entities, ideology_entities)
        
        # 第二步：创建自定义关系
        if custom_relations:
            self.create_custom_relations(custom_relations)
        
        # 第三步：加载已知实体
        txt_dir = os.path.join(data_dir, 'txt')
        known_entities = self._load_known_entities(txt_dir)

        # 第三步补充：构建教学案例节点
        self._build_teaching_cases(known_entities, output_dir)
        
        # 第四步：处理图像
        img_dir = os.path.join(data_dir, 'img')
        if os.path.exists(img_dir):
            logger.info("\n处理图像数据...")
            self._process_images(img_dir, known_entities, output_dir)
        
        # 第五步：处理视频
        video_dir = os.path.join(data_dir, 'video')
        if os.path.exists(video_dir):
            logger.info("\n处理视频数据...")
            self._process_videos(video_dir, known_entities, output_dir)

        # 第五步补充：构建计算机知识点 -> 思政元素关系（包含选择性扩展后的新节点）
        self._build_computer_ideology_relations(output_dir)

        # 保存已知实体（此时已包含选择性扩展节点）
        self._save_entities_to_json(
            known_entities,
            os.path.join(output_dir, 'known_entities.json')
        )

        # 第六步补充：建立跨模态媒体连接
        self._build_cross_modal_links(output_dir)
        
        # 第七步：存储到Neo4j
        if self.driver:
            logger.info("\n存储到Neo4j数据库...")
            self._store_to_neo4j()
        
        # 第八步：保存图谱统计
        self._save_kg_stats(output_dir)
        
        logger.info("\n" + "=" * 50)
        logger.info("知识图谱构建完成")
        logger.info("=" * 50)
    
    def _process_images(self, 
                       img_dir: str,
                       known_entities: Dict[str, Dict],
                       output_dir: str):
        """
        处理图像 - 生成描述，用于增强知识点和思政节点的属性
        图像本身不存储为节点，仅作为内容来源
        """
        image_enrichment_records = {}  # 记录图像对节点属性的增强
        images = self.image_processor.load_images(img_dir)
        
        for img_name, img_data in images.items():
            if not self._valid_image_item(img_name, img_data):
                logger.warning(f"跳过异常图像: {img_name}")
                continue
            image_path = self._normalize_media_path(img_data['path'])
            name_semantics = self._semantic_text_from_name(img_name)
            
            # 生成图像描述
            caption = self.caption_generator.generate_image_caption(img_data['path'])
            ocr_text = self.caption_generator.recognize_text_from_image(img_data['path'])
            
            if caption:
                logger.info(f"[{img_name}] 描述: {caption}")

                enriched_text = f"{caption} {ocr_text or ''} {name_semantics}".strip()
                
                # 从描述提取实体
                caption_entities = self._extract_entities_from_text(enriched_text)

                # 有选择地扩展思政节点（默认关闭，需显式启用）
                self._expand_selective_entities(
                    source_name=img_name,
                    source_kind='caption',
                    source_text=enriched_text,
                    known_entities=known_entities,
                    media_type='image',
                    media_path=image_path,
                    extra_ocr_text=ocr_text or '',
                )
                
                # 匹配到已知实体
                matches = self._match_entities(
                    caption_entities,
                    known_entities,
                    context_text=enriched_text,
                    top_k=getattr(self, 'match_top_k', 12),
                    semantic_threshold=getattr(self, 'semantic_threshold', 0.45),
                )
                
                # 不创建图像节点，而是增强匹配实体的属性
                for matched_entity, similarity in matches:
                    if not self.graph.has_node(matched_entity):
                        continue
                    
                    # 获取现有节点属性
                    node_attrs = self.graph.nodes[matched_entity]
                    
                    # 更新图像相关属性
                    if 'image_captions' not in node_attrs:
                        node_attrs['image_captions'] = []
                    if 'image_ocr_texts' not in node_attrs:
                        node_attrs['image_ocr_texts'] = []
                    if 'image_paths' not in node_attrs:
                        node_attrs['image_paths'] = []
                    if 'image_count' not in node_attrs:
                        node_attrs['image_count'] = 0
                    
                    # 去重后添加
                    if caption not in node_attrs['image_captions']:
                        node_attrs['image_captions'].append(caption)
                    if ocr_text and ocr_text not in node_attrs['image_ocr_texts']:
                        node_attrs['image_ocr_texts'].append(ocr_text)
                    if image_path not in node_attrs['image_paths']:
                        node_attrs['image_paths'].append(image_path)
                        node_attrs['image_count'] += 1
                    
                    image_enrichment_records[f"{matched_entity}--{img_name}"] = {
                        'entity': matched_entity,
                        'image': img_name,
                        'image_path': image_path,
                        'caption': caption,
                        'ocr_text': ocr_text,
                        'similarity': similarity,
                    }
                    
                    logger.info(f"  → 增强节点 '{matched_entity}' (相似度: {similarity:.2f})")
        
        # 保存图像增强记录
        self._save_relations_to_json(
            image_enrichment_records,
            os.path.join(output_dir, 'image_enrichment_records.json')
        )
    
    def _process_videos(self,
                       video_dir: str,
                       known_entities: Dict[str, Dict],
                       output_dir: str):
        """
        处理视频并建立实体关系
        支持 Video 节点和 VideoClip 子节点
        """
        video_relations = {}
        videos = self.video_processor.load_videos(video_dir)
        
        for video_name, video_data in videos.items():
            if not self._valid_video_item(video_name, video_data):
                logger.warning(f"跳过异常视频: {video_name}")
                continue
            video_path = self._normalize_media_path(video_data['path'])
            video_url = self._media_url_from_path(video_path)
            clip_path, clip_url = self._resolve_existing_clip(video_data['path'])
            name_semantics = self._semantic_text_from_name(video_name)
            
            # 创建Video节点
            self.graph.add_node(video_name,
                                type='media_video',
                                labels=['Entity', 'Media', 'Video'],
                                path=video_path,
                                clip_path=clip_path,
                                clip_url=clip_url,
                                media_url=clip_url or video_url,
                                media_type='video',
                                feature_backend=getattr(self.video_processor.image_processor, 'feature_backend', 'unknown'),
                                features=video_data.get('features'),
                                fps=video_data.get('metadata', {}).get('fps', 0),
                                frame_count=video_data.get('metadata', {}).get('frame_count', 0),
                                width=video_data.get('metadata', {}).get('width', 0),
                                height=video_data.get('metadata', {}).get('height', 0),
                                duration=video_data.get('metadata', {}).get('duration', 0),
                                video_clips=[])  # 关联的视频剪辑列表
            
            self.media_records[video_name] = {
                'name': video_name,
                'path': video_path,
                'clip_path': clip_path,
                'media_url': clip_url or video_url,
                'media_type': 'video',
                'metadata': video_data.get('metadata', {}),
            }

            # 生成视频描述
            caption = self.caption_generator.generate_video_caption(video_data['path'])
            
            if caption:
                logger.info(f"[{video_name}] 描述: {caption}")
                enriched_text = f"{caption} {name_semantics}".strip()
                
                # 从描述提取实体
                caption_entities = self._extract_entities_from_text(enriched_text)

                # 有选择地扩展思政节点（默认关闭，需显式启用）
                self._expand_selective_entities(
                    source_name=video_name,
                    source_kind='caption',
                    source_text=enriched_text,
                    known_entities=known_entities,
                    media_type='video',
                    media_path=clip_path or video_path,
                )
                
                # 匹配到已知实体
                matches = self._match_entities(
                    caption_entities,
                    known_entities,
                    context_text=enriched_text,
                    top_k=getattr(self, 'match_top_k', 12),
                    semantic_threshold=getattr(self, 'semantic_threshold', 0.45),
                )
                
                # 建立关系
                for matched_entity, similarity in matches:
                    relation_key = f"{matched_entity}--{video_name}"
                    video_relations[relation_key] = {
                        'source_entity': matched_entity,
                        'media': video_name,
                        'media_type': 'video',
                        'media_path': clip_path or video_path,
                        'media_url': clip_url or video_url,
                        'clip_path': clip_path,
                        'caption': caption,
                        'similarity': similarity,
                        'metadata': video_data['metadata']
                    }
                    
                    # 添加到图中
                    self.graph.add_edge(matched_entity, video_name,
                                      relation='MEDIA_LINKED_VIDEO',
                                      similarity=similarity,
                                      caption=caption,
                                      media_path=clip_path or video_path,
                                      media_url=clip_url or video_url,
                                      clip_path=clip_path,
                                      media_type='video')
                    
                    self.relations['MEDIA_LINKED_VIDEO'].append(
                        (matched_entity, video_name, similarity)
                    )

                best_case = self.semantic_scorer.rank_candidates(
                    enriched_text,
                    self.case_records,
                    top_k=1,
                )
                if best_case and best_case[0]['score'] >= 0.35:
                    case_name = best_case[0]['name']
                    self.graph.add_edge(video_name, case_name,
                                      relation='LINKS_TO_CASE',
                                      similarity=float(best_case[0]['score']),
                                      caption=caption,
                                      media_type='video')
                    self.relations['LINKS_TO_CASE'].append((video_name, case_name, float(best_case[0]['score'])))
                    video_relations[f'{video_name}--{case_name}'] = {
                        'media': video_name,
                        'case': case_name,
                        'media_type': 'video',
                        'relation': 'LINKS_TO_CASE',
                        'similarity': float(best_case[0]['score']),
                        'caption': caption,
                    }
        
        # 保存视频关系
        self._save_relations_to_json(
            video_relations,
            os.path.join(output_dir, 'video_relations.json')
        )
    
    def _store_to_neo4j(self):
        """
        将知识图谱存储到Neo4j
        支持新的数据模型：图像作为属性，视频和视频剪辑作为节点
        """
        if not self.driver:
            logger.warning("Neo4j驱动未初始化")
            return
        
        try:
            # 入库前统一补齐知识点/思政节点的教学案例与媒体属性。
            self._enrich_dimension_node_properties()
            with self.driver.session(database=self.database) as session:
                # 清空现有数据（可选）
                # session.run("MATCH (n) DETACH DELETE n")
                
                # 创建实体节点
                for entity_id, (entity_name, attrs) in enumerate(self.graph.nodes(data=True)):
                    labels = self._node_labels(entity_name, attrs)
                    label_clause = ":".join(labels)
                    media_fields = self._resolve_persisted_media_fields(attrs)
                    
                    # 准备属性
                    props = {
                        'name': entity_name,
                        'id': entity_id,
                        'created_at': datetime.now().isoformat(),
                        'node_type': attrs.get('type', attrs.get('node_type', 'entity')),
                        'labels': labels,
                        'summary': attrs.get('summary', ''),
                        'title': attrs.get('title', ''),
                        'source_file': attrs.get('source_file', ''),
                        'path': media_fields.get('path', ''),
                        'clip_path': media_fields.get('clip_path', ''),
                        'clip_url': media_fields.get('clip_url', ''),
                        'relative_path': media_fields.get('relative_path', ''),
                        'media_url': media_fields.get('media_url', ''),
                        'media_type': attrs.get('media_type', ''),
                        'caption': attrs.get('caption', ''),
                        'ocr_text': attrs.get('ocr_text', ''),
                        'keywords': attrs.get('keywords', []),
                        'stage_tags': attrs.get('stage_tags', []),
                        'course_tags': attrs.get('course_tags', []),
                        'ideology_tags': attrs.get('ideology_tags', []),
                        'teaching_objectives': attrs.get('teaching_objectives', []),
                        'chapter_count': attrs.get('chapter_count', 0),
                        'chapter_titles': attrs.get('chapter_titles', []),
                        'isbn': attrs.get('isbn', ''),
                        'teaching_cases': attrs.get('teaching_cases', []),
                        'videos': attrs.get('videos', []),
                        'teaching_case_count': attrs.get('teaching_case_count', 0),
                        'video_count': attrs.get('video_count', 0),
                        'media_count': attrs.get('media_count', 0),
                        # 新增：图像作为属性而非节点
                        'image_captions': attrs.get('image_captions', []),
                        'image_ocr_texts': attrs.get('image_ocr_texts', []),
                        'image_paths': attrs.get('image_paths', []),
                        'image_count': attrs.get('image_count', 0),
                        # 新增：视频剪辑列表
                        'video_clips': attrs.get('video_clips', []),
                    }
                    
                    session.run(f"""
                        MERGE (n:{label_clause} {{name: $name}})
                        SET n += $props
                    """, name=entity_name, props=props)
                
                # 创建关系
                for source, target, data in self.graph.edges(data=True):
                    rel_type = self._normalize_relation_type(data.get('relation', 'CONNECTED'))
                    media_path = self._normalize_media_path(data.get('media_path', ''))
                    media_url = data.get('media_url', '') or self._media_url_from_path(media_path)
                    session.run(("""
                        MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                        MERGE (a)-[r:%s]->(b)
                        SET r.type = $type, r.similarity = $similarity, r.caption = $caption, r.created_at = $created_at, r.media_path = $media_path, r.relative_path = $media_path, r.media_url = $media_url, r.media_type = $media_type, r.ocr_text = $ocr_text
                    """ % rel_type), source=source, target=target,
                        type=rel_type,
                        similarity=data.get('similarity', 0.0),
                        caption=data.get('caption', ''),
                        media_path=media_path,
                        media_url=media_url,
                        media_type=data.get('media_type', ''),
                        ocr_text=data.get('ocr_text', ''),
                        created_at=datetime.now().isoformat())
                
                logger.info("已存储到Neo4j数据库")
        except Exception as e:
            logger.error(f"存储到Neo4j失败: {str(e)}")
    
    def _save_entities_to_json(self, entities: Dict, filepath: str):
        """保存实体到JSON文件"""
        data = {}
        for entity_name, entity_info in entities.items():
            data[entity_name] = {
                'type': entity_info['type'],
                'source': entity_info['source'],
                'sources': sorted(list(entity_info.get('sources', {entity_info['source']}))),
                'keywords': list(entity_info['keywords']),
                'count': entity_info['count']
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存实体到: {filepath}")
    
    def _save_relations_to_json(self, relations: Dict, filepath: str):
        """保存关系到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(relations, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存关系到: {filepath}")
    
    def _save_kg_stats(self, output_dir: str):
        """保存知识图谱统计信息"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'relation_types': list(self.relations.keys()),
            'relation_counts': {
                rel_type: len(rel_list)
                for rel_type, rel_list in self.relations.items()
            },
            'case_count': len(self.case_records),
            'media_count': len(self.media_records),
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0
        }
        if self.invalid_media_records:
            stats['invalid_media_records'] = self.invalid_media_records
        
        with open(os.path.join(output_dir, 'kg_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"\n知识图谱统计:")
        logger.info(f"  - 节点数: {stats['total_nodes']}")
        logger.info(f"  - 边数: {stats['total_edges']}")
        logger.info(f"  - 密度: {stats['density']:.4f}")
        for rel_type, count in stats['relation_counts'].items():
            logger.info(f"  - {rel_type}: {count}")
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("已关闭Neo4j连接")
    
    def insert_predefined_entities(self, computer_entities: List[str], ideology_entities: List[str]):
        """
        预先插入实体到知识图谱
        
        Args:
            computer_entities: 计算机核心知识点实体列表
            ideology_entities: 思政元素实体列表
        """
        logger.info("正在预先插入实体...")
        
        self.entity_types['computer_science'] = computer_entities
        self.entity_types['ideology'] = ideology_entities
        
        # 添加到图中
        for entity in computer_entities + ideology_entities:
            node_type = 'computer_science' if entity in computer_entities else 'ideology'
            self.graph.add_node(
                entity,
                type=node_type,
                labels=['Entity', 'KnowledgePoint'] if node_type == 'computer_science' else ['Entity', 'IdeologyElement'],
            )
        
        # 创建相似关系（同类实体间）
        for i, entity1 in enumerate(computer_entities):
            for j, entity2 in enumerate(computer_entities):
                if i != j:
                    self.graph.add_edge(entity1, entity2, relation='SIMILAR', similarity=0.5)
                    self.relations['SIMILAR'].append((entity1, entity2, 0.5))
        
        for i, entity1 in enumerate(ideology_entities):
            for j, entity2 in enumerate(ideology_entities):
                if i != j:
                    self.graph.add_edge(entity1, entity2, relation='SIMILAR', similarity=0.5)
                    self.relations['SIMILAR'].append((entity1, entity2, 0.5))
        
        logger.info(f"已插入 {len(computer_entities)} 个计算机实体和 {len(ideology_entities)} 个思政实体")
    
    def create_custom_relations(self, relations_data: List[Tuple[str, str, str]]):
        """
        创建自定义关系
        
        Args:
            relations_data: 关系数据 [(entity1, entity2, relation_type), ...]
        """
        logger.info("正在创建自定义关系...")
        
        for entity1, entity2, rel_type in relations_data:
            rel_type_norm = self._normalize_relation_type(rel_type)
            similarity = self._relation_score(rel_type_norm)
            
            self.graph.add_edge(entity1, entity2, relation=rel_type_norm, similarity=similarity)
            self.relations[rel_type_norm].append((entity1, entity2, similarity))
        
        logger.info(f"已创建 {len(relations_data)} 个自定义关系")
