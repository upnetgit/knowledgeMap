#!/usr/bin/env python3
"""
跨模态知识图谱Web查询界面
提供图形化界面查询和可视化
"""

from __future__ import annotations

import logging
import os
import re
import ast
import json
import importlib
import atexit
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
try:
    _neo4j_module = importlib.import_module("neo4j")
    GraphDatabase = _neo4j_module.GraphDatabase
except Exception:
    GraphDatabase = None

try:
    _flask_module = importlib.import_module("flask")
    Flask = _flask_module.Flask
    request = _flask_module.request
    jsonify = _flask_module.jsonify
    render_template = _flask_module.render_template
    send_from_directory = _flask_module.send_from_directory
except Exception:
    Flask = None
    request = jsonify = render_template = send_from_directory = None

try:
    _werkzeug_utils = importlib.import_module("werkzeug.utils")
    secure_filename = _werkzeug_utils.secure_filename
except Exception:
    secure_filename = None

try:
    from xmodaler.kg.processors import VideoEditor
except Exception:
    VideoEditor = None

try:
    from xmodaler.kg.semantic import SemanticScorer, summarize_text
except Exception:
    SemanticScorer = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if Flask is None or request is None or jsonify is None or render_template is None or send_from_directory is None:
    raise ImportError("Flask is required to run app.py")

app = Flask(__name__)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}
AUDIO_EXTS = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'}
TEXT_EXTS = {'.txt'}


def _safe_text(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(',') if item.strip()]


def _sanitize_upload_filename(filename: str) -> str:
    raw = str(filename or '').strip()
    suffix = Path(raw).suffix.lower()
    cleaned = raw
    if secure_filename:
        cleaned = secure_filename(cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        cleaned = f"upload_{uuid4().hex}{suffix}"
    elif suffix and Path(cleaned).suffix.lower() != suffix:
        cleaned = f"{Path(cleaned).stem}{suffix}"
    return cleaned


def _classify_media_suffix(suffix: str) -> Tuple[str, str]:
    suffix = str(suffix or '').lower().strip()
    if suffix in IMAGE_EXTS:
        return 'image', 'uploads/img'
    if suffix in VIDEO_EXTS:
        return 'video', 'uploads/video'
    if suffix in AUDIO_EXTS:
        # C 方案：音频与视频走同一路由，节点层通过 source_media_type 区分。
        return 'audio', 'uploads/video'
    return 'unknown', ''


def _filename_keywords(path_value: str) -> List[str]:
    stem = Path(path_value or '').stem
    # 文件名语义补充：把空格/下划线/连字符拆开，辅助匹配“特殊命名”的教学素材。
    chunks = re.split(r"[\s_\-]+", stem)
    terms: List[str] = []
    for chunk in chunks:
        part = _safe_text(chunk)
        if len(part) >= 2:
            terms.append(part)
    return list(dict.fromkeys(terms))


def _read_text_file(path: Path) -> str:
    """Read text file with lightweight encoding fallback."""
    raw = path.read_bytes()
    for enc in ('utf-8-sig', 'utf-8', 'gb18030'):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode('utf-8', errors='ignore')


def _split_text_paragraphs(text: str, max_paragraphs: int = 40, max_len: int = 260) -> List[str]:
    """Split text into manageable paragraphs for insertion and evidence linking."""
    normalized = _safe_text(text)
    if not normalized:
        return []

    coarse = [seg.strip() for seg in re.split(r"\n+|\r+", text) if _safe_text(seg)]
    if not coarse:
        coarse = [normalized]

    chunks: List[str] = []
    for para in coarse:
        para = _safe_text(para)
        if not para:
            continue
        if len(para) <= max_len:
            chunks.append(para)
            continue
        for sentence in re.split(r"(?<=[。！？；.!?;])", para):
            sentence = _safe_text(sentence)
            if not sentence:
                continue
            if len(sentence) <= max_len:
                chunks.append(sentence)
                continue
            for i in range(0, len(sentence), max_len):
                chunks.append(sentence[i:i + max_len])

    deduped: List[str] = []
    seen = set()
    for item in chunks:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
        if len(deduped) >= max_paragraphs:
            break
    return deduped


def _extract_existing_entities_from_text(session: Any, query_engine: Any, paragraphs: List[str]) -> List[str]:
    """Find existing entities related to text by exact mention + query term resolution."""
    found: List[str] = []
    seen = set()
    snippets = [p for p in paragraphs if p][:24]
    if not snippets:
        return []

    try:
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE size(coalesce(e.name, '')) >= 2
            WITH e, $paragraphs AS paragraphs
            WHERE any(p IN paragraphs WHERE p CONTAINS e.name)
            RETURN DISTINCT e.name AS name
            LIMIT 120
            """,
            paragraphs=snippets,
        )
        for row in result:
            name = _safe_text(row.get('name'))
            if name and name not in seen:
                seen.add(name)
                found.append(name)
    except Exception as e:
        logger.warning(f"文本实体显式匹配失败: {e}")

    resolver = getattr(query_engine, 'resolve_entity_name', None)
    extractor = getattr(query_engine, 'extract_query_terms', None)
    if callable(resolver) and callable(extractor):
        for para in snippets[:16]:
            for term in extractor(para)[:8]:
                resolved, _alts = resolver(term)
                resolved = _safe_text(resolved)
                if not resolved or resolved in seen:
                    continue
                exists = session.run(
                    "MATCH (e:Entity {name: $name}) RETURN COUNT(e) AS cnt",
                    name=resolved,
                ).single()
                if int((exists or {}).get('cnt', 0)) > 0:
                    seen.add(resolved)
                    found.append(resolved)
                if len(found) >= 120:
                    break
            if len(found) >= 120:
                break

    return found

# 模板已迁移到 templates/index.html 与 templates/annotate.html



def _annotation_enabled() -> bool:
    return bool(app.config.get('ENABLE_MANUAL_ANNOTATION', False))


def _data_root() -> Path:
    return Path(app.root_path) / 'data'


def _annotation_file() -> Path:
    path = Path(app.root_path) / 'kg_output' / 'manual_video_annotations.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_entity_choices() -> Dict[str, List[str]]:
    datamain = Path(app.root_path) / 'BERT_cn' / 'datamain.txt'
    if not datamain.exists():
        return {'computer_entities': [], 'ideology_entities': []}

    content = datamain.read_text(encoding='utf-8')
    computer_entities: List[str] = []
    ideology_entities: List[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or '=' not in line:
            continue
        key, value = [part.strip() for part in line.split('=', 1)]
        if key == 'COMPUTER_LABELS':
            computer_entities = [str(item).strip() for item in ast.literal_eval(value) if str(item).strip()]
        elif key == 'IDEOLOGY_LABELS':
            ideology_entities = [str(item).strip() for item in ast.literal_eval(value) if str(item).strip()]
    return {
        'computer_entities': computer_entities,
        'ideology_entities': ideology_entities,
    }


def _as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [_safe_text(item) for item in value if _safe_text(item)]
    text = _safe_text(value)
    return [text] if text else []


def _normalize_multi_values(payload: Dict[str, Any], list_key: str, single_key: str) -> List[str]:
    """兼容数组和单值字段，输出去重后的字符串列表。"""
    values: List[str] = []
    raw_list = payload.get(list_key)
    if isinstance(raw_list, list):
        values.extend(_safe_text(item) for item in raw_list)
    elif raw_list not in (None, ''):
        values.append(_safe_text(raw_list))

    single_value = _safe_text(payload.get(single_key))
    if single_value:
        values.append(single_value)

    normalized: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _list_videos_for_annotation() -> List[Dict[str, str]]:
    videos: List[Dict[str, str]] = []
    data_root = _data_root()
    suffixes = {'.mp4', '.avi', '.mov', '.mkv', '.ts', '.m2ts'}
    for subdir in ('video', 'video_fixed', 'uploads/video', 'clips'):
        base = data_root / subdir
        if not base.exists():
            continue
        for path in sorted(base.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            relative = path.relative_to(data_root).as_posix()
            videos.append({
                'name': path.name,
                'relative_path': relative,
                'url': f"/media/{relative}",
                'media_type': 'video',
                'source_media_type': 'video',
            })
    dedup: Dict[str, Dict[str, str]] = {}
    for item in videos:
        dedup[item['relative_path']] = item
    return list(dedup.values())


def _media_url_for_path(path_value: Optional[str]) -> str:
    """把本地文件路径转换成可由 `/media/...` 访问的 URL。"""
    if not path_value:
        return ''

    try:
        candidate = Path(str(path_value))
        if not candidate.is_absolute():
            candidate = (Path(app.root_path) / candidate).resolve()
        data_root = _data_root().resolve()
        try:
            relative = candidate.resolve().relative_to(data_root)
            return f"/media/{relative.as_posix()}"
        except Exception:
            return str(path_value)
    except Exception:
        return str(path_value)


def _preferred_playback_url(item: Dict) -> str:
    """统一播放URL策略：优先 clip_path，其次 clip_url，再回退 media_url/path。"""
    clip_path = _safe_text(item.get('clip_path'))
    clip_url = _safe_text(item.get('clip_url'))
    media_url = _safe_text(item.get('media_url'))
    path_value = _safe_text(item.get('path'))

    if clip_path:
        return _media_url_for_path(clip_path)
    if clip_url:
        return clip_url
    if media_url:
        return media_url
    if path_value:
        return _media_url_for_path(path_value)
    return ''


def _guess_media_type(labels: List[str], media_type: str, entity_name: str) -> str:
    media = _safe_text(media_type).lower()
    if media:
        return media
    label_set = {str(label) for label in (labels or [])}
    if 'Video' in label_set:
        return 'video'
    if 'Image' in label_set:
        return 'image'
    if str(entity_name).lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return 'video'
    return 'text'


class Neo4jQuery:
    """Neo4j数据库查询器"""

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 database: str = "neo4j",
                 semantic_model_dir: Optional[str] = None,
                 semantic_model_profile: str = "bert_cn_base",
                 semantic_device_mode: str = "cpu"):
        self.uri = uri
        self.database = database or "neo4j"
        self.semantic_model_dir = semantic_model_dir
        self.semantic_model_profile = str(semantic_model_profile or "bert_cn_base").strip().lower()
        self.semantic_device_mode = str(semantic_device_mode or "cpu").strip().lower()
        self.semantic = SemanticScorer(
            model_dir=semantic_model_dir,
            model_name=self.semantic_model_profile,
            device_mode=self.semantic_device_mode,
        ) if SemanticScorer else None
        self.semantic_model_resolved = str(getattr(self.semantic, 'model_dir', '') or '')
        self.semantic_model_available = bool(getattr(self.semantic, 'available', False))
        self.video_editor = VideoEditor() if VideoEditor else None
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self._test_connection()
        except Exception as e:
            logger.error(f"连接失败: {str(e)}")
            self.driver = None

    def _test_connection(self):
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1 AS num").single()
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"连接测试失败: {str(e)}")
            raise

    def _node_text(self, record: Dict) -> str:
        parts = [
            record.get('entity', ''),
            record.get('caption', ''),
            record.get('summary', ''),
            record.get('title', ''),
            record.get('media_type', ''),
            record.get('relation', ''),
            record.get('source_file', ''),
            ' '.join(record.get('stage_tags', []) or []),
            ' '.join(record.get('course_tags', []) or []),
            ' '.join(record.get('ideology_tags', []) or []),
            ' '.join(record.get('teaching_objectives', []) or []),
            ' '.join(record.get('knowledge_points', []) or []),
            ' '.join(record.get('videos', []) or []),
        ]
        return _safe_text(' '.join(str(part) for part in parts if part))

    @staticmethod
    def _normalize_entity_query(text: Optional[str]) -> str:
        return re.sub(r"\s+", "", _safe_text(text).lower())

    def extract_query_terms(self, query: str) -> List[str]:
        """把自然语言问题拆成检索词，并尝试匹配图中已有实体名。"""
        cleaned = _safe_text(query)
        compact = self._normalize_entity_query(cleaned)
        if not compact:
            return []

        terms: List[str] = [cleaned]
        seen = {cleaned}

        normalized = re.sub(r"[？?！!。,.，；;:：()（）\[\]【】\"'“”]", " ", cleaned)
        stop_words = [
            "应该", "注意", "什么", "如何", "怎么", "请问", "有哪些", "需要", "可以", "关于", "一下", "一下子",
            "的问题", "问题", "建议", "吗", "呢", "呀", "啊", "以及", "和", "与", "及", "的", "了", "在",
        ]
        for word in stop_words:
            normalized = normalized.replace(word, " ")

        chunks = [item.strip() for item in re.split(r"\s+", normalized) if item.strip()]
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            if chunk not in seen:
                terms.append(chunk)
                seen.add(chunk)

        # 规则抽取：学段与核心主题词（对“长句问法”做稳健拆分）
        stage_hits = re.findall(r"大[一二三四]|高[一二三]|初[一二三]|小学|初中|高中|大学|高职|基础|初级|入门|导论|概论|零基础", compact)
        for token in stage_hits:
            if token not in seen:
                terms.append(token)
                seen.add(token)

        domain_lexicon = [
            "规范化", "编程", "编程规范", "代码规范", "程序设计", "算法", "数据结构", "数据库", "操作系统", "计算机网络", "软件工程",
            "面向对象", "实验", "思政", "马克思主义", "创新理论", "文化建设", "核心价值体系", "大一", "基础", "初级", "入门", "导论", "概论",
        ]
        for token in domain_lexicon:
            if token in compact and token not in seen:
                terms.append(token)
                seen.add(token)

        # 借助图库实体做子串命中：例如“大一规范化编程...”可命中“大一/规范化/编程”
        if self.driver:
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(
                        """
                        MATCH (n:Entity)
                        WITH n, coalesce(n.name, '') AS name,
                             replace(coalesce(n.name, ''), ' ', '') AS compact_name
                        WHERE size(compact_name) >= 2
                          AND size(compact_name) <= 16
                          AND toLower($query) CONTAINS toLower(compact_name)
                        RETURN name
                        ORDER BY size(name) ASC
                        LIMIT 20
                        """,
                        query=compact,
                    )
                    for record in result:
                        name = _safe_text(record.get('name'))
                        if len(name) >= 2 and name not in seen:
                            terms.append(name)
                            seen.add(name)
            except Exception as e:
                logger.warning(f"查询词拆分阶段的实体提示失败: {e}")

        return terms[:10]

    def resolve_entity_name(self, entity: str) -> Tuple[str, List[str]]:
        """Resolve user input to an existing node name (exact -> contains -> fallback)."""
        if not self.driver:
            cleaned = _safe_text(entity)
            return cleaned, []

        cleaned = _safe_text(entity)
        compact = self._normalize_entity_query(cleaned)
        if not compact:
            return cleaned, []

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (n:Entity)
                    WITH n, coalesce(n.name, '') AS name,
                         toLower(replace(coalesce(n.name, ''), ' ', '')) AS compact_name,
                         toLower($query) AS q
                    WHERE compact_name = q
                       OR compact_name CONTAINS q
                       OR q CONTAINS compact_name
                    RETURN name,
                           CASE
                               WHEN compact_name = q THEN 3
                               WHEN compact_name CONTAINS q THEN 2
                               ELSE 1
                           END AS score
                    ORDER BY score DESC, size(name) ASC
                    LIMIT 8
                    """,
                    query=compact,
                )
                candidates = [record['name'] for record in result if record.get('name')]

                if candidates:
                    return candidates[0], candidates[1:]

                # 语义兜底：当精确/包含匹配失败时，用本地 BERT 对节点文本做排序，
                # 让“大一编程规范”这类复合短语也能落到最相近的教学节点上。
                if self.semantic:
                    candidate_map: Dict[str, Dict[str, Any]] = {}
                    semantic_result = session.run(
                        """
                        MATCH (n:Entity)
                        RETURN coalesce(n.name, '') AS name,
                               labels(n) AS labels,
                               coalesce(n.summary, '') AS summary,
                               coalesce(n.caption, '') AS caption,
                               coalesce(n.source_file, '') AS source_file,
                               coalesce(n.stage_tags, []) AS stage_tags,
                               coalesce(n.course_tags, []) AS course_tags,
                               coalesce(n.ideology_tags, []) AS ideology_tags,
                               coalesce(n.teaching_objectives, []) AS teaching_objectives,
                               coalesce(n.knowledge_points, []) AS knowledge_points,
                               coalesce(n.videos, []) AS videos
                        LIMIT 800
                        """
                    )
                    for record in semantic_result:
                        name = _safe_text(record.get('name'))
                        if not name:
                            continue
                        candidate_map[name] = {
                            'name': name,
                            'labels': record.get('labels') or [],
                            'summary': record.get('summary', ''),
                            'caption': record.get('caption', ''),
                            'source_file': record.get('source_file', ''),
                            'stage_tags': record.get('stage_tags', []),
                            'course_tags': record.get('course_tags', []),
                            'ideology_tags': record.get('ideology_tags', []),
                            'teaching_objectives': record.get('teaching_objectives', []),
                            'knowledge_points': record.get('knowledge_points', []),
                            'videos': record.get('videos', []),
                        }

                    ranked = self.semantic.rank_candidates(cleaned, candidate_map, top_k=8)
                    semantic_candidates = [item['name'] for item in ranked if item.get('name') and float(item.get('score', 0.0)) >= 0.34]
                    if semantic_candidates:
                        return semantic_candidates[0], semantic_candidates[1:]
        except Exception as e:
            logger.warning(f"实体解析失败，回退原始输入: {e}")

        return cleaned, []

    def _stage_match_bonus(self, text: str, stage: Optional[str]) -> float:
        if not stage:
            return 0.0
        text = _safe_text(text)
        stage = _safe_text(stage)
        if not text or not stage:
            return 0.0
        aliases = {
            '大一': ['大一', '大1', '基础', '初级', '入门', '导论', '概论', '零基础', '启蒙'],
            '小学': ['小学', '基础', '启蒙'],
            '初中': ['初中', '中学', '基础'],
            '高中': ['高中', '中学', '综合'],
            '大学': ['大学', '本科', '高等教育'],
            '高职': ['高职', '职业', '技能'],
            '本科': ['本科', '大学', '高等教育'],
        }
        tokens = aliases.get(stage, [stage])
        return 0.12 if any(token in text for token in tokens) else 0.0

    def _duration_bonus(self, record: Dict, duration: Optional[float]) -> float:
        if not duration:
            return 0.0
        fps = record.get('fps') or 0
        frame_count = record.get('frame_count') or 0
        if fps and frame_count:
            estimated = frame_count / max(float(fps), 1e-6)
            delta = abs(estimated - float(duration))
            return max(0.0, 0.12 * (1.0 - min(delta / max(float(duration), estimated, 1.0), 1.0)))
        return 0.0

    def _course_bonus(self, text: str, course: Optional[str]) -> float:
        if not course:
            return 0.0
        return 0.15 if _safe_text(course) in _safe_text(text) else 0.0

    def _prerequisite_bonus(self, text: str, prerequisites: Optional[List[str]]) -> float:
        if not prerequisites:
            return 0.0
        hits = sum(1 for item in prerequisites if item and item in text)
        return min(0.12, 0.04 * hits)

    def _objective_bonus(self, text: str, objective: Optional[str]) -> float:
        if not objective:
            return 0.0
        if self.semantic:
            return min(0.16, 0.16 * self.semantic.similarity(objective, text))
        return 0.08 if _safe_text(objective) in _safe_text(text) else 0.0

    def _tag_bonus(self, query_value: Optional[str], tags: Any, per_hit: float = 0.04, cap: float = 0.12) -> float:
        if not query_value or not tags:
            return 0.0
        query_text = _safe_text(query_value)
        if not query_text:
            return 0.0
        hits = 0
        for tag in _as_text_list(tags):
            if not tag:
                continue
            if query_text in tag or tag in query_text:
                hits += 1
        return min(cap, per_hit * hits)

    def _build_risk_and_suggestion(self, item: Dict,
                                   stage: Optional[str],
                                   duration: Optional[float],
                                   course: Optional[str],
                                   prerequisites: Optional[List[str]]) -> Tuple[str, str]:
        labels = item.get('labels', []) or []
        text = self._node_text(item)
        risks = []
        suggests = []

        if stage and self._stage_match_bonus(text, stage) < 0.05:
            risks.append('学段匹配度较低')
            suggests.append('建议教师二次筛选学段适配内容')

        if duration and self._duration_bonus(item, duration) < 0.03:
            risks.append('时长不确定或超出建议')
            suggests.append('建议按课时裁剪后使用')

        if course and self._course_bonus(text, course) < 0.05:
            risks.append('课程主题相关性一般')
            suggests.append('建议搭配课程导语或补充材料')

        prereq_hits = self._prerequisite_bonus(text, prerequisites)
        if prerequisites and prereq_hits < 0.04:
            risks.append('先修知识覆盖不足')
            suggests.append('建议先补充先修知识点')

        if 'Video' in labels:
            suggests.append('适合课堂导入或案例讲解')
        elif 'TeachingCase' in labels:
            suggests.append('适合课中讨论与思政融入')
        elif 'Image' in labels:
            suggests.append('适合概念可视化说明')

        risk_note = '；'.join(risks) if risks else '风险可控'
        suggested_use = '；'.join(dict.fromkeys(suggests)) if suggests else '可直接用于课堂推荐'
        return risk_note, suggested_use

    def _score_recommendation(self, item: Dict, entity: str,
                              stage: Optional[str] = None,
                              duration: Optional[float] = None,
                              focus: Optional[str] = None,
                              course: Optional[str] = None,
                              prerequisites: Optional[List[str]] = None,
                              objective: Optional[str] = None,
                              audience: Optional[str] = None,
                              constraints: Optional[str] = None) -> Tuple[float, str, str, str]:
        text = self._node_text(item)
        relation = item.get('relation', '')
        labels = item.get('labels', []) or []
        similarity = float(item.get('similarity') or 0.0)
        relation_bonus = {
            'SIMILAR': 0.08,
            'RELATED': 0.12,
            'MENTIONS': 0.10,
            'LINKS_TO_CASE': 0.16,
            'MEDIA_LINKED_IMAGE': 0.08,
            'MEDIA_LINKED_VIDEO': 0.08,
            'CASE_SUPPORTS': 0.15,
            'COMPUTER_REFLECTS_IDEOLOGY': 0.14,
        }.get(relation, 0.05)
        label_bonus = 0.12 if 'TeachingCase' in labels else 0.07 if 'Media' in labels else 0.04

        stage_bonus = self._stage_match_bonus(text, stage)
        duration_bonus = self._duration_bonus(item, duration)
        course_bonus = self._course_bonus(text, course)
        prereq_bonus = self._prerequisite_bonus(text, prerequisites)
        objective_bonus = self._objective_bonus(text, objective)

        focus_bonus = 0.0
        if focus:
            focus_text = ' '.join(_as_text_list(item.get('ideology_tags')) + _as_text_list(item.get('teaching_objectives')) + [text])
            if self.semantic:
                focus_bonus = min(0.18, 0.18 * self.semantic.similarity(focus, focus_text))
            else:
                focus_bonus = 0.10 if _safe_text(focus) in focus_text else 0.0
            focus_bonus += self._tag_bonus(focus, item.get('ideology_tags'), per_hit=0.05, cap=0.12)

        audience_bonus = 0.0
        if audience:
            audience_text = ' '.join([text] + _as_text_list(item.get('stage_tags')) + _as_text_list(item.get('course_tags')))
            if self.semantic:
                audience_bonus = min(0.10, 0.10 * self.semantic.similarity(audience, audience_text))
            else:
                audience_bonus = 0.06 if _safe_text(audience) in audience_text else 0.0

        constraints_bonus = 0.0
        if constraints:
            constraints_text = ' '.join([text] + _as_text_list(item.get('teaching_objectives')) + _as_text_list(item.get('ideology_tags')))
            if self.semantic:
                constraints_bonus = min(0.08, 0.08 * self.semantic.similarity(constraints, constraints_text))
            else:
                constraints_bonus = 0.04 if _safe_text(constraints) in constraints_text else 0.0

        scenario_tag_bonus = 0.0
        scenario_tag_bonus += self._tag_bonus(stage, item.get('stage_tags'), per_hit=0.05, cap=0.12)
        scenario_tag_bonus += self._tag_bonus(course, item.get('course_tags'), per_hit=0.05, cap=0.12)
        scenario_tag_bonus += self._tag_bonus(objective, item.get('teaching_objectives'), per_hit=0.04, cap=0.10)

        media_bonus = 0.0
        if 'TeachingCase' in labels:
            media_bonus = 0.15
        elif 'Video' in labels:
            media_bonus = 0.10
        elif 'Image' in labels:
            media_bonus = 0.06
        elif 'Media' in labels:
            media_bonus = 0.08

        score = 0.12 * similarity + relation_bonus + label_bonus
        score = min(1.0, score + stage_bonus + duration_bonus + course_bonus + prereq_bonus + objective_bonus)
        score = min(1.0, score + focus_bonus + audience_bonus + constraints_bonus + scenario_tag_bonus + media_bonus)

        reason_parts = [f"关系:{relation}"]
        if 'TeachingCase' in labels:
            reason_parts.append('教学案例节点')
        elif 'Video' in labels:
            reason_parts.append('视频媒体')
        elif 'Image' in labels:
            reason_parts.append('图像媒体')
        if stage:
            reason_parts.append(f"适配学段:{stage}")
        if item.get('stage_tags'):
            reason_parts.append(f"节点学段:{'/'.join(_as_text_list(item.get('stage_tags'))[:3])}")
        if duration:
            reason_parts.append(f"时长参考:{duration}")
        if focus:
            reason_parts.append(f"思政侧重:{focus}")
        if item.get('ideology_tags'):
            reason_parts.append(f"思政标签:{'/'.join(_as_text_list(item.get('ideology_tags'))[:3])}")
        if course:
            reason_parts.append(f"课程:{course}")
        if item.get('course_tags'):
            reason_parts.append(f"课程标签:{'/'.join(_as_text_list(item.get('course_tags'))[:3])}")
        if prerequisites:
            reason_parts.append(f"先修:{'/'.join(prerequisites[:3])}")
        if objective:
            reason_parts.append(f"目标:{objective}")
        if item.get('teaching_objectives'):
            reason_parts.append(f"节点目标:{'/'.join(_as_text_list(item.get('teaching_objectives'))[:3])}")
        if audience:
            reason_parts.append(f"对象:{audience}")
        if constraints:
            reason_parts.append(f"约束:{constraints}")

        risk_note, suggested_use = self._build_risk_and_suggestion(
            item,
            stage=stage,
            duration=duration,
            course=course,
            prerequisites=prerequisites,
        )
        return round(score, 4), '；'.join(reason_parts), risk_note, suggested_use

    def _query_two_hop_candidates(self, entity: str, limit: int = 120) -> List[Dict]:
        if not self.driver:
            return []
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (a {name: $entity})-[r1]-(m)-[r2]-(b)
                    WHERE coalesce(b.name, '') <> ''
                      AND coalesce(m.name, '') <> ''
                      AND b.name <> $entity
                      AND m.name <> b.name
                    RETURN b.name as entity,
                           type(r2) as relation,
                           coalesce(r2.similarity, 0.0) as similarity,
                           coalesce(r2.caption, '') as caption,
                           labels(b) as labels,
                           coalesce(b.summary, '') as summary,
                           coalesce(b.path, '') as path,
                           coalesce(b.clip_path, '') as clip_path,
                           coalesce(b.clip_url, '') as media_url,
                           coalesce(b.media_type, '') as media_type,
                           coalesce(b.source_file, '') as source_file,
                           coalesce(b.stage_tags, []) as stage_tags,
                           coalesce(b.course_tags, []) as course_tags,
                           coalesce(b.ideology_tags, []) as ideology_tags,
                           coalesce(b.teaching_objectives, []) as teaching_objectives,
                           coalesce(b.chapter_count, 0) as chapter_count,
                           coalesce(b.isbn, '') as isbn,
                           coalesce(b.frame_count, 0) as frame_count,
                           coalesce(b.fps, 0) as fps,
                           coalesce(b.width, 0) as width,
                           coalesce(b.height, 0) as height,
                           m.name as via_entity,
                           2 as hop
                    LIMIT $limit
                """, entity=entity, limit=max(20, limit))
                items = []
                for record in result:
                    item = dict(record)
                    item['media'] = item.get('media_type', '')
                    if not item.get('clip_url') and item.get('clip_path'):
                        item['clip_url'] = _media_url_for_path(item.get('clip_path', ''))
                    item['media_url'] = _preferred_playback_url(item)
                    items.append(item)
                return items
        except Exception as e:
            logger.warning(f"二跳候选查询失败: {e}")
            return []

    def query_connected_entities(self, entity: str) -> List[Dict]:
        if not self.driver:
            return []

        try:
            query_terms = self.extract_query_terms(entity)
            all_items: List[Dict] = []
            visited_entities = set()
            with self.driver.session(database=self.database) as session:
                for query_term in query_terms:
                    resolved_entity, entity_candidates = self.resolve_entity_name(query_term)
                    search_entities = [resolved_entity] + [item for item in entity_candidates if item and item != resolved_entity]
                    for search_entity in search_entities:
                        if not search_entity or search_entity in visited_entities:
                            continue
                        visited_entities.add(search_entity)
                        result = session.run("""
                        MATCH (a {name: $entity})-[r]-(b)
                        WHERE coalesce(b.name, '') <> ''
                        RETURN b.name as entity,
                               type(r) as relation,
                               coalesce(r.caption, '') as caption,
                               coalesce(r.similarity, 0.0) as similarity,
                               labels(b) as labels,
                               coalesce(b.summary, '') as summary,
                               coalesce(b.path, '') as path,
                               coalesce(b.clip_path, '') as clip_path,
                               coalesce(b.clip_url, '') as clip_url,
                               coalesce(b.media_url, '') as media_url,
                               coalesce(b.media_type, '') as media_type,
                               coalesce(b.source_file, '') as source_file,
                               coalesce(b.stage_tags, []) as stage_tags,
                               coalesce(b.course_tags, []) as course_tags,
                               coalesce(b.ideology_tags, []) as ideology_tags,
                               coalesce(b.teaching_objectives, []) as teaching_objectives,
                               coalesce(b.chapter_count, 0) as chapter_count,
                               coalesce(b.isbn, '') as isbn
                        ORDER BY similarity DESC
                        LIMIT 30
                    """, entity=search_entity)

                        for record in result:
                            item = dict(record)
                            item['media'] = _guess_media_type(item.get('labels') or [], item.get('media_type', ''), item.get('entity', ''))
                            if not item.get('clip_url') and item.get('clip_path'):
                                item['clip_url'] = _media_url_for_path(item.get('clip_path', ''))
                            item['media_url'] = _preferred_playback_url(item)
                            item['query_term'] = query_term
                            item['resolved_entity'] = search_entity
                            all_items.append(item)

            dedup: Dict[str, Dict] = {}
            for item in all_items:
                key = f"{item.get('entity','')}|{item.get('relation','')}|{item.get('media_url','')}"
                prev = dedup.get(key)
                if prev is None or float(item.get('similarity', 0.0)) > float(prev.get('similarity', 0.0)):
                    dedup[key] = item

            merged = list(dedup.values())
            merged.sort(key=lambda x: float(x.get('similarity', 0.0)), reverse=True)
            return merged[:50]
        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            return []

    def get_subgraph_data(self, entity: str, hops: int = 2, node_limit: int = 180, edge_limit: int = 400) -> Dict:
        if not self.driver:
            return {"nodes": [], "links": [], "stats": {"total_nodes": 0, "total_edges": 0}, "charts": {"relation_distribution": {}, "media_distribution": {}}}

        hops = 2 if int(hops) >= 2 else 1
        try:
            resolved_entity, _ = self.resolve_entity_name(entity)
            with self.driver.session(database=self.database) as session:
                node_result = session.run(
                    """
                    MATCH (c:Entity {name: $entity})
                    OPTIONAL MATCH p=(c)-[*1..2]-(n:Entity)
                    WITH collect(DISTINCT c) + collect(DISTINCT n) AS raw_nodes
                    UNWIND raw_nodes AS node
                    WITH DISTINCT node LIMIT $node_limit
                    RETURN node.name AS name,
                           labels(node) AS labels,
                           coalesce(node.media_type, '') AS media_type,
                           coalesce(node.summary, '') AS summary,
                           coalesce(node.path, '') AS path,
                           coalesce(node.media_url, '') AS media_url,
                           coalesce(node.caption, '') AS caption,
                           coalesce(node.knowledge_points, []) AS knowledge_points,
                           coalesce(node.knowledge_point_count, 0) AS knowledge_point_count,
                           coalesce(node.videos, []) AS videos,
                           coalesce(node.video_count, 0) AS video_count,
                           COUNT { (node)--() } AS connections
                    """,
                    entity=resolved_entity,
                    node_limit=max(20, int(node_limit)),
                )

                nodes = []
                node_names = []
                media_distribution: Dict[str, int] = {}
                for record in node_result:
                    name = record.get('name')
                    if not name:
                        continue
                    labels = record.get('labels') or []
                    media = _guess_media_type(labels, record.get('media_type', ''), name)
                    node_type = 'entity'
                    if 'TeachingCase' in labels:
                        node_type = 'case'
                    elif media == 'video':
                        node_type = 'video'
                    elif media == 'image':
                        node_type = 'image'
                    elif 'IdeologyElement' in labels:
                        node_type = 'ideology'
                    elif 'KnowledgePoint' in labels:
                        node_type = 'knowledge'
                    nodes.append({
                        'id': name,
                        'connections': record.get('connections', 0),
                        'type': node_type,
                        'labels': labels,
                        'media_type': media,
                        'summary': record.get('summary', ''),
                        'path': record.get('path', ''),
                        'media_url': record.get('media_url', ''),
                        'caption': record.get('caption', ''),
                        'knowledge_points': record.get('knowledge_points', []),
                        'knowledge_point_count': record.get('knowledge_point_count', 0),
                        'videos': record.get('videos', []),
                        'video_count': record.get('video_count', 0),
                    })
                    node_names.append(name)
                    media_distribution[media] = media_distribution.get(media, 0) + 1

                if not node_names:
                    return {"nodes": [], "links": [], "stats": {"total_nodes": 0, "total_edges": 0}, "charts": {"relation_distribution": {}, "media_distribution": {}}}

                edge_result = session.run(
                    """
                    MATCH (a:Entity)-[r]->(b:Entity)
                    WHERE a.name IN $names AND b.name IN $names
                    RETURN a.name AS source,
                           b.name AS target,
                           type(r) AS type,
                           coalesce(r.similarity, 0.0) AS similarity,
                           coalesce(r.caption, '') AS caption
                    LIMIT $edge_limit
                    """,
                    names=node_names,
                    edge_limit=max(50, int(edge_limit)),
                )

                relation_distribution: Dict[str, int] = {}
                links = []
                for record in edge_result:
                    rel_type = record.get('type', 'CONNECTED')
                    relation_distribution[rel_type] = relation_distribution.get(rel_type, 0) + 1
                    links.append({
                        'source': record.get('source'),
                        'target': record.get('target'),
                        'type': rel_type,
                        'strength': max(float(record.get('similarity') or 0.0), 0.3),
                        'caption': record.get('caption', ''),
                    })

                return {
                    'nodes': nodes,
                    'links': links,
                    'stats': {
                        'total_nodes': len(nodes),
                        'total_edges': len(links),
                        'center': resolved_entity,
                        'hops': hops,
                    },
                    'charts': {
                        'relation_distribution': relation_distribution,
                        'media_distribution': media_distribution,
                    },
                }
        except Exception as e:
            logger.error(f"获取子图失败: {str(e)}")
            return {"nodes": [], "links": [], "stats": {"total_nodes": 0, "total_edges": 0}, "charts": {"relation_distribution": {}, "media_distribution": {}}}

    def get_graph_data(self) -> Dict:
        if not self.driver:
            return {"nodes": [], "links": [], "stats": {"total_nodes": 0, "total_edges": 0}}

        try:
            with self.driver.session(database=self.database) as session:
                nodes_result = session.run("""
                    MATCH (n:Entity)
                    RETURN n.name as name,
                           labels(n) as labels,
                           coalesce(n.media_type, '') as media_type,
                           coalesce(n.summary, '') as summary,
                           coalesce(n.path, '') as path,
                           coalesce(n.media_url, '') as media_url,
                           coalesce(n.caption, '') as caption,
                           coalesce(n.knowledge_points, []) as knowledge_points,
                           coalesce(n.knowledge_point_count, 0) as knowledge_point_count,
                           coalesce(n.videos, []) as videos,
                           coalesce(n.video_count, 0) as video_count,
                           COUNT { (n)--() } as connections
                    LIMIT 300
                """)

                nodes = []
                for record in nodes_result:
                    labels = record['labels'] or []
                    node_type = 'entity'
                    if 'TeachingCase' in labels:
                        node_type = 'case'
                    elif 'Video' in labels:
                        node_type = 'video'
                    elif 'Image' in labels:
                        node_type = 'image'
                    elif 'IdeologyElement' in labels:
                        node_type = 'ideology'
                    elif 'KnowledgePoint' in labels:
                        node_type = 'knowledge'
                    nodes.append({
                        'id': record['name'],
                        'connections': record['connections'],
                        'type': node_type,
                        'labels': labels,
                        'media_type': record['media_type'],
                        'summary': record['summary'],
                        'path': record['path'],
                        'media_url': record.get('media_url', ''),
                        'caption': record['caption'],
                        'knowledge_points': record.get('knowledge_points', []),
                        'knowledge_point_count': record.get('knowledge_point_count', 0),
                        'videos': record.get('videos', []),
                        'video_count': record.get('video_count', 0),
                    })

                edges_result = session.run("""
                    MATCH (a:Entity)-[r]->(b:Entity)
                    RETURN a.name as source,
                           b.name as target,
                           type(r) as type,
                           coalesce(r.similarity, 0.0) as similarity,
                           coalesce(r.caption, '') as caption
                    LIMIT 500
                """)

                links = []
                for record in edges_result:
                    links.append({
                        'source': record['source'],
                        'target': record['target'],
                        'type': record['type'],
                        'strength': max(float(record['similarity'] or 0.0), 0.3),
                        'caption': record['caption'],
                    })

                total_nodes = session.run("MATCH (n:Entity) RETURN COUNT(n) as total_nodes").single()["total_nodes"]
                total_edges = session.run("MATCH ()-[r]->() RETURN COUNT(r) as total_edges").single()["total_edges"]

                return {
                    'nodes': nodes,
                    'links': links,
                    'stats': {
                        'total_nodes': total_nodes,
                        'total_edges': total_edges,
                    }
                }
        except Exception as e:
            logger.error(f"获取图数据失败: {str(e)}")
            return {"nodes": [], "links": [], "stats": {"total_nodes": 0, "total_edges": 0}}

    def query_similar_and_related_entities(self, entity: str, data_dir: str,
                                           stage: Optional[str] = None,
                                           duration: Optional[float] = None,
                                           focus: Optional[str] = None,
                                           course: Optional[str] = None,
                                           prerequisites: Optional[List[str]] = None,
                                           objective: Optional[str] = None,
                                           audience: Optional[str] = None,
                                           constraints: Optional[str] = None,
                                           top_k: int = 10) -> Dict:
        if not self.driver:
            return {"similar": [], "related": [], "videos": [], "recommendations": []}

        try:
            resolved_entity, entity_candidates = self.resolve_entity_name(entity)
            with self.driver.session(database=self.database) as session:
                search_entities = [resolved_entity] + [item for item in entity_candidates if item and item != resolved_entity]
                items: List[Dict] = []
                for search_entity in search_entities:
                    result = session.run("""
                    MATCH (a {name: $entity})-[r]-(b)
                    WHERE coalesce(b.name, '') <> ''
                    RETURN b.name as entity,
                           type(r) as relation,
                           coalesce(r.similarity, 0.0) as similarity,
                           coalesce(r.caption, '') as caption,
                           labels(b) as labels,
                           coalesce(b.summary, '') as summary,
                           coalesce(b.path, '') as path,
                           coalesce(b.clip_path, '') as clip_path,
                           coalesce(b.clip_url, '') as media_url,
                           coalesce(b.media_type, '') as media_type,
                           coalesce(b.source_file, '') as source_file,
                           coalesce(b.stage_tags, []) as stage_tags,
                           coalesce(b.course_tags, []) as course_tags,
                           coalesce(b.ideology_tags, []) as ideology_tags,
                           coalesce(b.teaching_objectives, []) as teaching_objectives,
                           coalesce(b.knowledge_points, []) as knowledge_points,
                           coalesce(b.knowledge_point_count, 0) as knowledge_point_count,
                           coalesce(b.videos, []) as videos,
                           coalesce(b.video_count, 0) as video_count,
                           coalesce(b.chapter_count, 0) as chapter_count,
                           coalesce(b.isbn, '') as isbn,
                           coalesce(b.frame_count, 0) as frame_count,
                           coalesce(b.fps, 0) as fps,
                           coalesce(b.width, 0) as width,
                           coalesce(b.height, 0) as height,
                           coalesce(b.image_paths, []) as image_paths,
                           coalesce(b.image_captions, []) as image_captions,
                           coalesce(b.image_ocr_texts, []) as image_ocr_texts
                    LIMIT 100
                """, entity=search_entity)

                    items.extend(dict(record) for record in result)
                    items.extend(self._query_two_hop_candidates(search_entity, limit=max(50, top_k * 12)))

                dedup_items: Dict[str, Dict] = {}
                for item in items:
                    key = item.get('entity')
                    if not key:
                        continue
                    prev = dedup_items.get(key)
                    if prev is None or float(item.get('similarity', 0.0)) > float(prev.get('similarity', 0.0)):
                        dedup_items[key] = item
                items = list(dedup_items.values())

                similar_entities = []
                related_entities = []
                videos = []
                images = []
                recommendations = []

                for item in items:
                    score, reason, risk_note, suggested_use = self._score_recommendation(
                        item,
                        entity,
                        stage=stage,
                        duration=duration,
                        focus=focus,
                        course=course,
                        prerequisites=prerequisites,
                        objective=objective,
                        audience=audience,
                        constraints=constraints,
                    )
                    item['score'] = score
                    item['reason'] = reason
                    item['risk_note'] = risk_note
                    item['suggested_use'] = suggested_use
                    item['entity_type'] = 'case' if 'TeachingCase' in (item.get('labels') or []) else item.get('media_type') or 'entity'
                    item['media'] = _guess_media_type(item.get('labels') or [], item.get('media_type', ''), item.get('entity', ''))
                    if not item.get('clip_url') and item.get('clip_path'):
                        item['clip_url'] = _media_url_for_path(item.get('clip_path', ''))
                    item['media_url'] = _preferred_playback_url(item)

                    relation = item.get('relation', '')
                    labels = item.get('labels') or []
                    if relation == 'SIMILAR':
                        similar_entities.append({
                            'entity': item['entity'],
                            'similarity': item.get('similarity', 0.0),
                            'labels': labels,
                            'relation': relation,
                            'summary': item.get('summary', ''),
                            'score': score,
                            'risk_note': risk_note,
                            'suggested_use': suggested_use,
                        })
                    else:
                        related_entities.append({
                            'entity': item['entity'],
                            'similarity': item.get('similarity', 0.0),
                            'labels': labels,
                            'relation': relation,
                            'summary': item.get('summary', ''),
                            'stage_tags': item.get('stage_tags', []),
                            'course_tags': item.get('course_tags', []),
                            'ideology_tags': item.get('ideology_tags', []),
                            'knowledge_points': item.get('knowledge_points', []),
                            'knowledge_point_count': item.get('knowledge_point_count', 0),
                            'videos': item.get('videos', []),
                            'video_count': item.get('video_count', 0),
                            'score': score,
                            'reason': reason,
                            'risk_note': risk_note,
                            'suggested_use': suggested_use,
                            'via_entity': item.get('via_entity', ''),
                            'hop': item.get('hop', 1),
                        })

                    media_type = str(item.get('media_type') or '').lower()
                    entity_name = str(item.get('entity') or '')
                    relation_hint = str(item.get('relation') or '')
                    looks_like_video_file = entity_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
                    if 'Video' in labels or media_type == 'video' or relation_hint == 'MEDIA_LINKED_VIDEO' or looks_like_video_file:
                        video_path = item.get('path')
                        clip_path = item.get('clip_path') or item.get('clip_url') or ''
                        clip_url = item.get('clip_url') or _media_url_for_path(clip_path)
                        video_url = _preferred_playback_url(item)

                        abs_video_path = video_path
                        if abs_video_path and not os.path.isabs(abs_video_path):
                            abs_video_path = os.path.join(data_dir, abs_video_path)

                        # 仅在没有现成 clip 且视频存在时，动态切60秒片段
                        if (not clip_path) and abs_video_path and os.path.exists(abs_video_path) and self.video_editor:
                            estimated_seconds = 0.0
                            fps = float(item.get('fps') or 0.0)
                            frame_count = int(item.get('frame_count') or 0)
                            if fps and frame_count:
                                estimated_seconds = frame_count / max(fps, 1e-6)
                            if estimated_seconds > 60:
                                start_time = max(0, int((estimated_seconds - 60) / 2))
                                clip_dir = os.path.join(data_dir, 'clips')
                                os.makedirs(clip_dir, exist_ok=True)
                                clip_path = os.path.join(clip_dir, f"{Path(item['entity']).stem}_clip.mp4")
                                if not self.video_editor.clip_video(abs_video_path, clip_path, start_time, 60):
                                    clip_path = ''
                                else:
                                    clip_url = _media_url_for_path(clip_path)

                        if not clip_url and clip_path:
                            clip_url = _media_url_for_path(clip_path)
                        if not video_url:
                            video_url = _media_url_for_path(video_path)

                        videos.append({
                            'name': item['entity'],
                            'caption': item.get('caption') or item.get('summary', ''),
                            'clip_path': clip_url,
                            'clip_file': clip_path,
                            'path': video_url,
                            'media': 'video',
                            'media_url': clip_url or video_url,
                            'source_file': video_path,
                            'score': score,
                            'reason': reason,
                            'risk_note': risk_note,
                            'suggested_use': suggested_use,
                        })

                    looks_like_image_file = entity_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
                    if 'Image' in labels or media_type == 'image' or relation_hint == 'MEDIA_LINKED_IMAGE' or looks_like_image_file:
                        image_url = item.get('media_url') or _media_url_for_path(item.get('path'))
                        images.append({
                            'name': item['entity'],
                            'caption': item.get('caption') or item.get('summary', ''),
                            'ocr_text': item.get('ocr_text', ''),
                            'path': image_url,
                            'media': 'image',
                            'media_url': image_url,
                            'source_file': item.get('path', ''),
                            'score': score,
                            'reason': reason,
                        })

                    # 兼容旧数据：历史版本图片仅写入实体属性 image_paths/image_captions/image_ocr_texts。
                    legacy_paths = item.get('image_paths') or []
                    legacy_caps = item.get('image_captions') or []
                    legacy_ocrs = item.get('image_ocr_texts') or []
                    for idx, legacy_path in enumerate(legacy_paths[:6]):
                        legacy_url = _media_url_for_path(legacy_path)
                        images.append({
                            'name': f"{item['entity']}_legacy_image_{idx + 1}",
                            'caption': legacy_caps[idx] if idx < len(legacy_caps) else item.get('summary', ''),
                            'ocr_text': legacy_ocrs[idx] if idx < len(legacy_ocrs) else '',
                            'path': legacy_url,
                            'media': 'image',
                            'media_url': legacy_url,
                            'source_file': legacy_path,
                            'score': score,
                            'reason': 'legacy_entity_image_property',
                        })

                    if 'TeachingCase' in labels or relation in {'LINKS_TO_CASE', 'MENTIONS'}:
                        recommendations.append({
                            'entity': item['entity'],
                            'relation': relation,
                            'labels': labels,
                            'score': score,
                            'reason': reason,
                            'summary': item.get('summary', ''),
                            'path': item.get('path', ''),
                            'stage_tags': item.get('stage_tags', []),
                            'course_tags': item.get('course_tags', []),
                            'ideology_tags': item.get('ideology_tags', []),
                            'knowledge_points': item.get('knowledge_points', []),
                            'knowledge_point_count': item.get('knowledge_point_count', 0),
                            'videos': item.get('videos', []),
                            'video_count': item.get('video_count', 0),
                            'teaching_objectives': item.get('teaching_objectives', []),
                            'chapter_count': item.get('chapter_count', 0),
                            'isbn': item.get('isbn', ''),
                            'risk_note': risk_note,
                            'suggested_use': suggested_use,
                            'via_entity': item.get('via_entity', ''),
                            'hop': item.get('hop', 1),
                        })

                similar_entities.sort(key=lambda x: x['similarity'], reverse=True)
                related_entities.sort(key=lambda x: x.get('score', 0.0), reverse=True)
                videos.sort(key=lambda x: x.get('score', 0.0), reverse=True)
                dedup_images: Dict[str, Dict[str, Any]] = {}
                for image_item in images:
                    key = _safe_text(image_item.get('media_url')) or _safe_text(image_item.get('source_file'))
                    if not key:
                        continue
                    prev = dedup_images.get(key)
                    if prev is None or float(image_item.get('score', 0.0)) > float(prev.get('score', 0.0)):
                        dedup_images[key] = image_item
                images = list(dedup_images.values())
                images.sort(key=lambda x: x.get('score', 0.0), reverse=True)
                recommendations.sort(key=lambda x: x.get('score', 0.0), reverse=True)

                return {
                    'similar': similar_entities[:top_k],
                    'related': related_entities[:top_k],
                    'videos': videos[:top_k],
                    'images': images[:top_k],
                    'recommendations': recommendations[:top_k],
                    'resolved_entity': resolved_entity,
                    'entity_candidates': entity_candidates,
                    'semantic_model': {
                        'profile': self.semantic_model_profile,
                        'resolved_path': self.semantic_model_resolved,
                        'available': self.semantic_model_available,
                    },
                }
        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            return {
                "similar": [],
                "related": [],
                "videos": [],
                "images": [],
                "recommendations": [],
                "resolved_entity": entity,
                "entity_candidates": [],
                "semantic_model": {
                    'profile': self.semantic_model_profile,
                    'resolved_path': self.semantic_model_resolved,
                    'available': self.semantic_model_available,
                },
            }

    def close(self):
        if self.driver:
            self.driver.close()


# 全局查询器
neo4j_query = None
caption_generator = None


def get_neo4j_query():
    global neo4j_query
    if neo4j_query is None:
        neo4j_query = Neo4jQuery(
            uri=app.config.get('NEO4J_URI', 'bolt://localhost:7687'),
            user=app.config.get('NEO4J_USERNAME') or app.config.get('NEO4J_USER', 'neo4j'),
            password=app.config.get('NEO4J_PASSWORD', 'password'),
            database=app.config.get('NEO4J_DATABASE', 'neo4j'),
            semantic_model_dir=app.config.get('QUERY_BERT_MODEL_DIR'),
            semantic_model_profile=app.config.get('QUERY_BERT_PROFILE', 'bert_cn_base'),
            semantic_device_mode=app.config.get('QUERY_DEVICE_MODE', 'cpu'),
        )
    return neo4j_query


def close_neo4j_query():
    """进程退出时关闭全局Neo4j连接，避免每个请求重复关闭。"""
    global neo4j_query
    if neo4j_query is not None:
        neo4j_query.close()
        neo4j_query = None


def get_caption_generator():
    global caption_generator
    if caption_generator is None:
        try:
            from xmodaler.kg.processors import CaptionGenerator
            caption_generator = CaptionGenerator(
                use_ocr=True,
                ocr_lang='chi_sim+eng',
                use_asr=True,
                use_xmodaler_video=True,
                xmodaler_model_type='tdconved',
                device_mode=app.config.get('QUERY_DEVICE_MODE', 'cpu'),
            )
        except Exception as e:
            logger.error(f"初始化媒体处理器失败: {e}")
            caption_generator = None
    return caption_generator


atexit.register(close_neo4j_query)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/annotate')
def annotate_page():
    if not _annotation_enabled():
        return jsonify({"error": "Manual annotation is disabled"}), 403
    return render_template('annotate.html')


@app.route('/api/annotation/entities', methods=['GET'])
def annotation_entities():
    if not _annotation_enabled():
        return jsonify({"error": "Manual annotation is disabled"}), 403
    return jsonify(_load_entity_choices())


@app.route('/api/annotation/videos', methods=['GET'])
def annotation_videos():
    if not _annotation_enabled():
        return jsonify({"error": "Manual annotation is disabled"}), 403
    return jsonify({'videos': _list_videos_for_annotation()})


@app.route('/api/annotation/save', methods=['POST'])
def annotation_save():
    if not _annotation_enabled():
        return jsonify({"error": "Manual annotation is disabled"}), 403

    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    choices = _load_entity_choices()
    computer_set = set(choices.get('computer_entities', []))
    ideology_set = set(choices.get('ideology_entities', []))

    video_name = _safe_text(payload.get('video_name'))
    video_path = _safe_text(payload.get('video_path'))
    computer_entities = _normalize_multi_values(payload, 'computer_entities', 'computer_entity')
    ideology_entities = _normalize_multi_values(payload, 'ideology_entities', 'ideology_entity')

    if not video_name or not video_path:
        return jsonify({'error': 'video_name 和 video_path 不能为空'}), 400
    if not computer_entities and not ideology_entities:
        return jsonify({'error': '至少选择一个计算机知识点或思政元素'}), 400

    invalid_computers = [item for item in computer_entities if item not in computer_set]
    if invalid_computers:
        return jsonify({'error': f'computer_entities 存在非法值: {invalid_computers}'}), 400

    invalid_ideologies = [item for item in ideology_entities if item not in ideology_set]
    if invalid_ideologies:
        return jsonify({'error': f'ideology_entities 存在非法值: {invalid_ideologies}'}), 400

    try:
        start_sec = max(0.0, float(payload.get('start_sec', 0.0)))
        end_sec = max(start_sec, float(payload.get('end_sec', start_sec)))
        confidence = float(payload.get('confidence', 0.85))
    except (TypeError, ValueError):
        return jsonify({'error': '数值字段格式错误'}), 400

    stage_tags = _normalize_multi_values(payload, 'stage_tags', 'stage')
    course_tags = _normalize_multi_values(payload, 'course_tags', 'course')
    teaching_objectives = _normalize_multi_values(payload, 'teaching_objectives', 'objective')
    lesson_phase = _safe_text(payload.get('lesson_phase'))
    difficulty = _safe_text(payload.get('difficulty'))
    audience = _safe_text(payload.get('audience'))
    source_media_type = _safe_text(payload.get('source_media_type')) or _classify_media_suffix(Path(video_path).suffix.lower())[0]

    primary_computer = computer_entities[0] if computer_entities else 'NONE'
    primary_ideology = ideology_entities[0] if ideology_entities else 'NONE'
    annotation_id = f"{video_name}__{start_sec:.1f}_{end_sec:.1f}__{primary_computer}__{primary_ideology}"
    record = {
        'annotation_id': annotation_id,
        'video_name': video_name,
        'video_path': video_path,
        'video_url': f"/media/{video_path}",
        'start_sec': round(start_sec, 1),
        'end_sec': round(end_sec, 1),
        'computer_entities': computer_entities,
        'ideology_entities': ideology_entities,
        'computer_entity': computer_entities[0] if computer_entities else '',
        'ideology_entity': ideology_entities[0] if ideology_entities else '',
        'caption': _safe_text(payload.get('caption')),
        'ocr_text': _safe_text(payload.get('ocr_text')),
        'confidence': max(0.0, min(1.0, confidence)),
        'stage_tags': stage_tags,
        'course_tags': course_tags,
        'teaching_objectives': teaching_objectives,
        'lesson_phase': lesson_phase,
        'difficulty': difficulty,
        'audience': audience,
        'annotator': _safe_text(payload.get('annotator')),
        'source_media_type': source_media_type,
        'created_at': datetime.utcnow().isoformat() + 'Z',
    }

    with _annotation_file().open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

    #保存 JSONL + 可选即时同步 Neo4j（默认）
    sync_to_neo4j = str(payload.get('sync_to_neo4j', 'true')).strip().lower() not in {'0', 'false', 'no', 'off'}
    sync_status = 'skipped'
    if sync_to_neo4j:
        query_engine = get_neo4j_query()
        if query_engine.driver:
            try:
                now = datetime.utcnow().isoformat() + 'Z'
                all_entities = list(dict.fromkeys(computer_entities + ideology_entities))
                with query_engine.driver.session(database=query_engine.database) as session:
                    session.run(
                        """
                        MERGE (v:Entity:Media:Video {name: $name})
                        SET v.media_type = 'video',
                            v.source_media_type = $source_media_type,
                            v.path = $path,
                            v.relative_path = $path,
                            v.media_url = $media_url,
                            v.caption = CASE WHEN $caption = '' THEN coalesce(v.caption, '') ELSE $caption END,
                            v.ocr_text = CASE WHEN $ocr_text = '' THEN coalesce(v.ocr_text, '') ELSE $ocr_text END,
                            v.stage_tags = CASE WHEN size($stage_tags)=0 THEN coalesce(v.stage_tags, []) ELSE $stage_tags END,
                            v.course_tags = CASE WHEN size($course_tags)=0 THEN coalesce(v.course_tags, []) ELSE $course_tags END,
                            v.teaching_objectives = CASE WHEN size($teaching_objectives)=0 THEN coalesce(v.teaching_objectives, []) ELSE $teaching_objectives END,
                            v.lesson_phase = CASE WHEN $lesson_phase = '' THEN coalesce(v.lesson_phase, '') ELSE $lesson_phase END,
                            v.difficulty = CASE WHEN $difficulty = '' THEN coalesce(v.difficulty, '') ELSE $difficulty END,
                            v.audience = CASE WHEN $audience = '' THEN coalesce(v.audience, '') ELSE $audience END,
                            v.updated_at = $updated_at
                        """,
                        name=video_name,
                        path=video_path,
                        media_url=f"/media/{video_path}",
                        source_media_type=source_media_type,
                        caption=record['caption'],
                        ocr_text=record['ocr_text'],
                        stage_tags=stage_tags,
                        course_tags=course_tags,
                        teaching_objectives=teaching_objectives,
                        lesson_phase=lesson_phase,
                        difficulty=difficulty,
                        audience=audience,
                        updated_at=now,
                    )
                    for entity in all_entities:
                        session.run(
                            """
                            MATCH (e:Entity {name: $entity})
                            MATCH (v:Entity {name: $video_name})
                            MERGE (e)-[r:MEDIA_LINKED_VIDEO]->(v)
                            SET r.similarity = CASE
                                WHEN r.similarity IS NULL THEN $confidence
                                WHEN r.similarity < $confidence THEN $confidence
                                ELSE r.similarity
                            END,
                                r.caption = CASE WHEN $caption = '' THEN coalesce(r.caption, '') ELSE $caption END,
                                r.ocr_text = CASE WHEN $ocr_text = '' THEN coalesce(r.ocr_text, '') ELSE $ocr_text END,
                                r.media_path = $path,
                                r.media_url = $media_url,
                                r.media_type = 'video',
                                r.annotation_source = 'manual',
                                r.stage_tags = $stage_tags,
                                r.course_tags = $course_tags,
                                r.teaching_objectives = $teaching_objectives,
                                r.lesson_phase = $lesson_phase,
                                r.difficulty = $difficulty,
                                r.audience = $audience,
                                r.updated_at = $updated_at
                            """,
                            entity=entity,
                            video_name=video_name,
                            confidence=record['confidence'],
                            caption=record['caption'],
                            ocr_text=record['ocr_text'],
                            path=video_path,
                            media_url=f"/media/{video_path}",
                            stage_tags=stage_tags,
                            course_tags=course_tags,
                            teaching_objectives=teaching_objectives,
                            lesson_phase=lesson_phase,
                            difficulty=difficulty,
                            audience=audience,
                            updated_at=now,
                        )
                sync_status = 'ok'
            except Exception as e:
                sync_status = f'failed: {e}'
                logger.error(f"人工标注同步 Neo4j 失败: {e}")
        else:
            sync_status = 'failed: neo4j_unavailable'

    return jsonify({'ok': True, 'annotation_id': annotation_id, 'record': record, 'neo4j_sync': sync_status})


@app.route('/api/query', methods=['GET'])
def query():
    entity = request.args.get('entity')
    if not entity:
        return jsonify({"error": "Entity parameter required"}), 400

    query_engine = get_neo4j_query()
    results = query_engine.query_connected_entities(entity)
    return jsonify(results)


@app.route('/api/query_advanced', methods=['GET'])
def query_advanced():
    entity = request.args.get('entity')
    data_dir = request.args.get('data_dir', './data')
    stage = request.args.get('stage')
    focus = request.args.get('focus') or request.args.get('ideology_focus')
    course = request.args.get('course')
    prerequisites = _split_csv(request.args.get('prerequisites'))
    objective = request.args.get('objective')
    audience = request.args.get('audience')
    constraints = request.args.get('constraints')
    duration_raw = request.args.get('duration')
    top_k_raw = request.args.get('top_k', '10')

    if not entity:
        return jsonify({"error": "Entity parameter required"}), 400

    try:
        duration = float(duration_raw) if duration_raw not in (None, '', 'null') else None
    except ValueError:
        duration = None

    try:
        top_k = max(1, min(50, int(top_k_raw)))
    except ValueError:
        top_k = 10

    query_engine = get_neo4j_query()
    results = query_engine.query_similar_and_related_entities(
        entity=entity,
        data_dir=data_dir,
        stage=stage,
        duration=duration,
        focus=focus,
        course=course,
        prerequisites=prerequisites,
        objective=objective,
        audience=audience,
        constraints=constraints,
        top_k=top_k,
    )
    return jsonify(results)


@app.route('/api/graph', methods=['GET'])
def get_graph():
    query_engine = get_neo4j_query()
    graph_data = query_engine.get_graph_data()
    return jsonify(graph_data)


@app.route('/api/graph/subgraph', methods=['GET'])
def get_subgraph():
    entity = request.args.get('entity', '')
    hops = request.args.get('hops', '2')
    if not entity:
        return jsonify({"error": "Entity parameter required"}), 400
    try:
        hops_int = max(1, min(2, int(hops)))
    except Exception:
        hops_int = 2

    query_engine = get_neo4j_query()
    graph_data = query_engine.get_subgraph_data(entity=entity, hops=hops_int)
    return jsonify(graph_data)


def _load_ingest_whitelist() -> Dict[str, List[str]]:
    choices = _load_entity_choices()
    computer = choices.get('computer_entities', [])
    ideology = choices.get('ideology_entities', [])
    return {
        'computer_entities': [item for item in computer if item],
        'ideology_entities': [item for item in ideology if item],
    }


def _match_terms(text: str, candidates: List[str]) -> List[str]:
    matched = []
    normalized = _safe_text(text)
    for term in candidates or []:
        if term and term in normalized:
            matched.append(term)
    return list(dict.fromkeys(matched))


def _semantic_match_terms(query_engine: Any, text: str, candidates: List[str], threshold: float = 0.36, top_k: int = 8) -> List[str]:
    if not text or not candidates:
        return []
    semantic = getattr(query_engine, 'semantic', None)
    if semantic is None:
        return []
    candidate_map = {term: {'name': term, 'type': 'whitelist_term'} for term in candidates if term}
    try:
        ranked = semantic.rank_candidates(text, candidate_map, top_k=max(top_k, 3))
    except Exception:
        return []
    return [item['name'] for item in ranked if item.get('name') and float(item.get('score', 0.0)) >= threshold]


def _collect_linked_entities(query_engine: Any, hint_text: str, whitelist: Dict[str, List[str]], media_name: str) -> List[str]:
    computer_pool = whitelist.get('computer_entities', [])
    ideology_pool = whitelist.get('ideology_entities', [])
    allowed_pool = list(dict.fromkeys(computer_pool + ideology_pool))

    linked = set(_match_terms(hint_text, allowed_pool))
    linked.update(_semantic_match_terms(query_engine, hint_text, allowed_pool))

    if hasattr(query_engine, 'extract_query_terms') and hasattr(query_engine, 'resolve_entity_name'):
        try:
            for term in query_engine.extract_query_terms(hint_text)[:10]:
                resolved, alternatives = query_engine.resolve_entity_name(term)
                for name in [resolved] + alternatives[:2]:
                    if name and name in allowed_pool:
                        linked.add(name)
        except Exception as e:
            logger.warning(f"实体自动建边候选解析失败: {e}")

    linked.discard(media_name)
    return sorted(linked)


def _score_image_entity_relevance(query_engine: Any,
                                  entity_name: str,
                                  caption: str,
                                  ocr_text: str,
                                  filename_terms: List[str]) -> float:
    """Score image-to-entity relevance for optional image-node ingestion."""
    entity = _safe_text(entity_name)
    context_text = _safe_text(' '.join([caption or '', ocr_text or '', ' '.join(filename_terms or [])]))
    if not entity or not context_text:
        return 0.0

    term_hit = 1.0 if entity in context_text else 0.0
    filename_hit = 1.0 if entity in _safe_text(' '.join(filename_terms or [])) else 0.0

    semantic = 0.0
    semantic_engine = getattr(query_engine, 'semantic', None)
    if semantic_engine:
        try:
            semantic = float(semantic_engine.similarity(entity, context_text))
        except Exception:
            semantic = 0.0

    score = (0.6 * semantic) + (0.3 * term_hit) + (0.1 * filename_hit)
    return max(0.0, min(1.0, round(score, 4)))


def _rank_image_entity_candidates(query_engine: Any,
                                  candidates: List[str],
                                  caption: str,
                                  ocr_text: str,
                                  filename_terms: List[str]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for entity in candidates or []:
        score = _score_image_entity_relevance(
            query_engine=query_engine,
            entity_name=entity,
            caption=caption,
            ocr_text=ocr_text,
            filename_terms=filename_terms,
        )
        evidence_parts = []
        if entity and entity in _safe_text(caption):
            evidence_parts.append('caption_hit')
        if entity and entity in _safe_text(ocr_text):
            evidence_parts.append('ocr_hit')
        if entity and entity in _safe_text(' '.join(filename_terms or [])):
            evidence_parts.append('filename_hit')
        if not evidence_parts:
            evidence_parts.append('semantic_match')
        ranked.append({
            'entity': entity,
            'score': score,
            'evidence': evidence_parts,
        })

    ranked.sort(key=lambda item: float(item.get('score', 0.0)), reverse=True)
    return ranked


@app.route('/api/upload_media', methods=['POST'])
def upload_media():
    file = request.files.get('file')
    if file is None or not file.filename:
        return jsonify({'error': '缺少上传文件'}), 400

    suffix = Path(file.filename).suffix.lower()
    media_type, subdir = _classify_media_suffix(suffix)
    if media_type == 'unknown':
        return jsonify({'error': f'不支持的文件类型: {suffix}'}), 400

    filename = _sanitize_upload_filename(file.filename)
    data_root = _data_root()
    abs_dir = data_root / subdir
    abs_dir.mkdir(parents=True, exist_ok=True)
    target_path = abs_dir / filename
    file.save(str(target_path))

    relative_path = target_path.relative_to(data_root).as_posix()
    return jsonify({
        'ok': True,
        'relative_path': relative_path,
        'media_type': 'video' if media_type in {'video', 'audio'} else 'image',
        'source_media_type': media_type,
        'media_url': f"/media/{relative_path}",
    })


@app.route('/api/ingest_media', methods=['POST'])
def ingest_media():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    relative_path = _safe_text(payload.get('relative_path'))
    if not relative_path:
        return jsonify({'error': 'relative_path 不能为空'}), 400

    data_root = _data_root().resolve()
    abs_path = (data_root / relative_path).resolve()
    try:
        abs_path.relative_to(data_root)
    except Exception:
        return jsonify({'error': '非法路径'}), 400
    if not abs_path.exists() or not abs_path.is_file():
        return jsonify({'error': '文件不存在'}), 404

    suffix = abs_path.suffix.lower()
    source_media_type, _ = _classify_media_suffix(suffix)
    is_video = source_media_type in {'video', 'audio'}
    is_image = source_media_type == 'image'
    if source_media_type == 'unknown':
        return jsonify({'error': '暂不支持的媒体类型'}), 400

    generator = get_caption_generator()
    if generator is None:
        return jsonify({'error': '媒体处理器不可用'}), 500

    caption = generator.generate_video_caption(str(abs_path)) if is_video else generator.generate_image_caption(str(abs_path))
    ocr_text = None if is_video else generator.recognize_text_from_image(str(abs_path))
    whitelist = _load_ingest_whitelist()
    filename_terms = _filename_keywords(relative_path)
    hint_text = ' '.join([caption or '', ocr_text or '', ' '.join(filename_terms)])

    query_engine = get_neo4j_query()
    if not query_engine.driver:
        return jsonify({'error': 'Neo4j 不可用'}), 500

    media_url = f"/media/{relative_path}"
    media_name = abs_path.name
    linked_entities = _collect_linked_entities(query_engine, hint_text, whitelist, media_name=media_name)

    with query_engine.driver.session(database=query_engine.database) as session:
        if is_video:
            session.run(
                """
                MERGE (v:Entity:Media:Video {name: $name})
                SET v.media_type = 'video',
                    v.source_media_type = $source_media_type,
                    v.path = $path,
                    v.relative_path = $relative_path,
                    v.media_url = $media_url,
                    v.caption = $caption,
                    v.updated_at = $updated_at
                """,
                name=media_name,
                path=relative_path,
                relative_path=relative_path,
                media_url=media_url,
                caption=caption or '',
                source_media_type=source_media_type,
                updated_at=datetime.utcnow().isoformat() + 'Z',
            )
            for entity in linked_entities:
                session.run(
                    """
                    MATCH (e:Entity {name: $entity})
                    MATCH (v:Entity {name: $video_name})
                    MERGE (e)-[r:MEDIA_LINKED_VIDEO]->(v)
                    SET r.similarity = coalesce(r.similarity, 0.8),
                        r.caption = $caption,
                        r.media_path = $relative_path,
                        r.media_url = $media_url,
                        r.media_type = 'video',
                        r.source_media_type = $source_media_type,
                        r.updated_at = $updated_at
                    """,
                    entity=entity,
                    video_name=media_name,
                    caption=caption or '',
                    relative_path=relative_path,
                    media_url=media_url,
                    source_media_type=source_media_type,
                    updated_at=datetime.utcnow().isoformat() + 'Z',
                )
        elif is_image:
            image_node_enable = bool(app.config.get('ENABLE_IMAGE_NODE_INGEST', True))
            image_node_threshold = float(app.config.get('IMAGE_NODE_MIN_SIMILARITY', 0.50) or 0.50)
            image_node_topk = max(1, int(app.config.get('IMAGE_NODE_MAX_LINKS', 3) or 3))

            ranked_candidates = _rank_image_entity_candidates(
                query_engine=query_engine,
                candidates=linked_entities,
                caption=caption or '',
                ocr_text=ocr_text or '',
                filename_terms=filename_terms,
            )
            high_related = [
                item for item in ranked_candidates
                if float(item.get('score', 0.0)) >= image_node_threshold
            ][:image_node_topk]

            if image_node_enable and high_related:
                session.run(
                    """
                    MERGE (i:Entity:Media:Image {name: $name})
                    SET i.media_type = 'image',
                        i.source_media_type = 'image',
                        i.path = $path,
                        i.relative_path = $relative_path,
                        i.media_url = $media_url,
                        i.caption = CASE WHEN $caption = '' THEN coalesce(i.caption, '') ELSE $caption END,
                        i.ocr_text = CASE WHEN $ocr_text = '' THEN coalesce(i.ocr_text, '') ELSE $ocr_text END,
                        i.updated_at = $updated_at
                    """,
                    name=media_name,
                    path=relative_path,
                    relative_path=relative_path,
                    media_url=media_url,
                    caption=caption or '',
                    ocr_text=ocr_text or '',
                    updated_at=datetime.utcnow().isoformat() + 'Z',
                )

                for item in high_related:
                    session.run(
                        """
                        MATCH (e:Entity {name: $entity})
                        MATCH (i:Entity:Image {name: $image_name})
                        MERGE (e)-[r:MEDIA_LINKED_IMAGE]->(i)
                        SET r.similarity = CASE
                            WHEN r.similarity IS NULL THEN $score
                            WHEN r.similarity < $score THEN $score
                            ELSE r.similarity
                        END,
                            r.caption = CASE WHEN $caption = '' THEN coalesce(r.caption, '') ELSE $caption END,
                            r.ocr_text = CASE WHEN $ocr_text = '' THEN coalesce(r.ocr_text, '') ELSE $ocr_text END,
                            r.media_path = $relative_path,
                            r.media_url = $media_url,
                            r.media_type = 'image',
                            r.evidence = $evidence,
                            r.updated_at = $updated_at
                        """,
                        entity=item['entity'],
                        image_name=media_name,
                        score=float(item.get('score', 0.0)),
                        caption=caption or '',
                        ocr_text=ocr_text or '',
                        relative_path=relative_path,
                        media_url=media_url,
                        evidence=item.get('evidence', []),
                        updated_at=datetime.utcnow().isoformat() + 'Z',
                    )
            else:
                for entity in linked_entities:
                    session.run(
                        """
                        MATCH (e:Entity {name: $entity})
                        SET e.image_captions = CASE
                            WHEN $caption = '' THEN coalesce(e.image_captions, [])
                            WHEN $caption IN coalesce(e.image_captions, []) THEN coalesce(e.image_captions, [])
                            ELSE coalesce(e.image_captions, []) + $caption
                        END,
                        e.image_ocr_texts = CASE
                            WHEN $ocr_text = '' THEN coalesce(e.image_ocr_texts, [])
                            WHEN $ocr_text IN coalesce(e.image_ocr_texts, []) THEN coalesce(e.image_ocr_texts, [])
                            ELSE coalesce(e.image_ocr_texts, []) + $ocr_text
                        END,
                        e.image_paths = CASE
                            WHEN $relative_path IN coalesce(e.image_paths, []) THEN coalesce(e.image_paths, [])
                            ELSE coalesce(e.image_paths, []) + $relative_path
                        END,
                        e.image_count = size(coalesce(e.image_paths, [])),
                        e.updated_at = $updated_at
                        """,
                        entity=entity,
                        caption=caption or '',
                        ocr_text=ocr_text or '',
                        relative_path=relative_path,
                        updated_at=datetime.utcnow().isoformat() + 'Z',
                    )

    image_top_entities = []
    if is_image:
        image_ranked = _rank_image_entity_candidates(
            query_engine=query_engine,
            candidates=linked_entities,
            caption=caption or '',
            ocr_text=ocr_text or '',
            filename_terms=filename_terms,
        )
        image_top_entities = image_ranked[:max(1, int(app.config.get('IMAGE_NODE_MAX_LINKS', 3) or 3))]

    primary_entity = linked_entities[0] if linked_entities else ''

    return jsonify({
        'ok': True,
        'summary': f"{media_name} 已处理，匹配实体 {len(linked_entities)} 个",
        'media_type': 'video' if is_video else 'image',
        'source_media_type': source_media_type,
        'caption': caption,
        'ocr_text': ocr_text,
        'linked_entities': linked_entities,
        'primary_entity': primary_entity,
        'image_top_entities': image_top_entities,
        'filename_terms': filename_terms,
    })


@app.route('/api/upload_text', methods=['POST'])
def upload_text():
    file = request.files.get('file')
    if file is None or not file.filename:
        return jsonify({'error': '缺少上传文件'}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in TEXT_EXTS:
        return jsonify({'error': f'仅支持文本文件: {", ".join(sorted(TEXT_EXTS))}'}), 400

    filename = _sanitize_upload_filename(file.filename)
    if Path(filename).suffix.lower() != '.txt':
        filename = f"{Path(filename).stem}.txt"

    data_root = _data_root()
    abs_dir = data_root / 'uploads' / 'txt'
    abs_dir.mkdir(parents=True, exist_ok=True)
    target_path = abs_dir / filename
    file.save(str(target_path))

    relative_path = target_path.relative_to(data_root).as_posix()
    return jsonify({
        'ok': True,
        'relative_path': relative_path,
        'media_type': 'text',
    })


@app.route('/api/ingest_text', methods=['POST'])
def ingest_text():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    relative_path = _safe_text(payload.get('relative_path'))
    if not relative_path:
        return jsonify({'error': 'relative_path 不能为空'}), 400

    data_root = _data_root().resolve()
    abs_path = (data_root / relative_path).resolve()
    try:
        abs_path.relative_to(data_root)
    except Exception:
        return jsonify({'error': '非法路径'}), 400
    if not abs_path.exists() or not abs_path.is_file():
        return jsonify({'error': '文件不存在'}), 404
    if abs_path.suffix.lower() not in TEXT_EXTS:
        return jsonify({'error': '仅支持 .txt 文本入图'}), 400

    try:
        raw_text = _read_text_file(abs_path)
    except Exception as e:
        return jsonify({'error': f'读取文本失败: {e}'}), 400

    paragraphs = _split_text_paragraphs(raw_text, max_paragraphs=40, max_len=260)
    if not paragraphs:
        return jsonify({'error': '文本内容为空'}), 400

    query_engine = get_neo4j_query()
    if not query_engine.driver:
        return jsonify({'error': 'Neo4j 不可用'}), 500

    whitelist = _load_ingest_whitelist()
    computer_pool = whitelist.get('computer_entities', [])
    ideology_pool = whitelist.get('ideology_entities', [])
    whitelist_pool = list(dict.fromkeys(computer_pool + ideology_pool))
    whitelist_available = bool(whitelist_pool)

    text_name = abs_path.name
    full_text = ' '.join(paragraphs)
    explicit_whitelist_hits = _match_terms(full_text, whitelist_pool)
    semantic_whitelist_hits = _semantic_match_terms(query_engine, full_text, whitelist_pool, threshold=0.42, top_k=12)
    whitelist_hits = list(dict.fromkeys(explicit_whitelist_hits + semantic_whitelist_hits))

    created_entities: List[str] = []
    linked_entities: List[str] = []
    paragraph_map: Dict[str, List[str]] = {}
    skipped_whitelist_creation = False

    now = datetime.utcnow().isoformat() + 'Z'
    with query_engine.driver.session(database=query_engine.database) as session:
        existing_entities = _extract_existing_entities_from_text(session, query_engine, paragraphs)
        linked_entities = list(dict.fromkeys(existing_entities))

        for entity_name in linked_entities:
            snippets = [p for p in paragraphs if entity_name in p][:5]
            if not snippets:
                snippets = paragraphs[:2]
            paragraph_map[entity_name] = snippets

        # C 方案：创建 TextDocument 节点，同时把文本摘要写入相关实体属性。
        session.run(
            """
            MERGE (d:Entity:TextDocument {name: $name})
            SET d.media_type = 'text',
                d.source_media_type = 'text',
                d.path = $relative_path,
                d.relative_path = $relative_path,
                d.media_url = $media_url,
                d.full_text = $full_text,
                d.text_paragraphs = $paragraphs,
                d.paragraph_count = size($paragraphs),
                d.updated_at = $updated_at
            """,
            name=text_name,
            relative_path=relative_path,
            media_url=f"/media/{relative_path}",
            full_text=full_text[:6000],
            paragraphs=paragraphs,
            updated_at=now,
        )

        for entity_name in linked_entities:
            snippets = paragraph_map.get(entity_name, [])
            session.run(
                """
                MATCH (e:Entity {name: $entity_name})
                MATCH (d:Entity:TextDocument {name: $doc_name})
                SET e.related_text_docs = reduce(acc = coalesce(e.related_text_docs, []), x IN [$doc_name] |
                    CASE WHEN x IN acc THEN acc ELSE acc + x END),
                    e.related_text_snippets = reduce(acc = coalesce(e.related_text_snippets, []), x IN $snippets |
                    CASE WHEN x IN acc THEN acc ELSE acc + x END),
                    e.updated_at = $updated_at
                MERGE (d)-[r:TEXT_MENTIONS]->(e)
                SET r.similarity = CASE
                    WHEN r.similarity IS NULL THEN 0.82
                    WHEN r.similarity < 0.82 THEN 0.82
                    ELSE r.similarity
                END,
                    r.evidence = reduce(acc = coalesce(r.evidence, []), x IN $snippets |
                    CASE WHEN x IN acc THEN acc ELSE acc + x END),
                    r.source = 'text_upload',
                    r.updated_at = $updated_at
                """,
                entity_name=entity_name,
                doc_name=text_name,
                snippets=snippets,
                updated_at=now,
            )

        # B 选项：白名单文件缺失/为空时，只做现有节点关联，不自动创建新实体。
        if not whitelist_available:
            skipped_whitelist_creation = True
        else:
            for term in whitelist_hits:
                exists = session.run(
                    "MATCH (e:Entity {name: $name}) RETURN COUNT(e) AS cnt",
                    name=term,
                ).single()
                if int((exists or {}).get('cnt', 0)) > 0:
                    if term not in linked_entities:
                        linked_entities.append(term)
                    continue

                if term in computer_pool:
                    session.run(
                        """
                        MERGE (e:Entity:KnowledgePoint {name: $name})
                        SET e.media_type = coalesce(e.media_type, 'text'),
                            e.source = 'whitelist_text_expansion',
                            e.updated_at = $updated_at
                        """,
                        name=term,
                        updated_at=now,
                    )
                else:
                    session.run(
                        """
                        MERGE (e:Entity:IdeologyElement {name: $name})
                        SET e.media_type = coalesce(e.media_type, 'text'),
                            e.source = 'whitelist_text_expansion',
                            e.updated_at = $updated_at
                        """,
                        name=term,
                        updated_at=now,
                    )

                created_entities.append(term)
                linked_entities.append(term)

                snippets = [p for p in paragraphs if term in p][:4] or paragraphs[:2]
                session.run(
                    """
                    MATCH (e:Entity {name: $entity_name})
                    MATCH (d:Entity:TextDocument {name: $doc_name})
                    MERGE (d)-[r:TEXT_MENTIONS]->(e)
                    SET r.similarity = CASE
                        WHEN r.similarity IS NULL THEN 0.8
                        WHEN r.similarity < 0.8 THEN 0.8
                        ELSE r.similarity
                    END,
                        r.evidence = reduce(acc = coalesce(r.evidence, []), x IN $snippets |
                        CASE WHEN x IN acc THEN acc ELSE acc + x END),
                        r.source = 'text_upload',
                        r.updated_at = $updated_at
                    """,
                    entity_name=term,
                    doc_name=text_name,
                    snippets=snippets,
                    updated_at=now,
                )

    linked_entities = sorted(list(dict.fromkeys(linked_entities)))
    created_entities = sorted(list(dict.fromkeys(created_entities)))
    primary_entity = created_entities[0] if created_entities else (linked_entities[0] if linked_entities else '')

    return jsonify({
        'ok': True,
        'summary': f"{text_name} 已处理，关联实体 {len(linked_entities)} 个，新增实体 {len(created_entities)} 个",
        'text_node': text_name,
        'relative_path': relative_path,
        'paragraph_count': len(paragraphs),
        'linked_entities': linked_entities,
        'created_entities': created_entities,
        'primary_entity': primary_entity,
        'whitelist_hits': whitelist_hits,
        'skipped_whitelist_creation': skipped_whitelist_creation,
    })


@app.route('/video/<path:filename>')
def serve_video(filename):
    video_dir = os.path.join(app.root_path, 'data', 'clips')
    return send_from_directory(video_dir, filename)


@app.route('/media/<path:filename>')
def serve_media(filename):
    data_dir = os.path.join(app.root_path, 'data')
    return send_from_directory(data_dir, filename)


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


@app.teardown_appcontext
def shutdown_session(exception=None):
    # Flask 每个请求都会触发 teardown_appcontext，
    # 这里不关闭全局 driver，避免后续请求出现 "Driver closed"。
    return None


if __name__ == '__main__':
    app.config['NEO4J_URI'] = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    app.config['NEO4J_USERNAME'] = os.getenv('NEO4J_USERNAME')
    app.config['NEO4J_USER'] = os.getenv('NEO4J_USER', 'neo4j')
    app.config['NEO4J_PASSWORD'] = os.getenv('NEO4J_PASSWORD', 'password')
    app.config['NEO4J_DATABASE'] = os.getenv('NEO4J_DATABASE', 'neo4j')
    app.config['QUERY_BERT_MODEL_DIR'] = os.getenv('QUERY_BERT_MODEL_DIR', '')
    app.config['QUERY_BERT_PROFILE'] = os.getenv('QUERY_BERT_PROFILE', 'bert_cn_finetuned')
    app.config['QUERY_DEVICE_MODE'] = os.getenv('QUERY_DEVICE_MODE', 'cpu')
    app.config['ENABLE_IMAGE_NODE_INGEST'] = os.getenv('ENABLE_IMAGE_NODE_INGEST', 'true').lower() in {'1', 'true', 'yes', 'on'}
    app.config['IMAGE_NODE_MIN_SIMILARITY'] = float(os.getenv('IMAGE_NODE_MIN_SIMILARITY', '0.50'))
    app.config['IMAGE_NODE_MAX_LINKS'] = int(os.getenv('IMAGE_NODE_MAX_LINKS', '3'))
    app.config['ENABLE_MANUAL_ANNOTATION'] = os.getenv('ENABLE_MANUAL_ANNOTATION', 'false').lower() in {'1', 'true', 'yes', 'on'}

    logger.info("启动Flask服务器...")
    logger.info("访问: http://localhost:6006")
    logger.info(f"人工标注页面: {'已启用' if app.config['ENABLE_MANUAL_ANNOTATION'] else '未启用'}")
    logger.info(f"查询语义模型档位: {app.config.get('QUERY_BERT_PROFILE', 'bert_cn_base')}")
    logger.info(f"查询语义设备: {app.config.get('QUERY_DEVICE_MODE', 'cpu')}")
    logger.info(f"高相关图片入图: {'已启用' if app.config.get('ENABLE_IMAGE_NODE_INGEST', True) else '未启用'}")
    if app.config['QUERY_BERT_MODEL_DIR']:
        logger.info(f"查询语义模型: {app.config['QUERY_BERT_MODEL_DIR']}")

    app.run(debug=True, host='0.0.0.0', port=6006)

