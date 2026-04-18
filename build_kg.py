#!/usr/bin/env python3
"""
知识图谱构建主脚本
使用方式:
    python build_kg.py          # 使用默认配置
    python build_kg.py --data-dir ./data --output-dir ./kg_output
"""

import os
import sys
import argparse
import logging
import ast
import subprocess
from pathlib import Path
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _load_entities_from_datamain(config_path: str):
    """从 datamain.txt 中读取 COMPUTER_LABELS 和 IDEOLOGY_LABELS。"""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"实体配置文件不存在: {path}")

    content = path.read_text(encoding='utf-8')
    computer_entities = None
    ideology_entities = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = [part.strip() for part in line.split('=', 1)]
        if key == 'COMPUTER_LABELS':
            computer_entities = ast.literal_eval(value)
        elif key == 'IDEOLOGY_LABELS':
            ideology_entities = ast.literal_eval(value)

    if not isinstance(computer_entities, list) or not isinstance(ideology_entities, list):
        raise ValueError(f"实体配置格式错误: {path}")

    computer_entities = [str(item).strip() for item in computer_entities if str(item).strip()]
    ideology_entities = [str(item).strip() for item in ideology_entities if str(item).strip()]
    return computer_entities, ideology_entities


def _parse_term_list(value: str):
    if not value:
        return []
    parts = re.split(r'[,，;；\n]+', value)
    return [item.strip() for item in parts if item and item.strip()]


def _load_terms_from_file(file_path: str) -> list:
    """从txt白名单文件读取短语，文件不存在/为空时返回空列表。"""
    if not file_path:
        return []
    path = Path(file_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists() or not path.is_file():
        return []

    try:
        content = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return []

    terms = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        terms.extend(_parse_term_list(line))
    return terms


def _merge_terms(*term_lists) -> list:
    merged = []
    seen = set()
    for term_list in term_lists:
        for item in term_list or []:
            term = str(item).strip()
            if not term or term in seen:
                continue
            seen.add(term)
            merged.append(term)
    return merged


def _run_video_preprocess(args) -> None:
    """在构建前调用视频兼容性检查/修复脚本。"""
    if args.skip_videos:
        logger.info("视频处理已跳过，视频预处理步骤自动跳过")
        return
    if not args.video_preprocess:
        return

    project_root = Path(__file__).resolve().parent
    script_path = project_root / 'tools' / 'video_compat_batch.py'
    if not script_path.exists():
        logger.warning(f"未找到视频预处理脚本，跳过: {script_path}")
        return

    video_dir = Path(args.data_dir) / 'video'
    if not video_dir.exists():
        logger.warning(f"未找到视频目录，跳过视频预处理: {video_dir}")
        return

    output_dir = Path(args.video_fix_output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    report_path = args.video_fix_report.strip() if isinstance(args.video_fix_report, str) else ''
    if report_path:
        report_file = Path(report_path)
        if not report_file.is_absolute():
            report_file = project_root / report_file
    else:
        report_file = Path(args.output_dir) / 'video_preprocess_report.json'
        if not report_file.is_absolute():
            report_file = project_root / report_file

    cmd = [
        sys.executable,
        str(script_path),
        str(video_dir),
        '--mode', args.video_fix_mode,
        '--output-dir', str(output_dir),
        '--report', str(report_file),
    ]
    if args.video_preprocess_fix:
        cmd.append('--fix')
    if int(args.video_fix_limit) > 0:
        cmd.extend(['--limit', str(int(args.video_fix_limit))])

    logger.info("开始视频兼容性预处理...")
    logger.info(f"  - 目录: {video_dir}")
    logger.info(f"  - 修复模式: {'修复' if args.video_preprocess_fix else '仅检查'}")
    logger.info(f"  - 输出策略: {args.video_fix_mode}")
    logger.info(f"  - 修复输出目录: {output_dir}")
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        raise RuntimeError(f"视频预处理失败，退出码: {result.returncode}")

    if args.video_preprocess_fix and args.video_fix_mode == 'separate':
        logger.info("视频已修复到 data/video_fixed（separate 模式），构图仍默认读取 data/video")
        logger.info("如需让本次构图使用修复后视频，请改用 --video-fix-mode inplace")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='构建跨模态知识图谱',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 使用默认配置
  python build_kg.py
  
  # 自定义数据目录
  python build_kg.py --data-dir /path/to/data --output-dir ./output
  
  # 跳过图像处理
  python build_kg.py --skip-images
  
  # 跳过视频处理
  python build_kg.py --skip-videos
        '''
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='数据目录路径 (默认: ./data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./kg_output',
        help='输出目录路径 (默认: ./kg_output)'
    )
    
    parser.add_argument(
        '--neo4j-uri',
        type=str,
        default=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        help='Neo4j数据库URI (默认: bolt://localhost:7687)'
    )
    
    parser.add_argument(
        '--neo4j-user',
        type=str,
        default=os.getenv('NEO4J_USERNAME') or os.getenv('NEO4J_USER', 'neo4j'),
        help='Neo4j用户名 (默认: neo4j)'
    )
    
    parser.add_argument(
        '--neo4j-password',
        type=str,
        default=os.getenv('NEO4J_PASSWORD', 'password'),
        help='Neo4j密码 (默认: password)'
    )

    parser.add_argument(
        '--neo4j-database',
        type=str,
        default=os.getenv('NEO4J_DATABASE', 'neo4j'),
        help='Neo4j数据库名 (默认: neo4j 或环境变量 NEO4J_DATABASE)'
    )
    
    parser.add_argument(
        '--skip-images',
        action='store_true',
        help='跳过图像处理'
    )
    
    parser.add_argument(
        '--skip-videos',
        action='store_true',
        help='跳过视频处理'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        choices=['zh', 'en'],
        default='zh',
        help='语言选择 (默认: zh)'
    )

    parser.add_argument(
        '--device-mode',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default=os.getenv('DEVICE_MODE', 'auto'),
        help='设备模式 (auto/cpu/cuda，默认: auto)'
    )
    
    parser.add_argument(
        '--computer-entities',
        type=str,
        default='',
        help='计算机核心知识点实体列表，用逗号分隔 (默认: 从 datamain.txt 读取)'
    )
    
    parser.add_argument(
        '--ideology-entities',
        type=str,
        default='',
        help='思政元素实体列表，用逗号分隔 (默认: 从 datamain.txt 读取)'
    )

    parser.add_argument(
        '--entity-config',
        type=str,
        default='BERT_cn/datamain.txt',
        help='预定义实体配置文件路径 (默认: BERT_cn/datamain.txt)'
    )
    
    parser.add_argument(
        '--custom-relations',
        type=str,
        default='',
        help='自定义关系列表，格式: entity1-entity2:relation_type，用分号分隔 (默认: 空)'
    )

    parser.add_argument(
        '--match-top-k',
        type=int,
        default=12,
        help='媒体到实体匹配返回数量上限 (默认: 12)'
    )

    parser.add_argument(
        '--semantic-threshold',
        type=float,
        default=0.45,
        help='语义匹配阈值，越高越严格 (默认: 0.45)'
    )

    parser.add_argument(
        '--relation-model-dir',
        type=str,
        default='BERT_cn/实体抽取/bert_relation_model',
        help='关系重排模型目录（默认: BERT_cn/实体抽取/bert_relation_model）'
    )

    parser.add_argument(
        '--relation-threshold',
        type=float,
        default=0.55,
        help='计算机-思政关系保留阈值 (默认: 0.55)'
    )

    parser.add_argument(
        '--disable-relation-rerank',
        action='store_true',
        help='关闭BERT关系重排，仅使用规则召回'
    )

    parser.add_argument(
        '--video-preprocess',
        action='store_true',
        help='在构图前执行视频兼容性预处理（调用 video_compat_batch.py）'
    )

    parser.add_argument(
        '--video-preprocess-fix',
        action='store_true',
        help='视频预处理执行修复；不加该参数时仅检查'
    )

    parser.add_argument(
        '--video-fix-mode',
        type=str,
        choices=['separate', 'inplace'],
        default=os.getenv('VIDEO_FIX_MODE', 'separate'),
        help='视频修复输出策略 (默认: separate)'
    )

    parser.add_argument(
        '--video-fix-output-dir',
        type=str,
        default=os.getenv('VIDEO_FIX_OUTPUT_DIR', 'data/video_fixed'),
        help='视频修复输出目录（separate 模式有效，默认: data/video_fixed）'
    )

    parser.add_argument(
        '--video-fix-limit',
        type=int,
        default=0,
        help='视频预处理最多处理多少文件（0 表示不限制）'
    )

    parser.add_argument(
        '--video-fix-report',
        type=str,
        default='',
        help='视频预处理报告路径（默认: <output-dir>/video_preprocess_report.json）'
    )

    parser.add_argument(
        '--use-xmodaler-video',
        action='store_true',
        default=True,
        help='使用 xmodaler 专业视频字幕模型（TDConvED, 默认开启）'
    )

    parser.add_argument(
        '--no-xmodaler-video',
        action='store_true',
        help='禁用 xmodaler 视频字幕模型，仅用 BLIP+OCR+ASR'
    )

    parser.add_argument(
        '--xmodaler-model-type',
        type=str,
        choices=['tdconved', 'ta'],
        default='tdconved',
        help='xmodaler 视频字幕模型类型（tdconved: MSR-VTT, ta: MSVD，默认: tdconved）'
    )

    parser.add_argument(
        '--enable-tden-retrieval',
        action='store_true',
        default=True,
        help='启用 TDEN 检索模型以增强多模态链接（默认开启）'
    )

    parser.add_argument(
        '--disable-tden-retrieval',
        action='store_true',
        help='禁用 TDEN 检索模型'
    )

    parser.add_argument(
        '--enable-selective-expansion',
        action='store_true',
        help='启用有选择的节点扩展（默认关闭）'
    )

    parser.add_argument(
        '--selective-expansion-terms',
        type=str,
        default='',
        help='兼容旧参数：允许扩展为新思政节点的短语白名单，用逗号/分号分隔'
    )

    parser.add_argument(
        '--selective-expansion-computer-terms',
        type=str,
        default='',
        help='允许扩展为新计算机知识点的短语白名单，用逗号/分号分隔'
    )

    parser.add_argument(
        '--selective-expansion-ideology-terms',
        type=str,
        default='',
        help='允许扩展为新思政元素的短语白名单，用逗号/分号分隔'
    )


    parser.add_argument(
        '--selective-expansion-source-scope',
        type=str,
        choices=['caption', 'ocr', 'both', 'caption+ocr'],
        default='both',
        help='扩展时允许使用的来源范围 (默认: both)'
    )

    parser.add_argument(
        '--selective-expansion-min-support',
        type=int,
        default=1,
        help='短语被采纳为新节点的最低证据数 (默认: 1)'
    )

    parser.add_argument(
        '--selective-expansion-min-score',
        type=float,
        default=0.55,
        help='白名单短语被采纳的最低语义分数 (默认: 0.55)'
    )

    parser.add_argument(
        '--selective-expansion-max-new-ideology',
        type=int,
        default=20,
        help='最多新增多少个思政节点 (默认: 20)'
    )

    parser.add_argument(
        '--selective-expansion-max-new-computer',
        type=int,
        default=20,
        help='最多新增多少个计算机知识点节点 (默认: 20)'
    )

    parser.add_argument(
        '--selective-expansion-max-new-total',
        type=int,
        default=20,
        help='最多新增多少个选择性扩展节点总数 (默认: 20)'
    )

    args = parser.parse_args()
    
    # 验证数据目录
    if not os.path.exists(args.data_dir):
        logger.error(f"数据目录不存在: {args.data_dir}")
        sys.exit(1)
    
    # 验证txt目录
    txt_dir = os.path.join(args.data_dir, 'txt')
    if not os.path.exists(txt_dir):
        logger.error(f"文本目录不存在: {txt_dir}")
        logger.info("请在data/txt/目录下放置.txt文件")
        sys.exit(1)

    # 可选：视频兼容性预处理
    _run_video_preprocess(args)

    try:
        # 导入KG构建器
        from xmodaler.kg import KGBuilder
        
        logger.info("初始化知识图谱构建器...")
        
        # 创建构建器
        use_xmodaler = not args.no_xmodaler_video and args.use_xmodaler_video
        
        # 从命令行参数和自动检测的txt文件读取扩展白名单
        project_root = Path(__file__).resolve().parent
        default_computer_terms_file = project_root / 'data' / 'computer_terms_whitelist.txt'
        default_ideology_terms_file = project_root / 'data' / 'ideology_terms_whitelist.txt'
        
        legacy_terms = _merge_terms(
            _parse_term_list(args.selective_expansion_terms),
            _load_terms_from_file(str(default_ideology_terms_file)),
        )
        computer_terms = _merge_terms(
            _parse_term_list(args.selective_expansion_computer_terms),
            _load_terms_from_file(str(default_computer_terms_file)),
        )
        ideology_terms = _merge_terms(
            _parse_term_list(args.selective_expansion_ideology_terms),
            _load_terms_from_file(str(default_ideology_terms_file)),
            legacy_terms,
        )
        
        # 自动启用选择性扩展（如果检测到文件）
        enable_auto_expansion = default_computer_terms_file.exists() or default_ideology_terms_file.exists()
        if enable_auto_expansion:
            logger.info(f"自动检测到扩展白名单文件，自动启用选择性扩展")
            if default_computer_terms_file.exists():
                logger.info(f"  - 计算机知识点白名单: {default_computer_terms_file}")
            if default_ideology_terms_file.exists():
                logger.info(f"  - 思政元素白名单: {default_ideology_terms_file}")
            args.enable_selective_expansion = True

        builder = KGBuilder(
            neo4j_uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database,
            language=args.language,
            device_mode=args.device_mode,
            relation_model_dir=args.relation_model_dir,
            relation_threshold=max(0.0, min(1.0, float(args.relation_threshold))),
            enable_relation_rerank=not args.disable_relation_rerank,
            use_xmodaler_video=use_xmodaler,
            xmodaler_model_type=args.xmodaler_model_type,
            enable_selective_node_expansion=args.enable_selective_expansion,
            selective_expansion_terms=legacy_terms,
            selective_expansion_computer_terms=computer_terms,
            selective_expansion_ideology_terms=ideology_terms,
            selective_expansion_source_scope=args.selective_expansion_source_scope,
            selective_expansion_min_support=max(1, int(args.selective_expansion_min_support)),
            selective_expansion_min_score=max(0.0, min(1.0, float(args.selective_expansion_min_score))),
            selective_expansion_max_new_total=max(0, int(args.selective_expansion_max_new_total)),
            selective_expansion_max_new_computer=max(0, int(args.selective_expansion_max_new_computer)),
            selective_expansion_max_new_ideology=max(0, int(args.selective_expansion_max_new_ideology)),
        )
        
        logger.info(f"数据目录: {args.data_dir}")
        logger.info(f"输出目录: {args.output_dir}")
        logger.info(f"处理图像: {not args.skip_images}")
        logger.info(f"处理视频: {not args.skip_videos}")
        logger.info(f"语言: {args.language}")
        logger.info(f"设备模式: {args.device_mode}")
        logger.info(f"Neo4j数据库: {args.neo4j_database}")
        logger.info(f"匹配TopK: {args.match_top_k}")
        logger.info(f"语义阈值: {args.semantic_threshold}")
        logger.info(f"关系重排: {not args.disable_relation_rerank}")
        logger.info(f"关系阈值: {args.relation_threshold}")
        logger.info(f"选择性节点扩展: {args.enable_selective_expansion}")
        if args.enable_selective_expansion:
            logger.info(f"旧版思政白名单数量: {len(legacy_terms)}")
            logger.info(f"计算机白名单数量: {len(computer_terms)}")
            logger.info(f"思政白名单数量: {len(ideology_terms)}")
            logger.info(f"扩展来源范围: {args.selective_expansion_source_scope}")
            logger.info(f"最小证据数: {args.selective_expansion_min_support}")
            logger.info(f"最小语义分数: {args.selective_expansion_min_score}")
            logger.info(f"最大新增总节点: {args.selective_expansion_max_new_total}")
            logger.info(f"最大新增计算机节点: {args.selective_expansion_max_new_computer}")
            logger.info(f"最大新增思政节点: {args.selective_expansion_max_new_ideology}")
        logger.info(f"视频预处理: {args.video_preprocess}")
        if args.video_preprocess:
            logger.info(f"视频预处理修复: {args.video_preprocess_fix}")
            logger.info(f"视频修复输出策略: {args.video_fix_mode}")

        if args.computer_entities and args.ideology_entities:
            computer_entities = [item.strip() for item in args.computer_entities.split(',') if item.strip()]
            ideology_entities = [item.strip() for item in args.ideology_entities.split(',') if item.strip()]
            logger.info("实体来源: 命令行参数")
        else:
            computer_entities, ideology_entities = _load_entities_from_datamain(args.entity_config)
            logger.info(f"实体来源: {args.entity_config}")

        logger.info(f"预置计算机实体数: {len(computer_entities)}")
        logger.info(f"预置思政实体数: {len(ideology_entities)}")

        custom_relations = []
        if args.custom_relations:
            for rel in args.custom_relations.split(';'):
                if ':' in rel:
                    entities, rel_type = rel.split(':', 1)
                    if '-' in entities:
                        entity1, entity2 = entities.split('-', 1)
                        custom_relations.append((entity1.strip(), entity2.strip(), rel_type.strip()))
        
        # 构建知识图谱
        builder.build_kg(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            computer_entities=computer_entities,
            ideology_entities=ideology_entities,
            custom_relations=custom_relations,
            match_top_k=max(1, int(args.match_top_k)),
            semantic_threshold=max(0.0, min(1.0, float(args.semantic_threshold)))
        )
        
        # 关闭连接
        builder.close()
        
        logger.info("✓ 知识图谱构建完成!")
        logger.info(f"✓ 结果已保存到: {args.output_dir}")
        logger.info("")
        logger.info("后续步骤:")
        logger.info("1. 检查kg_output/目录下的JSON文件")
        logger.info("2. 运行 'python app.py' 启动Web服务")
        logger.info("3. 访问 http://localhost:5000 查询知识图谱")
        
        return 0
    
    except ImportError as e:
        logger.error(f"导入错误: {str(e)}")
        logger.error("请确保已安装所有依赖: pip install -r requirements.txt")
        return 1
    
    except Exception as e:
        logger.error(f"构建失败: {str(e)}")
        logger.error("请检查:")
        logger.error("  1. Neo4j数据库是否正在运行")
        logger.error("  2. 数据目录结构是否正确")
        logger.error("  3. 数据文件是否完整")
        return 1


if __name__ == '__main__':
    sys.exit(main())
