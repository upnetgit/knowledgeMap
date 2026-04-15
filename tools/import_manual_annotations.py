#!/usr/bin/env python3
"""
将人工视频标注(JSONL)导入Neo4j。

默认读取: kg_output/manual_video_annotations.jsonl
"""

import os
import json
import ast
import argparse
import importlib
from pathlib import Path


def load_entity_sets(project_root: Path):
    datamain = project_root / 'BERT_cn' / 'datamain.txt'
    if not datamain.exists():
        return set(), set()

    computer_entities = set()
    ideology_entities = set()
    for raw_line in datamain.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or '=' not in line:
            continue
        key, value = [part.strip() for part in line.split('=', 1)]
        if key == 'COMPUTER_LABELS':
            computer_entities = {str(item).strip() for item in ast.literal_eval(value) if str(item).strip()}
        elif key == 'IDEOLOGY_LABELS':
            ideology_entities = {str(item).strip() for item in ast.literal_eval(value) if str(item).strip()}
    return computer_entities, ideology_entities


def iter_records(jsonl_path: Path):
    if not jsonl_path.exists():
        return
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main():
    parser = argparse.ArgumentParser(description='导入人工视频标注到Neo4j')
    parser.add_argument('--annotation-file', type=str, default='kg_output/manual_video_annotations.jsonl')
    parser.add_argument('--neo4j-uri', type=str, default=os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
    parser.add_argument('--neo4j-user', type=str, default=os.getenv('NEO4J_USERNAME') or os.getenv('NEO4J_USER', 'neo4j'))
    parser.add_argument('--neo4j-password', type=str, default=os.getenv('NEO4J_PASSWORD', 'password'))
    parser.add_argument('--neo4j-database', type=str, default=os.getenv('NEO4J_DATABASE', 'neo4j'))
    args = parser.parse_args()

    try:
        _neo4j = importlib.import_module('neo4j')
        graph_database = _neo4j.GraphDatabase
    except Exception as e:
        raise ImportError("neo4j driver is required: pip install neo4j") from e

    project_root = Path(__file__).resolve().parent.parent
    annotation_path = Path(args.annotation_file)
    if not annotation_path.is_absolute():
        annotation_path = project_root / annotation_path

    computer_set, ideology_set = load_entity_sets(project_root)

    driver = graph_database.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    imported = 0
    skipped = 0

    with driver.session(database=args.neo4j_database) as session:
        for record in iter_records(annotation_path):
            computer_entity = str(record.get('computer_entity', '')).strip()
            ideology_entity = str(record.get('ideology_entity', '')).strip()
            if computer_set and computer_entity not in computer_set:
                skipped += 1
                continue
            if ideology_set and ideology_entity not in ideology_set:
                skipped += 1
                continue

            video_name = str(record.get('video_name', '')).strip()
            if not video_name:
                skipped += 1
                continue

            session.run(
                """
                MERGE (v:Entity:Media:Video {name: $video_name})
                SET v.path = $video_path,
                    v.media_type = 'video',
                    v.caption = $caption,
                    v.ocr_text = $ocr_text,
                    v.last_manual_annotation_at = $created_at

                MERGE (k:Entity:KnowledgePoint {name: $computer_entity})
                MERGE (i:Entity:IdeologyElement {name: $ideology_entity})

                MERGE (k)-[r1:MEDIA_LINKED_VIDEO]->(v)
                SET r1.type = 'MEDIA_LINKED_VIDEO',
                    r1.similarity = $confidence,
                    r1.caption = $caption,
                    r1.media_type = 'video',
                    r1.media_path = $video_path,
                    r1.ocr_text = $ocr_text,
                    r1.start_sec = $start_sec,
                    r1.end_sec = $end_sec,
                    r1.annotation_id = $annotation_id,
                    r1.annotator = $annotator,
                    r1.created_at = $created_at

                MERGE (k)-[r2:COMPUTER_REFLECTS_IDEOLOGY]->(i)
                SET r2.type = 'COMPUTER_REFLECTS_IDEOLOGY',
                    r2.similarity = CASE WHEN r2.similarity IS NULL THEN $confidence ELSE max(r2.similarity, $confidence) END,
                    r2.caption = $caption,
                    r2.annotation_id = $annotation_id,
                    r2.annotator = $annotator,
                    r2.created_at = $created_at
                """,
                video_name=video_name,
                video_path=str(record.get('video_path', '')).strip(),
                caption=str(record.get('caption', '')).strip(),
                ocr_text=str(record.get('ocr_text', '')).strip(),
                computer_entity=computer_entity,
                ideology_entity=ideology_entity,
                confidence=float(record.get('confidence', 0.85) or 0.85),
                start_sec=float(record.get('start_sec', 0.0) or 0.0),
                end_sec=float(record.get('end_sec', 0.0) or 0.0),
                annotation_id=str(record.get('annotation_id', '')).strip(),
                annotator=str(record.get('annotator', '')).strip(),
                created_at=str(record.get('created_at', '')).strip(),
            )
            imported += 1

    driver.close()
    print(f'导入完成: imported={imported}, skipped={skipped}, file={annotation_path}')


if __name__ == '__main__':
    main()

