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
    render_template_string = _flask_module.render_template_string
    send_from_directory = _flask_module.send_from_directory
except Exception:
    Flask = None
    request = jsonify = render_template_string = send_from_directory = None

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

if Flask is None or request is None or jsonify is None or render_template_string is None or send_from_directory is None:
    raise ImportError("Flask is required to run app.py")

app = Flask(__name__)


def _safe_text(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(',') if item.strip()]

# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>跨模态知识图谱查询系统</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
            padding: 30px;
        }
        
        .sidebar {
            border-right: 2px solid #eee;
            padding-right: 20px;
        }
        
        .section {
            margin-bottom: 30px;
        }
        
        .section h2 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        input[type="text"] {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            padding: 12px 25px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: background 0.3s;
        }
        
        button:hover {
            background: #764ba2;
        }
        
        .results {
            background: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .result-item {
            background: white;
            padding: 12px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
            border-radius: 3px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .result-item:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transform: translateX(5px);
        }
        
        .result-item .entity-name {
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        
        .result-item .relation {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .result-item .media-info {
            font-size: 0.85em;
            color: #999;
        }
        
        #graph-container {
            border: 1px solid #ddd;
            border-radius: 5px;
            background: #f8f9fa;
            height: 600px;
            position: relative;
        }
        
        .legend {
            margin-top: 15px;
            font-size: 0.9em;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .legend-color {
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 10px;
        }
        
        .tooltip {
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.9em;
            pointer-events: none;
            z-index: 1000;
        }
        
        .stats {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .stat-item {
            background: white;
            padding: 12px;
            border-radius: 5px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        
        .video-container {
            margin-top: 20px;
            background: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
        }
        
        .video-item {
            margin-bottom: 15px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #5cb85c;
        }
        
        .video-item video {
            width: 100%;
            max-height: 300px;
            border-radius: 5px;
        }
        
        .entity-section {
            margin-bottom: 20px;
        }
        
        .entity-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .entity-tag {
            background: #e9ecef;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
            color: #495057;
        }

        .recommend-card {
            background: #fff;
            border-left: 4px solid #667eea;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 10px;
        }

        .recommend-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 4px;
        }

        .recommend-meta {
            font-size: 0.86em;
            color: #666;
            margin-bottom: 4px;
        }

        .recommend-risk {
            font-size: 0.85em;
            color: #b94a48;
        }

        .recommend-suggest {
            font-size: 0.85em;
            color: #3c763d;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 跨模态知识图谱查询系统</h1>
            <p>连接计算机和思想实体的多模态知识网络</p>
        </div>
        
        <div class="content">
            <div class="sidebar">
                <div class="section">
                    <h2>🔍 查询</h2>
                    <div class="search-box">
                        <input type="text" id="searchInput" placeholder="输入实体名称...">
                        <button onclick="searchEntity()">查询</button>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 结果</h2>
                    <div class="results" id="results"></div>
                </div>
                
                <div class="section">
                    <h2>📈 统计</h2>
                    <div id="stats"></div>
                </div>
            </div>
            
            <div class="main">
                <div class="section">
                    <h2>🗺️ 知识图谱可视化</h2>
                    <div id="graph-container"></div>
                    <div class="legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background: #667eea;"></div>
                            <span>文本实体 (连接媒体数)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #f0ad4e;"></div>
                            <span>图像 (图像标注)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #5cb85c;"></div>
                            <span>视频 (视频标注)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #d9534f;"></div>
                            <span>实体关系</span>
                        </div>
                    </div>
                </div>
                
                <div class="section entity-section">
                    <h2>📂 实体详情</h2>
                    <div id="entityDetails"></div>
                </div>
                
                <div class="section video-container">
                    <h2>📹 视频播放</h2>
                    <div id="videoPlayer"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // 初始化知识图谱
        function initializeGraph() {
            fetch('/api/graph')
                .then(response => response.json())
                .then(data => {
                    renderGraph(data);
                    updateStats(data.stats);
                })
                .catch(error => console.error('Error:', error));
        }
        
        // 渲染知识图谱
        function renderGraph(data) {
            const container = document.getElementById('graph-container');
            container.innerHTML = ''; // 清空容器
            
            const width = container.offsetWidth;
            const height = container.offsetHeight;
            
            const svg = d3.select('#graph-container')
                .append('svg')
                .attr('width', width)
                .attr('height', height);
            
            const simulation = d3.forceSimulation(data.nodes)
                .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2));
            
            const links = svg.selectAll('line')
                .data(data.links)
                .enter()
                .append('line')
                .attr('stroke', '#999')
                .attr('stroke-opacity', 0.6)
                .attr('stroke-width', d => Math.sqrt(d.strength) * 2);
            
            const nodes = svg.selectAll('circle')
                .data(data.nodes)
                .enter()
                .append('circle')
                .attr('r', d => 5 + d.connections * 2)
                .attr('fill', d => {
                    if (d.type === 'image') return '#f0ad4e';
                    if (d.type === 'video') return '#5cb85c';
                    return '#667eea';
                })
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));
            
            const labels = svg.selectAll('text')
                .data(data.nodes)
                .enter()
                .append('text')
                .attr('x', 0)
                .attr('y', 0)
                .attr('dy', '.35em')
                .attr('text-anchor', 'middle')
                .text(d => d.id.substring(0, 10))
                .attr('font-size', '12px')
                .attr('fill', '#333');
            
            simulation.on('tick', () => {
                links
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                nodes
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
                
                labels
                    .attr('x', d => d.x)
                    .attr('y', d => d.y);
            });
            
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
        }
        
        // 搜索实体
        function searchEntity() {
            const entity = document.getElementById('searchInput').value;
            if (!entity) return;
            
            fetch(`/api/query?entity=${encodeURIComponent(entity)}`)
                .then(response => response.json())
                .then(data => {
                    displayResults(data);
                    displayEntityDetails(entity);
                })
                .catch(error => console.error('Error:', error));
        }
        
        // 显示搜索结果
        function displayResults(results) {
            const container = document.getElementById('results');
            container.innerHTML = '';
            
            if (results.length === 0) {
                container.innerHTML = '<div style="color: #999;">未找到相关实体</div>';
                return;
            }
            
            results.forEach(result => {
                const item = document.createElement('div');
                item.className = 'result-item';
                item.innerHTML = `
                    <div class="entity-name">${result.entity}</div>
                    <div class="relation">关系: ${result.relation}</div>
                    <div class="media-info">媒体: ${result.media || 'N/A'}</div>
                `;
                container.appendChild(item);
            });
        }
        
        // 显示实体详情
        function displayEntityDetails(entity) {
            const container = document.getElementById('entityDetails');
            container.innerHTML = `<div style="color: #666;">加载中...</div>`;
            
            fetch(`/api/query_advanced?entity=${encodeURIComponent(entity)}`)
                .then(response => response.json())
                .then(data => {
                    container.innerHTML = '';
                    
                    // 显示相似实体
                    if (data.similar.length > 0) {
                        const similarSection = document.createElement('div');
                        similarSection.className = 'section';
                        similarSection.innerHTML = '<h3>相似实体</h3>';
                        
                        data.similar.forEach(item => {
                            const tag = document.createElement('div');
                            tag.className = 'entity-tag';
                            tag.innerText = item.entity;
                            similarSection.appendChild(tag);
                        });
                        
                        container.appendChild(similarSection);
                    }
                    
                    // 显示相关实体
                    if (data.related.length > 0) {
                        const relatedSection = document.createElement('div');
                        relatedSection.className = 'section';
                        relatedSection.innerHTML = '<h3>相关实体</h3>';
                        
                        data.related.forEach(item => {
                            const tag = document.createElement('div');
                            tag.className = 'entity-tag';
                            tag.innerText = item.entity;
                            relatedSection.appendChild(tag);
                        });
                        
                        container.appendChild(relatedSection);
                    }

                    // 显示教学推荐（Phase4.5: 可解释推理结果）
                    if (data.recommendations && data.recommendations.length > 0) {
                        const recSection = document.createElement('div');
                        recSection.className = 'section';
                        recSection.innerHTML = '<h3>教学推荐</h3>';

                        data.recommendations.forEach(item => {
                            const card = document.createElement('div');
                            card.className = 'recommend-card';

                            const viaHint = item.via_entity ? ` | 通过: ${item.via_entity}` : '';
                            const hopHint = item.hop ? ` | 跳数: ${item.hop}` : '';
                            const stageHint = (item.stage_tags || []).length ? ` | 学段: ${(item.stage_tags || []).join('/')}` : '';
                            const courseHint = (item.course_tags || []).length ? ` | 课程: ${(item.course_tags || []).slice(0, 3).join('/')}` : '';

                            card.innerHTML = `
                                <div class="recommend-title">${item.entity || '未命名推荐'}（评分: ${item.score ?? 0}）</div>
                                <div class="recommend-meta">关系: ${item.relation || 'UNKNOWN'}${viaHint}${hopHint}${stageHint}${courseHint}</div>
                                <div class="recommend-meta">理由: ${item.reason || '无'}</div>
                                <div class="recommend-risk">风险: ${item.risk_note || '风险可控'}</div>
                                <div class="recommend-suggest">建议: ${item.suggested_use || '可直接用于课堂推荐'}</div>
                            `;
                            recSection.appendChild(card);
                        });

                        container.appendChild(recSection);
                    }
                    
                    // 显示视频
                    if (data.videos.length > 0) {
                        const videoSection = document.getElementById('videoPlayer');
                        videoSection.innerHTML = '';
                        
                        data.videos.forEach(video => {
                            const videoItem = document.createElement('div');
                            videoItem.className = 'video-item';
                            const videoSrc = video.clip_path || video.clip_url || video.path || '';
                            videoItem.innerHTML = `
                                <h4>${video.name}</h4>
                                <p>${video.caption}</p>
                                <video controls>
                                    <source src="${videoSrc}" type="video/mp4">
                                    您的浏览器不支持视频标签。
                                </video>
                            `;
                            videoSection.appendChild(videoItem);
                        });
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    container.innerHTML = `<div style="color: red;">加载失败</div>`;
                });
        }
        
        // 更新统计信息
        function updateStats(stats) {
            const container = document.getElementById('stats');
            container.innerHTML = `
                <div class="stats">
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value">${stats.total_nodes}</div>
                            <div class="stat-label">实体数量</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">${stats.total_edges}</div>
                            <div class="stat-label">关系数量</div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // 页面加载时初始化
        document.addEventListener('DOMContentLoaded', initializeGraph);
    </script>
</body>
</html>
'''

MANUAL_ANNOTATION_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>视频人工标注</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f7fb; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; padding: 20px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { border: 1px solid #ddd; border-radius: 8px; padding: 16px; }
        h1, h2 { margin-top: 0; }
        label { display: block; margin-top: 10px; font-weight: bold; }
        input, select, textarea, button { width: 100%; padding: 8px; margin-top: 6px; box-sizing: border-box; }
        textarea { min-height: 100px; }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .status { margin-top: 12px; color: #1f6f43; }
        .error { color: #b42318; }
    </style>
</head>
<body>
<div class="container">
    <h1>视频人工标注（可选步骤）</h1>
    <p>该页面仅用于人工补充标注，不会影响现有自动构建流程。</p>
    <div class="grid">
        <div class="panel">
            <h2>视频播放</h2>
            <label for="videoSelect">选择视频</label>
            <select id="videoSelect"></select>
            <video id="videoPlayer" controls style="width:100%; margin-top:12px; max-height:420px;"></video>
            <div class="row">
                <button type="button" onclick="markStart()">记录开始时间</button>
                <button type="button" onclick="markEnd()">记录结束时间</button>
            </div>
        </div>
        <div class="panel">
            <h2>标注信息</h2>
            <form id="annotationForm" onsubmit="saveAnnotation(event)">
                <div class="row">
                    <div>
                        <label for="startSec">开始秒</label>
                        <input id="startSec" type="number" step="0.1" min="0" value="0">
                    </div>
                    <div>
                        <label for="endSec">结束秒</label>
                        <input id="endSec" type="number" step="0.1" min="0" value="60">
                    </div>
                </div>
                <label for="computerEntity">计算机知识点</label>
                <select id="computerEntity" required></select>
                <label for="ideologyEntity">思政元素</label>
                <select id="ideologyEntity" required></select>
                <label for="caption">人工描述</label>
                <textarea id="caption" placeholder="例如：讲解栈和队列差异，强调逻辑推理。"></textarea>
                <label for="ocrText">字幕/OCR文本</label>
                <textarea id="ocrText" placeholder="可选，填写视频中关键文字。"></textarea>
                <label for="confidence">置信度 (0~1)</label>
                <input id="confidence" type="number" min="0" max="1" step="0.01" value="0.85">
                <label for="annotator">标注人</label>
                <input id="annotator" type="text" placeholder="可选">
                <button type="submit" style="margin-top:12px;">保存标注</button>
            </form>
            <div id="status" class="status"></div>
        </div>
    </div>
</div>

<script>
let videoItems = [];

function showStatus(message, isError=false) {
    const box = document.getElementById('status');
    box.className = isError ? 'status error' : 'status';
    box.textContent = message;
}

function secondsNow() {
    const player = document.getElementById('videoPlayer');
    return Number(player.currentTime || 0).toFixed(1);
}

function markStart() {
    document.getElementById('startSec').value = secondsNow();
}

function markEnd() {
    document.getElementById('endSec').value = secondsNow();
}

function updateVideoPlayer() {
    const idx = Number(document.getElementById('videoSelect').value || 0);
    const item = videoItems[idx];
    const player = document.getElementById('videoPlayer');
    if (!item) {
        player.removeAttribute('src');
        return;
    }
    player.src = item.url;
    player.load();
}

async function loadVideos() {
    const resp = await fetch('/api/annotation/videos');
    const data = await resp.json();
    videoItems = data.videos || [];
    const select = document.getElementById('videoSelect');
    select.innerHTML = '';
    videoItems.forEach((item, idx) => {
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = item.name;
        select.appendChild(opt);
    });
    select.onchange = updateVideoPlayer;
    updateVideoPlayer();
}

function fillSelect(id, values) {
    const select = document.getElementById(id);
    select.innerHTML = '';
    (values || []).forEach((value) => {
        const opt = document.createElement('option');
        opt.value = value;
        opt.textContent = value;
        select.appendChild(opt);
    });
}

async function loadEntities() {
    const resp = await fetch('/api/annotation/entities');
    const data = await resp.json();
    fillSelect('computerEntity', data.computer_entities || []);
    fillSelect('ideologyEntity', data.ideology_entities || []);
}

async function saveAnnotation(event) {
    event.preventDefault();
    const idx = Number(document.getElementById('videoSelect').value || 0);
    const item = videoItems[idx];
    if (!item) {
        showStatus('请先选择视频。', true);
        return;
    }

    const payload = {
        video_name: item.name,
        video_path: item.relative_path,
        start_sec: Number(document.getElementById('startSec').value || 0),
        end_sec: Number(document.getElementById('endSec').value || 0),
        computer_entity: document.getElementById('computerEntity').value,
        ideology_entity: document.getElementById('ideologyEntity').value,
        caption: document.getElementById('caption').value,
        ocr_text: document.getElementById('ocrText').value,
        confidence: Number(document.getElementById('confidence').value || 0.85),
        annotator: document.getElementById('annotator').value,
    };

    const resp = await fetch('/api/annotation/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) {
        showStatus(data.error || '保存失败', true);
        return;
    }
    showStatus('保存成功: ' + (data.annotation_id || '')); 
}

async function init() {
    try {
        await loadEntities();
        await loadVideos();
    } catch (err) {
        showStatus('初始化失败: ' + err, true);
    }
}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
'''


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


def _list_videos_for_annotation() -> List[Dict[str, str]]:
    videos: List[Dict[str, str]] = []
    data_root = _data_root()
    suffixes = {'.mp4', '.avi', '.mov', '.mkv'}
    for subdir in ('video', 'clips'):
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
            })
    return videos


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


class Neo4jQuery:
    """Neo4j数据库查询器"""

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 database: str = "neo4j",
                 semantic_model_dir: Optional[str] = None,
                 semantic_model_profile: str = "bert_cn_base"):
        self.uri = uri
        self.database = database or "neo4j"
        self.semantic_model_dir = semantic_model_dir
        self.semantic_model_profile = str(semantic_model_profile or "bert_cn_base").strip().lower()
        self.semantic = SemanticScorer(
            model_dir=semantic_model_dir,
            model_name=self.semantic_model_profile,
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
        ]
        return _safe_text(' '.join(str(part) for part in parts if part))

    def _stage_match_bonus(self, text: str, stage: Optional[str]) -> float:
        if not stage:
            return 0.0
        text = _safe_text(text)
        stage = _safe_text(stage)
        if not text or not stage:
            return 0.0
        aliases = {
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
            'SIMILAR': 0.18,
            'RELATED': 0.12,
            'MENTIONS': 0.10,
            'LINKS_TO_CASE': 0.15,
            'MEDIA_LINKED_IMAGE': 0.08,
            'MEDIA_LINKED_VIDEO': 0.08,
        }.get(relation, 0.05)
        label_bonus = 0.15 if 'TeachingCase' in labels else 0.08 if 'Media' in labels else 0.04
        score = min(1.0, 0.45 * similarity + relation_bonus + label_bonus)

        if self.semantic and focus:
            score = min(1.0, score + 0.20 * self.semantic.similarity(focus, text))
        elif focus:
            score = min(1.0, score + 0.08 * (1.0 if focus in text else 0.0))

        score = min(1.0, score + self._stage_match_bonus(text, stage))
        score = min(1.0, score + self._duration_bonus(item, duration))
        score = min(1.0, score + self._course_bonus(text, course))
        score = min(1.0, score + self._prerequisite_bonus(text, prerequisites))
        score = min(1.0, score + self._objective_bonus(text, objective))

        reason_parts = [f"关系:{relation}"]
        if 'TeachingCase' in labels:
            reason_parts.append('教学案例')
        elif 'Video' in labels:
            reason_parts.append('视频媒体')
        elif 'Image' in labels:
            reason_parts.append('图像媒体')
        if stage:
            reason_parts.append(f"适配学段:{stage}")
        if duration:
            reason_parts.append(f"时长参考:{duration}")
        if focus:
            reason_parts.append(f"思政侧重:{focus}")
        if course:
            reason_parts.append(f"课程:{course}")
        if prerequisites:
            reason_parts.append(f"先修:{'/'.join(prerequisites[:3])}")
        if objective:
            reason_parts.append(f"目标:{objective}")
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
                    MATCH (a:Entity {name: $entity})-[r1]-(m:Entity)-[r2]-(b:Entity)
                    WHERE b.name <> $entity AND m.name <> b.name
                    RETURN b.name as entity,
                           type(r2) as relation,
                           coalesce(r2.similarity, 0.0) as similarity,
                           coalesce(r2.caption, '') as caption,
                           labels(b) as labels,
                           coalesce(b.summary, '') as summary,
                           coalesce(b.path, '') as path,
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
                return [dict(record) for record in result]
        except Exception as e:
            logger.warning(f"二跳候选查询失败: {e}")
            return []

    def query_connected_entities(self, entity: str) -> List[Dict]:
        if not self.driver:
            return []

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (a:Entity {name: $entity})-[r]-(b:Entity)
                    RETURN b.name as entity,
                           type(r) as relation,
                           coalesce(r.caption, '') as caption,
                           coalesce(r.similarity, 0.0) as similarity,
                           labels(b) as labels,
                           coalesce(b.summary, '') as summary,
                           coalesce(b.path, '') as path,
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
                """, entity=entity)

                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            return []

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
                           coalesce(n.caption, '') as caption,
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
                        'caption': record['caption'],
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

                links = [
                    {
                        'source': record['source'],
                        'target': record['target'],
                        'type': record['type'],
                        'strength': max(float(record['similarity'] or 0.0), 0.3),
                        'caption': record['caption'],
                    }
                    for record in edges_result
                ]

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
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (a:Entity {name: $entity})-[r]-(b:Entity)
                    RETURN b.name as entity,
                           type(r) as relation,
                           coalesce(r.similarity, 0.0) as similarity,
                           coalesce(r.caption, '') as caption,
                           labels(b) as labels,
                           coalesce(b.summary, '') as summary,
                           coalesce(b.path, '') as path,
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
                           coalesce(b.height, 0) as height
                    LIMIT 100
                """, entity=entity)

                items = [dict(record) for record in result]
                items.extend(self._query_two_hop_candidates(entity, limit=max(50, top_k * 12)))

                # 按实体去重，保留同实体中分数更高的候选
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
                            'score': score,
                            'reason': reason,
                            'risk_note': risk_note,
                            'suggested_use': suggested_use,
                            'via_entity': item.get('via_entity', ''),
                            'hop': item.get('hop', 1),
                        })

                    if 'Video' in labels or item.get('media_type') == 'video':
                        video_path = item.get('path')
                        clip_path = video_path
                        if video_path and os.path.exists(video_path) and self.video_editor:
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
                                if not self.video_editor.clip_video(video_path, clip_path, start_time, 60):
                                    clip_path = video_path

                        clip_url = _media_url_for_path(clip_path)
                        video_url = _media_url_for_path(video_path)

                        videos.append({
                            'name': item['entity'],
                            'caption': item.get('caption') or item.get('summary', ''),
                            'clip_path': clip_url,
                            'clip_file': clip_path,
                            'path': video_url,
                            'source_file': video_path,
                            'score': score,
                            'reason': reason,
                            'risk_note': risk_note,
                            'suggested_use': suggested_use,
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
                recommendations.sort(key=lambda x: x.get('score', 0.0), reverse=True)

                return {
                    'similar': similar_entities[:top_k],
                    'related': related_entities[:top_k],
                    'videos': videos[:top_k],
                    'recommendations': recommendations[:top_k],
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
                "recommendations": [],
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
        )
    return neo4j_query


def close_neo4j_query():
    """进程退出时关闭全局Neo4j连接，避免每个请求重复关闭。"""
    global neo4j_query
    if neo4j_query is not None:
        neo4j_query.close()
        neo4j_query = None


atexit.register(close_neo4j_query)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/annotate')
def annotate_page():
    if not _annotation_enabled():
        return jsonify({"error": "Manual annotation is disabled"}), 403
    return render_template_string(MANUAL_ANNOTATION_TEMPLATE)


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
    computer_entity = _safe_text(payload.get('computer_entity'))
    ideology_entity = _safe_text(payload.get('ideology_entity'))

    if not video_name or not video_path:
        return jsonify({'error': 'video_name 和 video_path 不能为空'}), 400
    if computer_entity not in computer_set:
        return jsonify({'error': 'computer_entity 不在预定义列表中'}), 400
    if ideology_entity not in ideology_set:
        return jsonify({'error': 'ideology_entity 不在预定义列表中'}), 400

    try:
        start_sec = max(0.0, float(payload.get('start_sec', 0.0)))
        end_sec = max(start_sec, float(payload.get('end_sec', start_sec)))
        confidence = float(payload.get('confidence', 0.85))
    except (TypeError, ValueError):
        return jsonify({'error': '数值字段格式错误'}), 400

    annotation_id = f"{video_name}__{start_sec:.1f}_{end_sec:.1f}__{computer_entity}__{ideology_entity}"
    record = {
        'annotation_id': annotation_id,
        'video_name': video_name,
        'video_path': video_path,
        'video_url': f"/media/{video_path}",
        'start_sec': round(start_sec, 1),
        'end_sec': round(end_sec, 1),
        'computer_entity': computer_entity,
        'ideology_entity': ideology_entity,
        'caption': _safe_text(payload.get('caption')),
        'ocr_text': _safe_text(payload.get('ocr_text')),
        'confidence': max(0.0, min(1.0, confidence)),
        'annotator': _safe_text(payload.get('annotator')),
        'created_at': datetime.utcnow().isoformat() + 'Z',
    }

    with _annotation_file().open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

    return jsonify({'ok': True, 'annotation_id': annotation_id, 'record': record})


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
    app.config['ENABLE_MANUAL_ANNOTATION'] = os.getenv('ENABLE_MANUAL_ANNOTATION', 'false').lower() in {'1', 'true', 'yes', 'on'}

    logger.info("启动Flask服务器...")
    logger.info("访问: http://localhost:5000")
    logger.info(f"人工标注页面: {'已启用' if app.config['ENABLE_MANUAL_ANNOTATION'] else '未启用'}")
    logger.info(f"查询语义模型档位: {app.config.get('QUERY_BERT_PROFILE', 'bert_cn_base')}")
    if app.config['QUERY_BERT_MODEL_DIR']:
        logger.info(f"查询语义模型: {app.config['QUERY_BERT_MODEL_DIR']}")

    app.run(debug=True, host='0.0.0.0', port=5000)
