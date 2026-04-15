#!/usr/bin/env python3
"""
批量检查并修复不兼容视频。

默认行为：只检查，不修改文件。
修复方式：
- separate：输出到 data/video_fixed（默认）
- inplace：原地替换，使用临时文件成功后覆盖

主要针对：
- 伪 mp4（例如扩展名是 .mp4，但容器是 mpegts）
- 浏览器 HTML5 不友好的封装/编码组合
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


MP4_CONTAINER_FAMILIES = {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}
HTML5_VIDEO_CODECS = {"h264", "avc1", "hevc", "h265", "av1", "vp9", "vp8", "mpeg4"}
HTML5_AUDIO_CODECS = {"aac", "mp3", "opus", "vorbis", "pcm_s16le"}


@dataclass
class VideoCheckResult:
    path: str
    exists: bool
    container: str = ""
    video_codec: str = ""
    audio_codec: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0
    issues: List[str] = None
    browser_compatible: bool = False

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["issues"] = self.issues or []
        return data


def _run(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _ffprobe_json(path: Path) -> Optional[Dict]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    code, out, err = _run(cmd)
    if code != 0:
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


def _extract_stream_info(probe: Dict) -> Tuple[str, str, str, int, int, float]:
    fmt = probe.get("format", {}) or {}
    container = str(fmt.get("format_name", "") or "")
    duration = float(fmt.get("duration", 0.0) or 0.0)
    video_codec = ""
    audio_codec = ""
    width = 0
    height = 0
    for stream in probe.get("streams", []) or []:
        codec_type = str(stream.get("codec_type", "") or "")
        codec_name = str(stream.get("codec_name", "") or "")
        if codec_type == "video" and not video_codec:
            video_codec = codec_name
            width = int(stream.get("width", 0) or 0)
            height = int(stream.get("height", 0) or 0)
        elif codec_type == "audio" and not audio_codec:
            audio_codec = codec_name
    return container, video_codec, audio_codec, width, height, duration


def check_video(path: Path) -> VideoCheckResult:
    result = VideoCheckResult(path=str(path), exists=path.exists(), issues=[])
    if not path.exists():
        result.issues.append("文件不存在")
        return result

    probe = _ffprobe_json(path)
    if not probe:
        result.issues.append("ffprobe 无法读取文件")
        return result

    container, video_codec, audio_codec, width, height, duration = _extract_stream_info(probe)
    result.container = container
    result.video_codec = video_codec
    result.audio_codec = audio_codec
    result.width = width
    result.height = height
    result.duration = duration

    ext = path.suffix.lower()
    container_tokens = {token.strip().lower() for token in container.split(",") if token.strip()}

    if ext == ".mp4" and not (container_tokens & MP4_CONTAINER_FAMILIES):
        result.issues.append(f"扩展名为 .mp4，但容器是 {container or 'unknown'}")

    if "mpegts" in container_tokens and ext == ".mp4":
        result.issues.append("伪 MP4：实际是 MPEG-TS 封装")

    if video_codec and video_codec.lower() not in HTML5_VIDEO_CODECS:
        result.issues.append(f"视频编码 {video_codec} 可能不兼容浏览器")

    if audio_codec and audio_codec.lower() not in HTML5_AUDIO_CODECS:
        result.issues.append(f"音频编码 {audio_codec} 可能不兼容浏览器")

    if not video_codec:
        result.issues.append("未检测到视频流")

    result.browser_compatible = len(result.issues) == 0
    return result


def _ensure_output_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base


def _remux_to_mp4(input_path: Path, output_path: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-map", "0",
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]
    code, _, _ = _run(cmd)
    return code == 0 and output_path.exists()


def _reencode_to_mp4(input_path: Path, output_path: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-map", "0:v:0?",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-sn",
        str(output_path),
    ]
    code, _, _ = _run(cmd)
    return code == 0 and output_path.exists()


def repair_video(input_path: Path, mode: str, output_dir: Path) -> Dict:
    check = check_video(input_path)
    if not check.exists:
        return {"path": str(input_path), "fixed": False, "error": "文件不存在", "check": check.to_dict()}

    if check.browser_compatible:
        return {"path": str(input_path), "fixed": False, "skipped": True, "reason": "已兼容", "check": check.to_dict()}

    if mode == "separate":
        _ensure_output_dir(output_dir)
        target = output_dir / f"{input_path.stem}.mp4"
    else:
        target = input_path.with_suffix(input_path.suffix + ".tmp.mp4")

    repaired = _remux_to_mp4(input_path, target)
    strategy = "remux"
    if not repaired:
        repaired = _reencode_to_mp4(input_path, target)
        strategy = "reencode"

    if not repaired:
        if target.exists():
            try:
                target.unlink()
            except Exception:
                pass
        return {
            "path": str(input_path),
            "fixed": False,
            "error": "ffmpeg 修复失败",
            "check": check.to_dict(),
        }

    final_path = target
    if mode == "inplace":
        backup = input_path.with_suffix(input_path.suffix + ".bak")
        try:
            if backup.exists():
                backup.unlink()
            shutil.move(str(input_path), str(backup))
            os.replace(str(target), str(input_path))
            final_path = input_path
        except Exception as exc:
            return {
                "path": str(input_path),
                "fixed": False,
                "error": f"原地替换失败: {exc}",
                "check": check.to_dict(),
            }

    repaired_check = check_video(final_path)
    return {
        "path": str(input_path),
        "fixed": True,
        "strategy": strategy,
        "output_path": str(final_path),
        "check_before": check.to_dict(),
        "check_after": repaired_check.to_dict(),
    }


def iter_videos(root: Path, extensions: Optional[List[str]] = None):
    exts = {e.lower() for e in (extensions or [".mp4", ".avi", ".mov", ".mkv", ".ts", ".m4v"])}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量检查并修复不兼容视频")
    parser.add_argument("video_dir", nargs="?", default="data/video", help="视频目录，默认 data/video")
    parser.add_argument("--fix", action="store_true", help="执行修复；默认仅检查")
    parser.add_argument("--mode", choices=["separate", "inplace"], default=os.getenv("VIDEO_FIX_MODE", "separate"), help="修复模式：separate 或 inplace（默认 separate）")
    parser.add_argument("--output-dir", default=os.getenv("VIDEO_FIX_OUTPUT_DIR", "data/video_fixed"), help="separate 模式输出目录，默认 data/video_fixed")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少个文件，0 表示不限制")
    parser.add_argument("--report", default="", help="可选：输出 JSON 报告路径")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    root = Path(args.video_dir).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if not root.exists():
        print(f"[ERROR] 目录不存在: {root}")
        return 2

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    videos = list(iter_videos(root))
    if args.limit and args.limit > 0:
        videos = videos[: args.limit]

    if not videos:
        print(f"[INFO] 未找到视频文件: {root}")
        return 0

    reports: List[Dict] = []
    bad_count = 0
    fixed_count = 0
    skipped_count = 0

    print(f"[INFO] 扫描到 {len(videos)} 个视频文件")
    print(f"[INFO] 模式: {'修复' if args.fix else '仅检查'} | 输出模式: {args.mode}")

    for idx, path in enumerate(videos, 1):
        check = check_video(path)
        has_issue = not check.browser_compatible
        if has_issue:
            bad_count += 1
        print(f"[{idx}/{len(videos)}] {path.name}")
        if check.exists:
            print(f"  - 容器: {check.container or 'unknown'} | 视频: {check.video_codec or 'unknown'} | 音频: {check.audio_codec or 'unknown'}")
            print(f"  - 分辨率: {check.width}x{check.height} | 时长: {check.duration:.1f}s")
        if check.issues:
            for issue in check.issues:
                print(f"  ! {issue}")
        else:
            print("  ✓ 浏览器兼容")

        item_report: Dict = {"check": check.to_dict()}
        if args.fix and has_issue:
            result = repair_video(path, args.mode, output_dir)
            item_report["repair"] = result
            if result.get("fixed"):
                fixed_count += 1
                print(f"  -> 已修复: {result.get('output_path')} ({result.get('strategy')})")
            else:
                print(f"  -> 修复失败: {result.get('error', 'unknown')}")
        elif args.fix:
            skipped_count += 1
            print("  -> 跳过（已兼容）")
        reports.append({"path": str(path), **item_report})

    summary = {
        "scanned": len(videos),
        "incompatible": bad_count,
        "fixed": fixed_count,
        "skipped": skipped_count,
        "mode": args.mode,
        "fix_enabled": bool(args.fix),
        "video_dir": str(root),
        "output_dir": str(output_dir),
    }

    print("\n[SUMMARY]", json.dumps(summary, ensure_ascii=False, indent=2))

    if args.report:
        report_path = Path(args.report).expanduser()
        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"summary": summary, "items": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] JSON 报告已保存: {report_path}")

    return 0 if (not args.fix or fixed_count >= 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())

