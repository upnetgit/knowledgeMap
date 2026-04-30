import argparse
import json
import os
import re
import shutil
import tempfile
import time
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote
import requests
from moviepy import AudioFileClip, VideoFileClip

try:
    from DrissionPage import ChromiumPage
except Exception:
    ChromiumPage = None


SESSION = requests.Session()
HEADERS = {
    'cookie':'buvid3=F0DC9FE3-A0FF-191B-ED58-DEB7FF5396C048908infoc; b_nut=1768360548; _uuid=1AC85EE4-EA2C-C951-71036-810F725D10103E1050805infoc; home_feed_column=5; browser_resolution=1536-686; buvid_fp=85cf874449aaa6f26a40849761fd8761; buvid4=9EF70B1E-5704-9DA6-3743-A5A03BECCD6450317-026011411-V2TRSDnHpRUypoBctNFojI+J+mcPz0JEz42fssKFW/WCjoSLhC8rxDTMV1UZLssB; CURRENT_QUALITY=0; rpdid=0zbfvSgdTX|9XmdhgZN|4Fl|3w1VFRn3; SESSDATA=11840c29%2C1783913643%2C57457%2A12CjCNPPvnb_-rKkuoFhsQHmhSHCAS-enLe4YTSvuK22tVuo3kb7HevxZL4h0mkHiMoD8SVkROUE1OQjBKQUFmSGE3Rk5LVGoyWHNIcF9yNFBya0xBZm5jVllxQ2JrbkpFODZKaVdaN2hwaFNuaEhadjJjQmY2QzVMbFZGMldpbGRWQ19QWHNDUW1nIIEC; bili_jct=c6d2b9395020225b3401d2403153c417; DedeUserID=433250180; DedeUserID__ckMd5=440b1c3a75c15dc6; sid=5vdjbpnr; theme-tip-show=SHOWED; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3Njg2MzE0MzIsImlhdCI6MTc2ODM3MjE3MiwicGx0IjotMX0.CHrBjdwWgoWig2sUvhOyEmUU2J03KrMezy--T65vN1M; bili_ticket_expires=1768631372; theme-avatar-tip-show=SHOWED; b_lsid=743B93104_19BBBB4A228; CURRENT_FNVAL=2000',
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
    ),
    "referer": "https://search.bilibili.com/all",
}


def clean_filename(name: str, max_len: int = 120) -> str:
    """清理 Windows/Linux 文件名非法字符。"""
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', '_', str(name).strip())
    name = re.sub(r'\s+', ' ', name).strip().rstrip('. ')
    if not name:
        name = 'untitled'
    return name[:max_len]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_response(url: str, params: Optional[dict] = None, stream: bool = False) -> requests.Response:
    resp = SESSION.get(url=url, headers=HEADERS, params=params, timeout=30, stream=stream)
    resp.raise_for_status()
    return resp


def _response_body_to_json(body) -> Optional[Dict]:
    if body is None:
        return None
    if isinstance(body, dict):
        return body
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", errors="ignore")
    if isinstance(body, str):
        return json.loads(body)
    return None


def _create_browser_page():
    if ChromiumPage is None:
        return None
    try:
        return ChromiumPage()
    except Exception:
        return None


def _sync_session_cookies_from_browser(browser_page) -> bool:
    """将浏览器会话 cookie 同步到 requests.Session。"""
    if browser_page is None:
        return False
    try:
        cookie_items = []
        if hasattr(browser_page, "cookies"):
            try:
                cookie_items = browser_page.cookies(all_info=True)
            except Exception:
                cookie_items = browser_page.cookies()
        if not cookie_items:
            return False

        for item in cookie_items:
            if isinstance(item, dict):
                name = item.get("name")
                value = item.get("value")
                domain = item.get("domain")
                path = item.get("path", "/")
                if name and value is not None:
                    SESSION.cookies.set(name, value, domain=domain, path=path)
        return True
    except Exception:
        return False


def _search_videos_by_requests(keyword: str, page: int = 1, page_size: int = 20, referer: Optional[str] = None) -> Dict:
    url = "https://api.bilibili.com/x/web-interface/search/type"
    params = {
        "search_type": "video",
        "keyword": keyword,
        "page": page,
        "page_size": page_size,
    }
    headers = dict(HEADERS)
    if referer:
        headers["referer"] = referer
    resp = SESSION.get(url=url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def search_videos(keyword: str, page: int = 1, page_size: int = 20, browser_page=None) -> List[Dict]:
    """按关键词搜索视频，优先使用浏览器上下文，避免静态请求触发风控。"""
    data = None
    search_url = (
        "https://search.bilibili.com/all?"
        f"keyword={quote(keyword)}"
        f"&from_source=webtop_search&search_source=2"
        f"&page={page}"
    )

    if browser_page is not None:
        try:
            browser_page.listen.start("api.bilibili.com/x/web-interface/search/type")
            browser_page.get(search_url)
            packet = browser_page.listen.wait(timeout=12)
            body = getattr(getattr(packet, "response", None), "body", None)
            data = _response_body_to_json(body)

            if not isinstance(data, dict):
                # 某些版本/场景下 listen 能抓到包但 body 为空，此时用浏览器 cookie 回退。
                _sync_session_cookies_from_browser(browser_page)
                data = _search_videos_by_requests(keyword, page, page_size, referer=search_url)
        except Exception as exc:
            print(f"[WARN] 浏览器搜索失败，回退 requests: {exc}")
            try:
                _sync_session_cookies_from_browser(browser_page)
                data = _search_videos_by_requests(keyword, page, page_size, referer=search_url)
            except Exception:
                data = None

    if data is None:
        data = _search_videos_by_requests(keyword, page, page_size, referer=search_url)

    result = data.get("data", {}).get("result", [])
    videos: List[Dict] = []
    for item in result:
        bvid = item.get("bvid") or item.get("bvids")
        title = item.get("title") or item.get("author") or bvid
        if not bvid:
            continue
        videos.append(
            {
                "bvid": bvid,
                "title": re.sub(r"<.*?>", "", str(title)).strip(),
                "author": str(item.get("author") or "").strip(),
                "mid": str(item.get("mid") or item.get("mid_str") or "").strip(),
                "raw": item,
            }
        )
    return videos


def matches_uploader(item: Dict, up_name: Optional[str] = None, up_mid: Optional[str] = None) -> bool:
    """判断搜索结果是否属于指定 UP 主。"""
    if not up_name and not up_mid:
        return True

    author = str(item.get("author") or "").strip()
    mid = str(item.get("mid") or "").strip()
    if up_mid and mid and str(up_mid).strip() == mid:
        return True
    if up_name and author and str(up_name).strip() == author:
        return True
    return False


def get_video(bv: str, browser_page=None) -> Tuple[str, str, str]:
    link = f"https://www.bilibili.com/video/{bv}/"
    html = ""
    if browser_page is not None:
        try:
            browser_page.get(link)
            time.sleep(1)
            html = getattr(browser_page, "html", "") or ""
        except Exception as exc:
            print(f"[WARN] 浏览器打开视频页失败，回退 requests: {exc}")

    if not html:
        response = get_response(link)
        html = response.text

    info_match = re.findall(r"<script>window.__playinfo__=(.*?)</script>", html)
    title_match = re.findall(r'<h1[^>]*data-title="(.*?)"', html)
    if not info_match:
        raise RuntimeError(f"无法解析视频播放信息: {bv}")
    if not title_match:
        title = bv
    else:
        title = re.sub(r"<.*?>", "", title_match[0])

    json_data = json.loads(info_match[0])
    dash = json_data.get("data", {}).get("dash", {})
    audio_list = dash.get("audio", [])
    video_list = dash.get("video", [])
    if not audio_list or not video_list:
        raise RuntimeError(f"视频流解析失败: {bv}")

    audio_url = audio_list[0]["baseUrl"]
    video_url = video_list[0]["baseUrl"]
    return title, audio_url, video_url


def save(title: str, audio_url: str, video_url: str, output_dir: str = "video") -> Tuple[str, bool]:
    """下载视频和音频并合并成 mp4。返回 (输出路径, 是否已跳过)。"""
    ensure_dir(output_dir)
    safe_title = clean_filename(title)
    output_file = os.path.join(output_dir, f"{safe_title}.mp4")

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        return output_file, True

    audio_content = get_response(url=audio_url, stream=True).content
    video_content = get_response(url=video_url, stream=True).content

    temp_dir = tempfile.mkdtemp(prefix="bili_merged_")
    video_clip = None
    audio_clip = None
    final_clip = None
    try:
        video_path = os.path.join(temp_dir, "temp_video.mp4")
        audio_path = os.path.join(temp_dir, "temp_audio.m4a")

        with open(video_path, "wb") as f:
            f.write(video_content)
        with open(audio_path, "wb") as f:
            f.write(audio_content)

        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        final_clip = video_clip.with_audio(audio_clip)

        merged_path = os.path.join(temp_dir, "merged.mp4")
        final_clip.write_videofile(
            merged_path,
            codec="libx264",
            audio_codec="aac",
            fps=video_clip.fps,
            logger=None,
        )

        with open(merged_path, "rb") as f:
            merged_content = f.read()

        with open(output_file, "wb") as f:
            f.write(merged_content)

        return output_file, False
    finally:
        for clip in (final_clip, video_clip, audio_clip):
            try:
                if clip is not None:
                    clip.close()
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)


def iter_keyword_videos(
    keyword: str,
    start_page: int,
    max_pages: int,
    page_size: int,
    up_name: Optional[str] = None,
    up_mid: Optional[str] = None,
    browser_page=None,
) -> Iterable[Dict]:
    seen = set()
    for page in range(start_page, start_page + max_pages):
        try:
            items = search_videos(keyword=keyword, page=page, page_size=page_size, browser_page=browser_page)
        except Exception as exc:
            print(f"[WARN] 搜索第 {page} 页失败: {exc}")
            break

        if not items:
            break

        for item in items:
            bvid = item["bvid"]
            if bvid in seen:
                continue
            if not matches_uploader(item, up_name=up_name, up_mid=up_mid):
                continue
            seen.add(bvid)
            yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="B站关键词搜索爬虫（下载视频并合并音视频）")
    parser.add_argument("keyword", help="搜索关键词，例如：思政、数据库、Java")
    parser.add_argument("--start-page", type=int, default=1, help="起始页码，默认 1")
    parser.add_argument("--max-pages", type=int, default=2, help="最多抓取页数，默认 2")
    parser.add_argument("--page-size", type=int, default=20, help="每页条数，默认 20")
    parser.add_argument("--limit", type=int, default=0, help="最多下载多少个视频，0 表示不限制")
    parser.add_argument("--output-dir", default="video", help="输出目录，默认 video")
    parser.add_argument("--up-name", default="", help="只下载指定 UP 主昵称的视频，可选")
    parser.add_argument("--up-mid", default="", help="只下载指定 UP 主 mid 的视频，可选")
    parser.add_argument("--no-browser", action="store_true", help="强制只用 requests，不启用浏览器上下文")
    args = parser.parse_args()

    browser_page = None if args.no_browser else _create_browser_page()
    if browser_page is None and not args.no_browser:
        print("[WARN] 未能初始化浏览器上下文，将使用 requests 回退；若遇 412，请安装并登录 DrissionPage 浏览器会话。")

    downloaded = 0
    for item in iter_keyword_videos(
        args.keyword,
        args.start_page,
        args.max_pages,
        args.page_size,
        up_name=args.up_name or None,
        up_mid=args.up_mid or None,
        browser_page=browser_page,
    ):
        bvid = item["bvid"]
        author = item.get("author") or "未知UP主"
        try:
            title, audio_url, video_url = get_video(bvid, browser_page=browser_page)
            print(f"[INFO] {title} ({bvid}) - {author}")
            output_path, skipped = save(title, audio_url, video_url, output_dir=args.output_dir)
            if skipped:
                print(f"[SKIP] 已存在: {output_path}")
            else:
                print(f"[OK] 保存到: {output_path}")
            downloaded += 1
        except Exception as exc:
            print(f"[WARN] 下载失败 {bvid}: {exc}")
            continue

        if args.limit and downloaded >= args.limit:
            break

    try:
        if browser_page is not None and hasattr(browser_page, "close"):
            browser_page.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()

