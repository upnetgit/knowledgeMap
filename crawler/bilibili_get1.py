import requests
import re
import json
from DrissionPage import ChromiumPage

def getResponse(url):
    headers = {
        'cookie':'buvid3=F0DC9FE3-A0FF-191B-ED58-DEB7FF5396C048908infoc; b_nut=1768360548; _uuid=1AC85EE4-EA2C-C951-71036-810F725D10103E1050805infoc; home_feed_column=5; browser_resolution=1536-686; buvid_fp=85cf874449aaa6f26a40849761fd8761; buvid4=9EF70B1E-5704-9DA6-3743-A5A03BECCD6450317-026011411-V2TRSDnHpRUypoBctNFojI+J+mcPz0JEz42fssKFW/WCjoSLhC8rxDTMV1UZLssB; CURRENT_QUALITY=0; rpdid=0zbfvSgdTX|9XmdhgZN|4Fl|3w1VFRn3; SESSDATA=11840c29%2C1783913643%2C57457%2A12CjCNPPvnb_-rKkuoFhsQHmhSHCAS-enLe4YTSvuK22tVuo3kb7HevxZL4h0mkHiMoD8SVkROUE1OQjBKQUFmSGE3Rk5LVGoyWHNIcF9yNFBya0xBZm5jVllxQ2JrbkpFODZKaVdaN2hwaFNuaEhadjJjQmY2QzVMbFZGMldpbGRWQ19QWHNDUW1nIIEC; bili_jct=c6d2b9395020225b3401d2403153c417; DedeUserID=433250180; DedeUserID__ckMd5=440b1c3a75c15dc6; sid=5vdjbpnr; theme-tip-show=SHOWED; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3Njg2MzE0MzIsImlhdCI6MTc2ODM3MjE3MiwicGx0IjotMX0.CHrBjdwWgoWig2sUvhOyEmUU2J03KrMezy--T65vN1M; bili_ticket_expires=1768631372; theme-avatar-tip-show=SHOWED; b_lsid=743B93104_19BBBB4A228; CURRENT_FNVAL=2000',
        'referer':'https://search.bilibili.com/all?keyword=%E6%80%9D%E6%94%BF&from_source=webtop_search&spm_id_from=333.1007&search_source=2',
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }
    response = requests.get(url=url,headers=headers)
    return response

def getVideo(bv):
    link = f'https://www.bilibili.com/video/{bv}/?spm_id_from=333.337.search-card.all.click&vd_source=623812403a19922c528216ae139efeb4'
    response = getResponse(link)
    html = response.text
    info = re.findall('<script>window.__playinfo__=(.*?)</script>',html)[0]
    title = re.findall('<h1 data-title="(.*?)"',html)[0]

    json_data = json.loads(info)
    audio_url = json_data['data']['dash']['audio'][0]['baseUrl']
    video_url = json_data['data']['dash']['video'][0]['baseUrl']
    #print(title,audio_url,video_url)
    return title,audio_url,video_url

def save(title,audio_url,video_url):
    audio_content = getResponse(url=audio_url).content
    video_content = getResponse(url=video_url).content

    with open('video\\' + title + '.mp4', mode='wb') as video:
        # 使用临时目录处理
        import tempfile
        import os
        from moviepy import VideoFileClip, AudioFileClip

        # 手动管理临时文件，避免自动清理时的占用问题
        temp_dir = tempfile.mkdtemp()

        try:
            # 写入临时文件
            video_path = os.path.join(temp_dir, "temp_video.mp4")
            audio_path = os.path.join(temp_dir, "temp_audio.mp3")

            with open(video_path, "wb") as f:
                f.write(video_content)
            with open(audio_path, "wb") as f:
                f.write(audio_content)

            # 加载视频和音频
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)

            final_clip = video_clip.with_audio(audio_clip)

            # 写入临时输出文件
            output_path = os.path.join(temp_dir, "merged.mp4")
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=video_clip.fps,
                logger=None
            )

            # 读取合并后的视频内容
            with open(output_path, "rb") as f:
                merged_content = f.read()

        finally:
            # 确保所有剪辑对象都被关闭
            try:
                if 'video_clip' in locals():
                    video_clip.close()
            except:
                pass

            try:
                if 'audio_clip' in locals():
                    audio_clip.close()
            except:
                pass

            try:
                if 'final_clip' in locals():
                    final_clip.close()
            except:
                pass

            # 手动清理临时文件
            import shutil
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass

        # 将合并后的内容写入最终文件
        video.write(merged_content)




if __name__ == '__main__':
    driver = ChromiumPage()
    driver.listen.start('api.bilibili.com/x/space/wbi/arc/search')
    driver.get('https://space.bilibili.com/244239984/upload/video')
    resp = driver.listen.wait()
    json_data = resp.response.body
    for Page in range(2):
        for index in json_data['data']['list']['vlist']:
            bv = index['bvid']
            title,audio_url,video_url = getVideo(bv=bv)
            print(title)
            save(title, audio_url, video_url)
        driver.ele('css:.vui_button vui_pagenation--btn vui_pagenation--btn-side').click()
        print("final "+ Page)
