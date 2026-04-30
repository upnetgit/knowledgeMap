#http://221.122.117.73/docinpic.jsp?sid=" + (o = void 0 !== readerConfig.flash_param_hzq ? readerConfig.flash_param_hzq : o) + "&file=" + r + "&width=40&pageno=" + s
import requests
import re
import pprint
import os

headers={
    'cookie':'mobilefirsttip=tip; docin_session_id=dd8b717b-02ce-4dfb-b9de-059e93754bbe; jumpIn=400; isbaiduspider=false; saveFinProductToCookieValue=33609623; cookie_id=CB848A44FD2000012BCCB9B09D4090A0; time_id=202612117450; remindClickId=-1; downloadClickId=-1; booksaveClickId=-1; payReadClickId=-1; partnerLogin=-1; partner_tips=1; vip_alert_adv=-1; can_copy_alert=-1; payReadClickId_v2=-1; showFeekClickId=-1; addComdocs=-1; showShareClickId=-1; _ga_ZYR13KTSXC=GS2.1.s1768988701$o1$g0$t1768988701$j60$l0$h0; _ga=GA1.1.738213653.1768988701; editOnlineloadClickId=-1; JSESSIONID=CE88AFD92F7B28149063B7D58EF8A99B',
    'referer':'https://www.docin.com/d-41955.html',
    'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0'
}
file=688958170
url = f'https://www.docin.com/p-{file}.html'
response = requests.get(url=url,headers=headers)
html = response.text
#print(html)
sid = re.findall('flash_param_hzq:"(.*?)"',html)[0]
print(sid)
name = re.findall('productName:"(.*?)"',html)[0]
print(name)
page = re.findall(r'<span class="info_txt"><em>(\d+)</em>页</span><span class="info_txt">',html)[0]
print(page)

os.makedirs(f'img\\{name}', exist_ok=True)
for p in range(1,int(page)+1):
    img = f'http://221.122.117.73/docinpic.jsp?sid={sid}&file={file}&width=940&pageno={p}'
    content = requests.get(url=img,headers=headers).content
    with open(f'img\\{name}\\{str(p)}.jpg', mode='wb') as f:
        f.write(content)
    print("finish:"+str(p))
print("download:"+name)
