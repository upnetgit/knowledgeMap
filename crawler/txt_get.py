import requests
import re
import pprint

def getResponse(url):
    headers = {
        'cookie':'__jsluid_s=544f38ff4e7227b39b8f7f38b0e0845a; __51vcke__3KMKPwwkYWTr4RrO=e9ffcc24-fb6b-5fb9-9162-5b5bd96ebcdd; __51vuft__3KMKPwwkYWTr4RrO=1768554254614; __51uvsct__3KMKPwwkYWTr4RrO=2; __vtins__3KMKPwwkYWTr4RrO=%7B%22sid%22%3A%20%228849bf31-3f2b-5b8b-af30-712be8fb8287%22%2C%20%22vd%22%3A%204%2C%20%22stt%22%3A%20308683%2C%20%22dr%22%3A%20268481%2C%20%22expires%22%3A%201768622458023%2C%20%22ct%22%3A%201768620658023%7D',
        'referer':'https://www.sinobook.com.cn/b2c/scrp/book.cfm?sFieldName=bname&name1=1',
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }
    response = requests.get(url=url,headers=headers)
    return response

def getPage(keyword, base_url="https://www.baidu.com/"):

    search_url = f"{base_url.rstrip('/')}/search/newssearch1.cfm"

    gbk_keyword_bytes = keyword.encode('gbk')

    form_data = {
        'sKeyword': gbk_keyword_bytes,
        'aaa': b'1'
    }

    headers = {
        'cookie': '__jsluid_s=544f38ff4e7227b39b8f7f38b0e0845a; __51vcke__3KMKPwwkYWTr4RrO=e9ffcc24-fb6b-5fb9-9162-5b5bd96ebcdd; __51vuft__3KMKPwwkYWTr4RrO=1768554254614; __51uvsct__3KMKPwwkYWTr4RrO=2; __vtins__3KMKPwwkYWTr4RrO=%7B%22sid%22%3A%20%228849bf31-3f2b-5b8b-af30-712be8fb8287%22%2C%20%22vd%22%3A%204%2C%20%22stt%22%3A%20308683%2C%20%22dr%22%3A%20268481%2C%20%22expires%22%3A%201768622458023%2C%20%22ct%22%3A%201768620658023%7D',
        'referer': 'https://www.sinobook.com.cn/b2c/scrp/book.cfm?sFieldName=bname&name1=1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'
    }
    response = requests.post(search_url, data=form_data, headers=headers, timeout=10)
    response.encoding = 'gbk'

    return response.text

def getMenu(num):
    link = f'https://www.sinobook.com.cn/b2c/scrp/bookdetail.cfm?iBookNo={num}'
    response = getResponse(link)
    html = response.text
    #print(html)
    extracted_content = {}

    pattern = r'<td class="tdCaptionD"[^>]*>([^<]+)：</td>.*?<td class="Text"[^>]*>(.*?)</td>'
    matches = re.findall(pattern, html, re.DOTALL | re.MULTILINE)

    pattern_simple = r'class="tdbname">\s*(.+?)\s*<a'
    name = re.findall(pattern_simple, html, re.DOTALL)[0]
    name = re.sub(r'<[^>]+>', '', name)
    name = re.sub(r'[\r\n\t]+', ' ', name)
    name = name.strip()

    for title, content in matches:
        clean_title = title.strip().replace('：', '').replace(':', '')
        clean_title = ''.join(clean_title.split())

        clean_content = content.strip()

        clean_content = clean_content.replace('&nbsp;', '')
        clean_content = re.sub(r'<br\s*/?>', '\n', clean_content, flags=re.IGNORECASE)
        clean_content = re.sub(r'<P>', '\n', clean_content, flags=re.IGNORECASE)
        clean_content = re.sub(r'</P>', '', clean_content, flags=re.IGNORECASE)
        clean_content = re.sub(r'<[^>]+>', '', clean_content)
        clean_content = re.sub(r'\n\s*\n', '\n\n', clean_content)
        clean_content = clean_content.strip()

        extracted_content[clean_title] = clean_content
    return name,extracted_content

def save(sections, filename='extracted_content'):
    with open('txt\\' + filename + '.txt', mode='w', encoding='utf-8') as f:
        for title in ['内容简介', '章节目录']:
            content = sections.get(title, "（未找到）")
            f.write(f"【{title}】\n")
            f.write("=" * 50 + "\n")

            if content and not content.isspace():
                f.write(content + "\n")
            else:
                f.write("（空）\n")

            f.write("\n\n")

    print(f"已保存到文件: {filename}")

if __name__ == '__main__':

    search_keyword = "思政"
    target_website = "https://www.sinobook.com.cn/b2c/scrp/book.cfm?sFieldName=bname&name1=1"
    result = getPage(search_keyword, target_website)
    answer = re.findall(r'<a href="bookdetail\.cfm\?iBookNo=(.*?)" target="_blank" style=" font-size:14px;">', result)
    print(answer)
    for index in answer:
        name,content = getMenu(index)
        save(content,name)
