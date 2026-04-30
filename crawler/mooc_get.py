# -*- coding: utf-8 -*-


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
import json

from Cryptodome.Cipher import AES
import requests

import subprocess
import os
import tempfile

from xxlimited_35 import Null

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Node.js的完整路径
NODE_JS_PATH = r"E:\PyCharm 2025.3.1\Nodejs\node.exe"
# m3u8.js文件的绝对路径
M3U8_JS_ABS_PATH = os.path.join(SCRIPT_DIR, "m3u8.js")


def extract_js_variables(html):
    """
    从HTML中提取JavaScript变量
    """
    variables = {}

    # 1. 提取 var 声明的变量
    var_patterns = [
        (r'var\s+(\w+)\s*=\s*([^;]+);', "var变量"),
        (r'(\w+)\s*=\s*([^;]+);', "直接赋值变量"),
        (r'(\w+)\s*:\s*([^,}]+)', "对象属性")
    ]

    for pattern, pattern_name in var_patterns:
        matches = re.findall(pattern, html)
        for name, value in matches:
            # 清理值
            value = value.strip()

            # 处理字符串值
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            # 处理数字值
            elif value.isdigit():
                value = int(value)

            # 跳过太长的值（可能是复杂的对象）
            elif len(value) > 100:
                continue

            variables[name] = value

    # 2. 提取特定的重要变量
    specific_patterns = {
        'video_id': r'var\s+video_id\s*=\s*(\d+);',
        'videoTitle': r'var\s+videoTitle\s*=\s*["\']([^"\']+)["\'];',
        'OP_CONFIG.mongo_id': r'OP_CONFIG\.mongo_id\s*=\s*["\']([^"\']+)["\'];',

    }

    for key, pattern in specific_patterns.items():
        match = re.search(pattern, html)
        if match:
            variables[key] = match.group(1)
    print(variables)
    return variables

def gid(url):
    """
    使用Selenium获取慕课网页面
    """

    chrome_options = Options()

    # 无头模式
    chrome_options.add_argument('--headless')

    # 避免被检测为自动化工具
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    # 添加用户代理
    chrome_options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36')

    # 初始化浏览器
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(url)
        # 等待页面加载
        wait = WebDriverWait(driver, 10)

        # 等待标题出现
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "title")))
        time.sleep(3)  # 额外等待JS执行

        # 获取页面标题
        title = driver.title
        print(f"页面标题: {title}")

        # 获取完整HTML
        html = driver.page_source
        variables = extract_js_variables(html)

        return variables['OP_CONFIG.mongo_id'],variables['videoTitle']

    finally:
        # 关闭浏览器
        driver.quit()

def get_response_content(driver, url, headers=None):
    """
    使用Selenium执行JavaScript获取原始响应内容
    返回二进制内容，类似于requests.get(url).content
    """
    try:
        # 构建JavaScript脚本获取原始响应
        script = """
        var url = arguments[0];
        var callback = arguments[1];

        // 使用fetch API获取原始响应
        fetch(url, {
            method: 'GET',
            credentials: 'include',  // 包含cookies
            headers: {
                'Accept': '*/*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('HTTP error, status = ' + response.status);
            }
            return response.arrayBuffer();
        })
        .then(arrayBuffer => {
            // 将ArrayBuffer转换为Base64字符串
            var bytes = new Uint8Array(arrayBuffer);
            var binary = '';
            for (var i = 0; i < bytes.length; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            var base64 = btoa(binary);
            callback(base64);
        })
        .catch(error => {
            console.error('Fetch error:', error);
            callback(null);
        });
        """

        # 执行异步JavaScript代码
        result = driver.execute_async_script(script, url)

        if result:
            import base64
            # 将Base64解码为字节
            return base64.b64decode(result)
        else:
            return None

    except Exception as e:
        print(f"获取响应内容错误: {e}")
        return None

def gjson_with_content(w_url, return_raw=False):
    """
    扩展版gjson函数，可以选择返回原始响应内容或提取info字段
    return_raw: True返回原始二进制内容，False返回info字段
    """
    # 创建浏览器
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)

    try:
        # 访问首页并添加cookies
        driver.get("https://www.imooc.com")

        cookies = [
            {'name': 'imooc_uuid', 'value': 'fe3ef526-75a9-41d8-94a6-aec63819fd40', 'domain': '.imooc.com'},
            {'name': 'apsid',
             'value': 'cxY2M5MWY5NzZhYjVlYzdlOWRkOTcwMTlkM2NmNWQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMTE4MTAzNDgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADZlZmNjYzc3OTgwOGJlMGIzY2NjNmE1OWE0OTMzNDQ0LYJcaS2CXGk%3DNj',
             'domain': '.imooc.com'},
            {'name': 'loginstate', 'value': '1', 'domain': '.imooc.com'},
            {'name': 'Hm_lvt_f0cfcccd7b1393990c78efdeebff3968', 'value': '1767670287', 'domain': '.imooc.com'}
        ]

        for cookie in cookies:
            driver.add_cookie(cookie)

        # 访问API URL
        print(f"访问URL: {w_url}")

        if return_raw:
            # 获取原始二进制内容
            content = get_response_content(driver, w_url)
            driver.quit()
            return content
        else:
            # 原有逻辑，提取info字段
            driver.get(w_url)
            time.sleep(3)  # 等待页面加载

            response = driver.page_source

            pre_match = re.search(r'<pre[^>]*>(.*?)</pre>', response, re.DOTALL)
            if pre_match:
                json_str = pre_match.group(1).strip()
                #print(f"从<pre>标签提取到JSON字符串，长度: {len(json_str)}")
                #print(f"JSON前200字符: {json_str[:200]}")

                try:
                    data = json.loads(json_str)
                    if data.get('result') == 1 or data.get('code') == 200:
                        info = data.get('data', {}).get('info', '')
                        if info:
                            #print(f"从JSON中提取到info字段，长度: {len(info)}")
                            return info
                except json.JSONDecodeError as e:
                    print(f"解析JSON失败: {e}")
                    # 尝试修复JSON字符串
                    try:
                        # 移除可能的控制字符
                        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
                        data = json.loads(json_str)
                        if data.get('result') == 1 or data.get('code') == 200:
                            info = data.get('data', {}).get('info', '')
                            if info:
                                #print(f"修复后提取到info字段，长度: {len(info)}")
                                return info
                    except:
                        print("修复JSON失败")

            # 方法2: 尝试从<body>标签中提取JSON
            if '<body>' in response:
                start = response.find('<body>') + 6
                end = response.find('</body>')
                if 6 < start < end:
                    body_content = response[start:end].strip()

                    # 清理HTML标签
                    body_content = re.sub(r'<[^>]+>', '', body_content)
                    body_content = body_content.strip()

                    if body_content:
                        try:
                            data = json.loads(body_content)
                            if data.get('result') == 1 or data.get('code') == 200:
                                info = data.get('data', {}).get('info', '')
                                if info:
                                    print(f"从body内容提取到info字段，长度: {len(info)}")
                                    return info
                        except:
                            pass


            json_patterns = [
                r'\{"code":\s*200[^}]*"info"\s*:\s*"([^"]+)"',
                r'\{"result":\s*1[^}]*"info"\s*:\s*"([^"]+)"',
                r'"info"\s*:\s*"([^"]+)"'
            ]

            for pattern in json_patterns:
                matches = re.findall(pattern, response)
                for match in matches:
                    if len(match) > 10:  # 确保不是太短的字符串
                        return match

            print("所有提取方法都失败")
            return None

    except Exception as e:
        print(f"函数错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if not return_raw:  # return_raw为True时已经在函数内部关闭driver了
            driver.quit()

def gjson(w_url):
    """获取info字段"""
    return gjson_with_content(w_url, return_raw=False)

def get_binary_content(w_url):
    """获取原始二进制内容"""
    return gjson_with_content(w_url, return_raw=True)

def decode_with_embedded_js(encoded_str,eno=0):
    """使用嵌入的m3u8.js内容进行解码"""
    if not encoded_str:
        print("错误: 编码字符串为空")
        return None

    try:
        # 创建完整的JavaScript脚本
        # 这里需要确保a对象有所有需要的函数
        full_script = f"""
// 解码函数
const n = function(t, e) {{
    function r(t, e) {{
        var r = "";
        if ("object" == typeof t)
            for (var n = 0; n < t.length; n++)
                r += String.fromCharCode(t[n]);
        t = r || t;
        for (var i, o, s = new Uint8Array(t.length), a = e.length, n = 0; n < t.length; n++)
            o = n % a,
            i = t[n],
            i = i.toString().charCodeAt(0),
            s[n] = i ^ e.charCodeAt(o);
        return s
    }}
    function h(t) {{
        var e = "";
        if ("object" == typeof t)
            for (var r = 0; r < t.length; r++)
                e += String.fromCharCode(t[r]);
        t = e || t;
        var n = new Uint8Array(t.length);
        for (r = 0; r < t.length; r++)
            n[r] = t[r].toString().charCodeAt(0);
        var i, o, r = 0;
        for (r = 0; r < n.length; r++)
            0 != (i = n[r] % 3) && r + i < n.length && (o = n[r + 1],
            n[r + 1] = n[r + i],
            n[r + i] = o,
            r = r + i + 1);
        return n
    }}
    function m(t) {{
        var e = "";
        if ("object" == typeof t)
            for (var r = 0; r < t.length; r++)
                e += String.fromCharCode(t[r]);
        t = e || t;
        var n = new Uint8Array(t.length);
        for (r = 0; r < t.length; r++)
            n[r] = t[r].toString().charCodeAt(0);
        var r = 0
          , i = 0
          , o = 0
          , s = 0;
        for (r = 0; r < n.length; r++)
            o = n[r] % 2,
            o && r++,
            s++;
        var a = new Uint8Array(s);
        for (r = 0; r < n.length; r++)
            o = n[r] % 2,
            a[i++] = o ? n[r++] : n[r];
        return a
    }}
    function k(t, e) {{
        var r = 0
          , n = 0
          , i = 0
          , o = 0
          , s = "";
        if ("object" == typeof t)
            for (var r = 0; r < t.length; r++)
                s += String.fromCharCode(t[r]);
        t = s || t;
        var a = new Uint8Array(t.length);
        for (r = 0; r < t.length; r++)
            a[r] = t[r].toString().charCodeAt(0);
        for (r = 0; r < t.length; r++)
            if (0 != (o = a[r] % 5) && 1 != o && r + o < a.length && (i = a[r + 1],
            n = r + 2,
            a[r + 1] = a[r + o],
            a[o + r] = i,
            (r = r + o + 1) - 2 > n))
                for (; n < r - 2; n++)
                    a[n] = a[n] ^ e.charCodeAt(n % e.length);
        for (r = 0; r < t.length; r++)
            a[r] = a[r] ^ e.charCodeAt(r % e.length);
        return a
    }}

    var a = {{
        q: r,
        h: h,
        m: m,
        k: k
    }};

    var s = {{
        data: {{
            info: t
        }}
    }};

    var l = s.data.info;
    var u = l.substring(l.length - 4).split("");

    for (var c = 0; c < u.length; c++)
        u[c] = u[c].toString().charCodeAt(0) % 4;

    u.reverse();

    var d = [];
    for (var c = 0; c < u.length; c++) {{
        d.push(l.substring(u[c] + 1, u[c] + 2));
        l = l.substring(0, u[c] + 1) + l.substring(u[c] + 2);
    }}

    s.data.encrypt_table = d;
    s.data.key_table = [];

    for (var c in s.data.encrypt_table) {{
        if ("q" == s.data.encrypt_table[c] || "k" == s.data.encrypt_table[c]) {{
            s.data.key_table.push(l.substring(l.length - 12));
            l = l.substring(0, l.length - 12);
        }}
    }}

    s.data.key_table.reverse();
    s.data.info = l;

    var f = new Array(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,-1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1);

    s.data.info = (function(t) {{
        var e, r, n, i, o, s, a;
        for (s = t.length,
        o = 0,
        a = ""; o < s; ) {{
            do {{
                e = f[255 & t.charCodeAt(o++)]
            }} while (o < s && -1 == e);
            if (-1 == e)
                break;
            do {{
                r = f[255 & t.charCodeAt(o++)]
            }} while (o < s && -1 == r);
            if (-1 == r)
                break;
            a += String.fromCharCode(e << 2 | (48 & r) >> 4);
            do {{
                if (61 == (n = 255 & t.charCodeAt(o++)))
                    return a;
                n = f[n]
            }} while (o < s && -1 == n);
            if (-1 == n)
                break;
            a += String.fromCharCode((15 & r) << 4 | (60 & n) >> 2);
            do {{
                if (61 == (i = 255 & t.charCodeAt(o++)))
                    return a;
                i = f[i]
            }} while (o < s && -1 == i);
            if (-1 == i)
                break;
            a += String.fromCharCode((3 & n) << 6 | i)
        }}
        return a
    }})(s.data.info);

    for (var c in s.data.encrypt_table) {{
        var h = s.data.encrypt_table[c];
        if ("q" == h || "k" == h) {{
            var p = s.data.key_table.pop();
            s.data.info = a[s.data.encrypt_table[c]](s.data.info, p);
        }} else {{
            // 确保函数存在
            if (a[s.data.encrypt_table[c]]) {{
                s.data.info = a[s.data.encrypt_table[c]](s.data.info);
            }} else {{
                console.error('错误: 函数 ' + s.data.encrypt_table[c] + ' 不存在');
                return null;
            }}
        }}
    }}

    if (e)
        return s.data.info;

    var g = "";
    for (c = 0; c < s.data.info.length; c++)
        g += String.fromCharCode(s.data.info[c]);
    return g;
}};

// 解码函数
try {{
    const encodedStr = `{encoded_str}`;
    const e_no = {eno};

    const result = n(encodedStr,e_no);
    if (result) {{
        console.log(result);
    }} else {{
        console.error("解码返回null");
    }}
}} catch (error) {{
    console.error('解码错误:', error.message);
    console.error('错误堆栈:', error.stack);
    process.exit(1);
}}
"""

        # 将脚本写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8') as f:
            f.write(full_script)
            temp_file = f.name

        #print(f"创建临时JS文件: {temp_file}")

        # 执行Node.js
        result = subprocess.run(
            [NODE_JS_PATH, temp_file],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=10
        )

        # 清理临时文件
        try:
            os.unlink(temp_file)
        except:
            pass

        #print(f"Node.js返回码: {result.returncode}")

        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                #print(f"解码成功，输出长度: {len(output)}")
                return output
            else:
                print("解码返回空结果")
                if result.stderr:
                    print(f"Node.js stderr: {result.stderr[:500]}")
                return None
        else:
            print(f"Node.js错误输出: {result.stderr}")
            return None

    except Exception as e:
        print(f"执行错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Referer': 'https://www.imooc.com/',
    }
    cookies = {
        'imooc_uuid': 'fe3ef526-75a9-41d8-94a6-aec63819fd40',
        'apsid': 'cxY2M5MWY5NzZhYjVlYzdlOWRkOTcwMTlkM2NmNWQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMTE4MTAzNDgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADZlZmNjYzc3OTgwOGJlMGIzY2NjNmE1OWE0OTMzNDQ0LYJcaS2CXGk%3DNj',
        'loginstate': '1',
        'Hm_lvt_f0cfcccd7b1393990c78efdeebff3968': '1767670287',
    }

    for i in range(144,150,1):
        start_url = f'https://www.imooc.com/video/24{i}'
        _id,title = gid(start_url)
        url=f"https://www.imooc.com/course/playlist/25339?t=m3u8&_id={_id}&cdn=aliyun1"
        m3u8_info = gjson(url)
        #print(_id,m3u8_info)
        decoded_result = decode_with_embedded_js(m3u8_info,0)
        #print(decoded_result)
        m3u8_url=re.findall('0\n(.*?)\n', decoded_result)[0]

        m3u8_m = gjson(m3u8_url)
        m3u8_data = decode_with_embedded_js(m3u8_m,0)
        #print(m3u8_data)
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        key_url = re.findall('URI="(.*?)"',m3u8_data)[0]
        key_data = gjson(key_url)
        #print(key_data)
        key_str = decode_with_embedded_js(key_data,1)
        #print(key_str)

        numbers = re.findall(r'\d+', key_str)
        my_list = [int(num) for num in numbers]
        #print(my_list)
        fin_list = my_list[2::]
        #print(fin_list)

        key = bytes(fin_list)
        ic = AES.new(key, AES.MODE_CBC)

        #print(key)
        ts_list = re.findall(',\n(.*?)\n#',m3u8_data)
        for ts in ts_list:
            ts_content = requests.get(ts,headers=headers, cookies=cookies).content
            content = ic.decrypt(ts_content)
            with open('video\\' + title +'.mp4',mode='ab') as m:
                m.write(content)
            #print(ts)
        print("finish download " + title)






