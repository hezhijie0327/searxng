# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quark search engine for searxng"""

from urllib.parse import urlencode
import re
import json

from searx.utils import html_to_text

# Metadata
about = {
    "website": "https://quark.sm.cn/",
    "wikidata_id": None,
    "use_official_api": False,
    "require_api_key": False,
    "results": "JSON",
    "language": "zh",
}

# Engine Configuration
categories = ["general"]
paging = True
time_range_support = False  # Quark暂不支持时间范围过滤

# Base URL
base_url = "https://quark.sm.cn/s"

# Cookies needed for requests
cookies = {
    'x5sec': '7b22733b32223a2238336538313064336131336531636562222c2277616762726964676561643b32223a223633336237343538343031396264353265663535333661623261626632353431434d485330373447454d7a513359662f2f2f2f2f2f77456f42444459746475652b502f2f2f2f3842227d'
}

# Headers for requests
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://quark.sm.cn/"
}

def request(query, params):
    """Generate search request parameters"""
    query_params = {
        "q": query,
        "layout": "html",
        "page": params["pageno"]
    }

    params["url"] = f"{base_url}?{urlencode(query_params)}"
    params["cookies"] = cookies
    params["headers"] = headers
    return params

def response(resp):
    """Parse search results from Quark"""
    results = []
    html_content = resp.text

    # 改进的正则表达式匹配
    pattern = r'<script\s+type="application/json"\s+id="s-data-[^"]+"\s+data-used-by="hydrate">(.*?)</script>'
    matches = re.findall(pattern, html_content, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            initial_data = data.get('data', {}).get('initialData', {})
            sc_type = data.get('extraData', {}).get('sc', '')  # 获取搜索结果类型

            # 根据不同结构提取数据
            if sc_type == 'nature_result':
                # 处理视频/文章类型
                title = initial_data.get('title', '')
                content = initial_data.get('desc', '')
                link = initial_data.get('url', '')
            elif sc_type in ['baike', 'ss_text', 'ss_pic']:
                # 处理百科/文本/图片类型
                title = initial_data.get('titleProps', {}).get('content', '')
                content = initial_data.get('summaryProps', {}).get('content', '')
                link = initial_data.get('nuProps', {}).get('nu', '') or \
                       initial_data.get('sourceProps', {}).get('dest_url', '')
            else:
                # 处理其他类型
                title = initial_data.get('title', '') or initial_data.get('titleProps', {}).get('content', '')
                content = initial_data.get('desc', '') or initial_data.get('summaryProps', {}).get('content', '')
                link = initial_data.get('url', '') or initial_data.get('nuProps', {}).get('nu', '') or \
                       initial_data.get('sourceProps', {}).get('dest_url', '')

            # 清理HTML标签和转义字符
            clean_title = html_to_text(title)
            clean_content = html_to_text(content)

            # 过滤无效数据
            if clean_title and clean_content and link:
                results.append({
                    "title": clean_title.strip(),
                    "url": link.strip(),
                    "content": clean_content.strip()
                })
        except json.JSONDecodeError:
            continue
        except KeyError as e:
            # 记录关键字段缺失错误（可选）
            continue

    return results
