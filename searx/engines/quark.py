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
    'x5sec': '7b22733b32223a2264643037333165383064613134303931222c2277616762726964676561643b32223a223366306638356463613432316133623963323363373135643966356164306661434b7a76314c3447454e6253707534454b415177324c58626e766a2f2f2f2f2f41513d3d227d'
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

    pattern = r'<script\s+type="application/json"\s+id="s-data-[^"]+"\s+data-used-by="hydrate">(.*?)</script>'
    matches = re.findall(pattern, html_content, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            initial_data = data.get('data', {}).get('initialData', {})

            title = (
                #initial_data.get('title', {}).get('content') or
                initial_data.get('title') or
                initial_data.get('titleProps', {}).get('content') or
                initial_data.get('props', [{}])[0].get('title')
            )

            content = (
                initial_data.get('desc') or
                #initial_data.get('summary', {}).get('content') or
                initial_data.get('summaryProps', {}).get('content') or
                initial_data.get('props', [{}])[0].get('summary')
            )

            link = (
                initial_data.get('url') or
                initial_data.get('nuProps', {}).get('nu') or
                #initial_data.get('source', {}).get('dest_url') or
                initial_data.get('sourceProps', {}).get('dest_url') or
                initial_data.get('title', {}).get('dest_url') or
                initial_data.get('props', [{}])[0].get('dest_url')
            )

            if title and content:
                results.append({
                    "title": html_to_text(title),
                    "url": link,
                    "content": html_to_text(content)
                })
        except json.JSONDecodeError:
            continue
        except KeyError as e:
            continue

    return results
