# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quark (Shenma) search engine for searxng"""

from urllib.parse import urlencode
import re
import json

from searx.utils import html_to_text, searx_useragent

# Metadata
about = {
    "website": "https://quark.sm.cn/",
    "wikidata_id": "Q48816502",
    "use_official_api": False,
    "require_api_key": False,
    "results": "HTML",
    "language": "zh",
}

# Engine Configuration
categories = ["general"]
paging = True
time_range_support = True

# Base URL
base_url = "https://quark.sm.cn"

time_range_dict = {'day': '4', 'week': '3', 'month': '2', 'year': '1'}

cookie_x5sec = ''


def is_quark_captcha(html):
    pattern = r'\{[^{]*?"action"\s*:\s*"captcha"\s*,\s*"url"\s*:\s*"([^"]+)"[^{]*?\}'
    match = re.search(pattern, html)

    if match:
        captcha_url = match.group(1)
        raise SearxEngineCaptchaException(suspended_time=0, message=f"CAPTCHA ({captcha_url})")


def request(query, params):
    query_params = {
        "q": query,
        "layout": "html",
        "page": params["pageno"]
    }

    if time_range_dict.get(params['time_range']):
        query_params["tl_request"] = time_range_dict.get(params['time_range'])

    params["url"] = f"{base_url}/s?{urlencode(query_params)}"
    params["cookies"] = {
        'x5sec': cookie_x5sec,
    }
    params["headers"] = {
        "User-Agent": searx_useragent(),
    }
    return params


def response(resp):
    results = []
    html_content = resp.text

    is_quark_captcha(html_content)

    pattern = r'<script\s+type="application/json"\s+id="s-data-[^"]+"\s+data-used-by="hydrate">(.*?)</script>'
    matches = re.findall(pattern, html_content, re.DOTALL)

    for match in matches:
        try:
            title = None
            content = None
            url = None

            data = json.loads(match)
            initial_data = data.get('data', {}).get('initialData', {})
            extra_data = data.get('extraData', {})

            if extra_data['sc'] == 'ss_note':
                title = initial_data.get('title', {}).get('content')
                content = initial_data.get('summary', {}).get('content')
                url = initial_data.get('source', {}).get('dest_url')

            if extra_data['sc'] == 'ss_pic' or extra_data['sc'] == 'ss_text':
                title = initial_data.get('titleProps', {}).get('content')
                content = initial_data.get('summaryProps', {}).get('content')
                url = initial_data.get('sourceProps', {}).get('dest_url')

            if extra_data['sc'] == 'baike_sc':
                title = initial_data.get('data', {}).get('title')
                content = initial_data.get('data', {}).get('abstract')
                url = initial_data.get('data', {}).get('url')

            if extra_data['sc'] == 'finance_shuidi':
                title = initial_data.get('company_name')
                content = initial_data.get('business_scope')
                url = initial_data.get('title_url')

            if extra_data['sc'] == 'nature_result':
                title = initial_data.get('title')
                content = initial_data.get('desc')
                url = initial_data.get('url')

            if extra_data['sc'] == 'news_uchq':
                feed_items = initial_data.get('feed', [])
                for item in feed_items:
                    title = item.get('title')
                    content = item.get('summary')
                    url = item.get('url')

                    if title and content:
                        results.append({
                            "title": html_to_text(title),
                            "url": url,
                            "content": html_to_text(content)
                        })
                # skip dups append for news_uchq
                continue

            if title and content:
                results.append({
                    "title": html_to_text(title),
                    "url": url,
                    "content": html_to_text(content)
                })
        except json.JSONDecodeError:
            continue
        except KeyError as e:
            continue

    return results
