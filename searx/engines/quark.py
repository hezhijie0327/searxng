# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quark (Shenma) search engine for searxng"""

from urllib.parse import urlencode
import re
import json

from searx.utils import html_to_text, searx_useragent
from searx.exceptions import SearxEngineCaptchaException

# Metadata
about = {
    "website": "https://m.quark.cn/",
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
base_url = "https://m.quark.cn"

time_range_dict = {'day': '4', 'week': '3', 'month': '2', 'year': '1'}


def is_alibaba_captcha(html):
    """
    Detects if the response contains an Alibaba X5SEC CAPTCHA page.

    Quark may return a CAPTCHA challenge after 9 requests in a short period.

    Typically, the ban duration is around 15 minutes.
    """
    pattern = r'\{[^{]*?"action"\s*:\s*"captcha"\s*,\s*"url"\s*:\s*"([^"]+)"[^{]*?\}'
    match = re.search(pattern, html)

    if match:
        captcha_url = match.group(1)
        raise SearxEngineCaptchaException(suspended_time=900, message=f"Alibaba CAPTCHA: {captcha_url}")


def request(query, params):
    query_params = {"q": query, "layout": "html", "page": params["pageno"]}

    if time_range_dict.get(params['time_range']):
        query_params["tl_request"] = time_range_dict.get(params['time_range'])

    params["url"] = f"{base_url}/s?{urlencode(query_params)}"
    params["headers"] = {
        "User-Agent": searx_useragent(),
    }
    return params


def response(resp):
    results = []
    html_content = resp.text

    is_alibaba_captcha(html_content)

    pattern = r'<script\s+type="application/json"\s+id="s-data-[^"]+"\s+data-used-by="hydrate">(.*?)</script>'
    matches = re.findall(pattern, html_content, re.DOTALL)

    for match in matches:
        data = json.loads(match)
        initial_data = data.get('data', {}).get('initialData', {})
        extra_data = data.get('extraData', {})

        sc = extra_data.get('sc')

        title, content, url = None, None, None

        # General
        if sc == 'ss_pic' or sc == 'ss_text':
            title = initial_data.get('titleProps', {}).get('content')
            content = initial_data.get('summaryProps', {}).get('content')
            url = initial_data.get('sourceProps', {}).get('dest_url')

        if sc == 'ss_note':
            title = initial_data.get('title', {}).get('content')
            content = initial_data.get('summary', {}).get('content')
            url = initial_data.get('source', {}).get('dest_url')

        if sc == 'nature_result':
            title = initial_data.get('title')
            content = initial_data.get('desc')
            url = initial_data.get('url')

        if sc == 'addition':
            title = initial_data.get('title', {}).get('content')
            content = initial_data.get('summary', {}).get('content')
            url = initial_data.get('source', {}).get('url')

        # Baike (Wiki)
        if sc == 'baike_sc':
            title = initial_data.get('data', {}).get('title')
            content = initial_data.get('data', {}).get('abstract')
            url = initial_data.get('data', {}).get('url')

        # Shuidi (Company DB)
        if sc == 'finance_shuidi':
            title = initial_data.get('company_name')
            content = " | ".join(
                filter(
                    None,
                    [
                        initial_data.get('establish_time'),
                        initial_data.get('company_status'),
                        initial_data.get('controled_type'),
                        initial_data.get('company_type'),
                        initial_data.get('capital'),
                        initial_data.get('address'),
                        initial_data.get('business_scope'),
                    ],
                )
            )
            url = initial_data.get('title_url')

        # Travel
        if sc == 'travel_dest_overview':
            title = initial_data.get('strong', {}).get('title')
            content = initial_data.get('strong', {}).get('baike_text')
            url = initial_data.get('strong', {}).get('baike_url')

        if sc == 'travel_ranking_list':
            title = initial_data.get('title', {}).get('text')
            content = initial_data.get('title', {}).get('title_tag')
            url = initial_data.get('title', {}).get('url')

        # News
        if sc == 'news_uchq':
            for item in initial_data.get('feed', []):
                results.append(
                    {
                        "title": html_to_text(item.get('title')),
                        "url": item.get('url'),
                        "content": html_to_text(item.get('summary')),
                    }
                )
            # skip dups append for news_uchq
            continue

        # AI Contents
        if sc == 'ai_page':
            for item in initial_data.get('list', []):
                content = (
                    " | ".join(map(str, item.get('content', [])))
                    if isinstance(item.get('content'), list)
                    else str(item.get('content'))
                )
                results.append(
                    {"title": html_to_text(item.get('title')), "url": item.get('url'), "content": html_to_text(content)}
                )
            # skip dups append for ai_page
            continue

        if title and content:
            results.append({"title": html_to_text(title), "url": url, "content": html_to_text(content)})

    return results
