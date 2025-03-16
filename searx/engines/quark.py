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

        source_category = extra_data.get('sc')
        source_category_parsers = {
            addition: parse_addition,
            ai_page: parse_ai_page,
            baike_sc: parse_baike,
            finance_shuidi: parse_finance_shuidi,
            nature_result: parse_nature_result,
            news_uchq: parse_news_uchq,
            ss_note: parse_ss_note,
            ss_pic: parse_ss_pic_text,
            ss_text: parse_ss_pic_text,
            travel_dest_overview: parse_travel_dest_overview,
            travel_ranking_list: parse_travel_ranking_list,
        }

        parsers = source_category_parsers.get(source_category)
        if parsers:
            parsed_results = parsers(initial_data)
            if isinstance(parsed_results, list):
                # Extend if the result is a list
                results.extend(parsed_results)
            else:
                # Append if it's a single result
                results.append(parsed_results)

    return results


def parse_addition(data):
    return {
        "title": html_to_text(data.get('title', {}).get('content')),
        "url": data.get('source', {}).get('url'),
        "content": html_to_text(data.get('summary', {}).get('content')),
    }


def parse_ai_page(data):
    results = []
    for item in data.get('list', []):
        content = (
            " | ".join(map(str, item.get('content', [])))
            if isinstance(item.get('content'), list)
            else str(item.get('content'))
        )
        results.append(
            {"title": html_to_text(item.get('title')), "url": item.get('url'), "content": html_to_text(content)}
        )
    return results


def parse_baike(data):
    return {
        "title": html_to_text(data.get('data', {}).get('title')),
        "url": data.get('data', {}).get('url'),
        "content": html_to_text(data.get('data', {}).get('abstract')),
    }


def parse_finance_shuidi(data):
    content = " | ".join(
        filter(
            None,
            [
                data.get('establish_time'),
                data.get('company_status'),
                data.get('controled_type'),
                data.get('company_type'),
                data.get('capital'),
                data.get('address'),
                data.get('business_scope'),
            ],
        )
    )
    return {
        "title": html_to_text(data.get('company_name')),
        "url": data.get('title_url'),
        "content": html_to_text(content),
    }


def parse_nature_result(data):
    return {"title": html_to_text(data.get('title')), "url": data.get('url'), "content": html_to_text(data.get('desc'))}


def parse_news_uchq(data):
    results = []
    for item in data.get('feed', []):
        results.append(
            {
                "title": html_to_text(item.get('title')),
                "url": item.get('url'),
                "content": html_to_text(item.get('summary')),
            }
        )
    return results


def parse_ss_note(data):
    return {
        "title": html_to_text(data.get('title', {}).get('content')),
        "url": data.get('source', {}).get('dest_url'),
        "content": html_to_text(data.get('summary', {}).get('content')),
    }


def parse_ss_pic_text(data):
    return {
        "title": html_to_text(data.get('titleProps', {}).get('content')),
        "url": data.get('sourceProps', {}).get('dest_url'),
        "content": html_to_text(data.get('summaryProps', {}).get('content')),
    }


def parse_travel_dest_overview(data):
    return {
        "title": html_to_text(data.get('strong', {}).get('title')),
        "url": data.get('strong', {}).get('baike_url'),
        "content": html_to_text(data.get('strong', {}).get('baike_text')),
    }


def parse_travel_ranking_list(data):
    return {
        "title": html_to_text(data.get('title', {}).get('text')),
        "url": data.get('title', {}).get('url'),
        "content": html_to_text(data.get('title', {}).get('title_tag')),
    }
