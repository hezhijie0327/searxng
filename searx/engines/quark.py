# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quark (Shenma) search engine for searxng"""

from urllib.parse import urlencode
from datetime import datetime
import re
import json

from searx.utils import html_to_text, gen_useragent
from searx.exceptions import SearxEngineAPIException, SearxEngineCaptchaException

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
categories = []
paging = True
results_per_page = 10

quark_category = 'general'

time_range_support = True
time_range_dict = {'day': '4', 'week': '3', 'month': '2', 'year': '1'}

CAPTCHA_PATTERN = r'\{[^{]*?"action"\s*:\s*"captcha"\s*,\s*"url"\s*:\s*"([^"]+)"[^{]*?\}'


def is_alibaba_captcha(html):
    """
    Detects if the response contains an Alibaba X5SEC CAPTCHA page.

    Quark may return a CAPTCHA challenge after 9 requests in a short period.

    Typically, the ban duration is around 15 minutes.
    """
    return bool(re.search(CAPTCHA_PATTERN, html))


def init(_):
    if quark_category not in ('general', 'images'):
        raise SearxEngineAPIException(f"Unsupported category: {quark_category}")


def request(query, params):
    page_num = params["pageno"]

    category_config = {
        'general': {
            'endpoint': 'https://m.quark.cn/s',
            'params': {
                "q": query,
                "layout": "html",
                "page": page_num,
            },
        },
        'images': {
            'endpoint': 'https://vt.sm.cn/api/pic/list',
            'params': {
                "query": query,
                "limit": results_per_page,
                "start": (page_num - 1) * results_per_page,
            },
        },
    }

    query_params = category_config[quark_category]['params']
    query_url = category_config[quark_category]['endpoint']

    if time_range_dict.get(params['time_range']) and quark_category == 'general':
        query_params["tl_request"] = time_range_dict.get(params['time_range'])

    params["url"] = f"{query_url}?{urlencode(query_params)}"
    params["headers"] = {
        "User-Agent": gen_useragent(),
    }
    return params


def response(resp):
    results = []
    text = resp.text

    if is_alibaba_captcha(text):
        raise SearxEngineCaptchaException(suspended_time=900, message="Alibaba CAPTCHA detected. Please try again later.")

    if quark_category == 'images':
        data = json.loads(text)
        for item in data.get('data', {}).get('hit', {}).get('imgInfo', {}).get('item', []):
            results.append(
                {
                    "template": "images.html",
                    "url": item.get("imgUrl"),
                    "thumbnail_src": item.get("img"),
                    "img_src": item.get("bigPicUrl"),
                    "title": item.get("title"),
                    "source": item.get("site"),
                    "resolution": f"{item['width']} x {item['height']}",
                    "publishedDate": datetime.fromtimestamp(int(item.get("publish_time"))),
                }
            )

    if quark_category == 'general':
        pattern = r'<script\s+type="application/json"\s+id="s-data-[^"]+"\s+data-used-by="hydrate">(.*?)</script>'
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            data = json.loads(match)
            initial_data = data.get('data', {}).get('initialData', {})
            extra_data = data.get('extraData', {})

            source_category = extra_data.get('sc')
            source_category_parsers = {
                'addition': parse_addition,
                'ai_page': parse_ai_page,
                'baike_sc': parse_baike_sc,
                'finance_shuidi': parse_finance_shuidi,
                'life_show_general_image': parse_life_show_general_image,
                'nature_result': parse_nature_result,
                'news_uchq': parse_news_uchq,
                'ss_note': parse_ss_note,
                'ss_pic': parse_ss_pic_text,
                'ss_text': parse_ss_pic_text,
                'travel_dest_overview': parse_travel_dest_overview,
                'travel_ranking_list': parse_travel_ranking_list,
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
            {
                "title": html_to_text(item.get('title')),
                "url": item.get('url'),
                "content": html_to_text(content),
                "publishedDate": datetime.fromtimestamp(int(item.get('source', {}).get('time'))),
            }
        )
    return results


def parse_baike_sc(data):
    return {
        "title": html_to_text(data.get('data', {}).get('title')),
        "url": data.get('data', {}).get('url'),
        "content": html_to_text(data.get('data', {}).get('abstract')),
    }


def parse_finance_shuidi(data):
    content = " | ".join(
      info for info in [
            data.get('establish_time'),
            data.get('company_status'),
            data.get('controled_type'),
            data.get('company_type'),
            data.get('capital'),
            data.get('address'),
            data.get('business_scope'),
        ] if info,
    )
    )
    return {
        "title": html_to_text(data.get('company_name')),
        "url": data.get('title_url'),
        "content": html_to_text(content),
    }


def parse_life_show_general_image(data):
    results = []
    for item in data.get('image', []):
        results.append(
            {
                "template": "images.html",
                "url": item.get("imgUrl"),
                "thumbnail_src": item.get("img"),
                "img_src": item.get("bigPicUrl"),
                "title": item.get("title"),
                "source": item.get("site"),
                "resolution": f"{item['width']} x {item['height']}",
                "publishedDate": datetime.fromtimestamp(int(item.get("publish_time"))),
            }
        )
    return results


def parse_nature_result(data):
    return {"title": html_to_text(data.get('title')), "url": data.get('url'), "content": html_to_text(data.get('desc'))}


def parse_news_uchq(data):
    results = []
    for item in data.get('feed', []):
        try:
            published_date = datetime.strptime(item.get('time'), "%Y-%m-%d")
        except (ValueError, TypeError):
            # Sometime Quark will return non-standard format like "1天前", set published_date as None
            published_date = None

        results.append(
            {
                "title": html_to_text(item.get('title')),
                "url": item.get('url'),
                "content": html_to_text(item.get('summary')),
                "thumbnail": item.get('image'),
                "publishedDate": published_date,
            }
        )
    return results


def parse_ss_note(data):
    return {
        "title": html_to_text(data.get('title', {}).get('content')),
        "url": data.get('source', {}).get('dest_url'),
        "content": html_to_text(data.get('summary', {}).get('content')),
        "publishedDate": datetime.fromtimestamp(int(data.get('source', {}).get('time'))),
    }


def parse_ss_pic_text(data):
    time_value = data.get('sourceProps', {}).get('time')
    if time_value is None or int(time_value) == 0:
        # Sometime Quark will return 0, set published_date as None
        published_date = None
    else:
        published_date = datetime.fromtimestamp(int(time_value))

    return {
        "title": html_to_text(data.get('titleProps', {}).get('content')),
        "url": data.get('sourceProps', {}).get('dest_url'),
        "content": html_to_text(data.get('summaryProps', {}).get('content')),
        "publishedDate": published_date,
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
