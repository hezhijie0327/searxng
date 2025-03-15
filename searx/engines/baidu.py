# SPDX-License-Identifier: AGPL-3.0-or-later
"""Baidu_

.. _Baidu: https://www.baidu.com
"""

# There exits a https://github.com/ohblue/baidu-serp-api/
# but we don't use it here (may we can learn from).

from urllib.parse import urlencode
from datetime import datetime
import time
import json
import re

from searx.exceptions import SearxEngineAPIException
from searx.utils import html_to_text

about = {
    "website": "https://www.baidu.com",
    "wikidata_id": "Q14772",
    "official_api_documentation": None,
    "use_official_api": False,
    "require_api_key": False,
    "results": "JSON",
    "language": "zh",
}

paging = True
categories = []
results_per_page = 10

baidu_category = 'general'

time_range_support = True
time_range_dict = {"day": 86400, "week": 604800, "month": 2592000, "year": 31536000}


def init(_):
    if baidu_category not in ('general', 'images', 'it'):
        raise SearxEngineAPIException(f"Unsupported category: {baidu_category}")


def request(query, params):
    page_num = params["pageno"]

    category_config = {
        'general': {
            'endpoint': 'https://www.baidu.com/s',
            'params': {
                "wd": query,
                "rn": results_per_page,
                "pn": (page_num - 1) * results_per_page,
                "tn": "json",
            },
        },
        'images': {
            'endpoint': 'https://image.baidu.com/search/acjson',
            'params': {
                "word": query,
                "rn": results_per_page,
                "pn": (page_num - 1) * results_per_page,
                "tn": "resultjson_com",
            },
        },
        'it': {
            'endpoint': 'https://kaifa.baidu.com/rest/v1/search',
            'params': {
                "wd": query,
                "pageSize": results_per_page,
                "pageNum": page_num,
                "paramList": f"page_num={page_num},page_size={results_per_page}",
                "position": 0,
            },
        },
    }

    query_params = category_config[baidu_category]['params']
    query_url = category_config[baidu_category]['endpoint']

    if params.get("time_range") in time_range_dict:
        now = int(time.time())
        past = now - time_range_dict[params["time_range"]]

        if baidu_category == 'general':
            query_params["gpc"] = f"stf={past},{now}|stftype=1"

        if baidu_category == 'it':
            query_params["paramList"] += f",timestamp_range={past}-{now}"

    params["url"] = f"{query_url}?{urlencode(query_params)}"
    return params


def response(resp):
    try:
        text = resp.text

        if baidu_category == 'images':
            # fix Invalid \escape
            text = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', text)

        data = json.loads(text, strict=False)
    except Exception as e:
        raise SearxEngineAPIException(f"Invalid response: {e}") from e

    parsers = {'general': parse_general, 'images': parse_images, 'it': parse_it}

    return parsers[baidu_category](data)


def parse_general(data):
    results = []
    if not data.get("feed", {}).get("entry"):
        raise SearxEngineAPIException("Invalid response")

    for entry in data["feed"]["entry"]:
        if not entry.get("title") or not entry.get("url"):
            continue

        published_date = None
        if entry.get("time"):
            try:
                published_date = datetime.fromtimestamp(entry["time"])
            except (ValueError, TypeError):
                published_date = None

        results.append(
            {
                "title": entry["title"],
                "url": entry["url"],
                "content": entry.get("abs", ""),
                "publishedDate": published_date,
            }
        )
    return results


def parse_images(data):
    results = []
    if "data" in data:
        for item in data["data"]:
            replace_url = item.get("replaceUrl", [{}])[0]
            from_url = replace_url.get("FromURL", "").replace("\\/", "/")
            img_src = replace_url.get("ObjURL", "").replace("\\/", "/")

            results.append(
                {
                    "template": "images.html",
                    "url": from_url,
                    "thumbnail_src": item.get("thumbURL", ""),
                    "img_src": img_src,
                    "content": html_to_text(item.get("fromPageTitleEnc", "")),
                    "title": html_to_text(item.get("fromPageTitle", "")),
                    "source": item.get("fromURLHost", ""),
                }
            )
    return results


def parse_it(data):
    results = []
    if not data.get("data", {}).get("documents", {}).get("data"):
        raise SearxEngineAPIException("Invalid response")

    for entry in data["data"]["documents"]["data"]:
        results.append(
            {
                'title': entry["techDocDigest"]["title"],
                'url': entry["techDocDigest"]["url"],
                'content': entry["techDocDigest"]["summary"],
            }
        )
    return results
