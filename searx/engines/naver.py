# SPDX-License-Identifier: AGPL-3.0-or-later
"""Naver search engine for searxng"""

from urllib.parse import urlencode
from lxml import html
import re

from searx.utils import (
    eval_xpath_getindex,
    eval_xpath_list,
    eval_xpath,
    extract_text,
    html_to_text,
)

# engine metadata
about = {
    "website": "https://search.naver.com",
    "wikidata_id": "Q1046638",
    "use_official_api": False,
    "require_api_key": False,
    "results": "HTML",
    "language": "ko",
}

categories = []
paging = True

time_range_support = True
time_range_dict = {"day": "1d", "week": "1w", "month": "1m", "year": "1y"}

base_url = "https://search.naver.com"

naver_category = "general"
"""Naver supports general, images, news, videos search.

- ``general``: search for general
- ``images``: search for images
- ``news``: search for news
- ``videos``: search for videos
"""

naver_category_dict = {"general": "web", "images": "image", "news": "news", "videos": "video"}


def init(_):
    if naver_category not in ('general', 'images', 'news', 'videos'):
        raise SearxEngineAPIException(f"Unsupported category: {naver_category}")


def request(query, params):
    query_params = {
        "query": query,
        "start": (params["pageno"] - 1) * 15 + 1,
    }

    if naver_category in naver_category_dict:
        query_params["where"] = naver_category_dict[naver_category]

    if params["time_range"] in time_range_dict:
        query_params["nso"] = f"p:{time_range_dict[params['time_range']]}"

    params["url"] = f"{base_url}/search.naver?{urlencode(query_params)}"
    return params


def response(resp):
    data = html.fromstring(resp.text)

    parsers = {'general': parse_general, 'images': parse_images, 'news': parse_news, 'videos': parse_videos}

    return parsers[naver_category](data)


def parse_general(data):
    results = []

    for item in eval_xpath_list(data, "//ul[@class='lst_total']/li[contains(@class, 'bx')]"):
        results.append({
            "title": extract_text(eval_xpath(item, ".//a[@class='link_tit']")),
            "url": eval_xpath_getindex(item, ".//a[@class='link_tit']/@href", 0),
            "content": html_to_text(extract_text(eval_xpath(item, ".//div[@class='total_dsc']"))),
        })

    return results


def parse_images(data):
    results = []

    for item in eval_xpath_list(data, "//ul[@class='lst_total']/li[contains(@class, 'bx')]"):
        results.append({
            "title": extract_text(eval_xpath(item, ".//a[@class='link_tit']")),
            "url": eval_xpath_getindex(item, ".//a[@class='link_tit']/@href", 0),
            "content": html_to_text(extract_text(eval_xpath(item, ".//div[@class='total_dsc']"))),
        })

    return results


def parse_news(data):
    results = []

    for item in eval_xpath_list(data, "//ul[@class='lst_total']/li[contains(@class, 'bx')]"):
        results.append({
            "title": extract_text(eval_xpath(item, ".//a[@class='link_tit']")),
            "url": eval_xpath_getindex(item, ".//a[@class='link_tit']/@href", 0),
            "content": html_to_text(extract_text(eval_xpath(item, ".//div[@class='total_dsc']"))),
        })

    return results


def parse_videos(data):
    results = []

    for item in eval_xpath_list(data, "//ul[@class='lst_total']/li[contains(@class, 'bx')]"):
        results.append({
            "title": extract_text(eval_xpath(item, ".//a[@class='link_tit']")),
            "url": eval_xpath_getindex(item, ".//a[@class='link_tit']/@href", 0),
            "content": html_to_text(extract_text(eval_xpath(item, ".//div[@class='total_dsc']"))),
        })

    return results
