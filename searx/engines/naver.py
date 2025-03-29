# SPDX-License-Identifier: AGPL-3.0-or-later
"""Naver search engine for searxng"""

from urllib.parse import urlencode
import re
from lxml import html

from searx.exceptions import SearxEngineAPIException
from searx.utils import (
    eval_xpath_getindex,
    eval_xpath_list,
    eval_xpath,
    extract_text,
    html_to_text,
    parse_duration_string,
    js_variable_to_python,
)

# engine metadata
about = {
    "website": "https://search.naver.com",
    "wikidata_id": "Q485639",
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
    page_num = params["pageno"]

    query_params = {
        "query": query,
    }

    if naver_category == 'news':
        query_params["start"] = (page_num - 1) * 10 + 1
    elif naver_category == 'videos':
        query_params["start"] = (page_num - 1) * 48 + 1
    elif naver_category == 'images':
        query_params["start"] = (page_num - 1) * 50 + 1
    else:
        # general
        query_params["start"] = (page_num - 1) * 15 + 1

    if naver_category in naver_category_dict:
        query_params["where"] = naver_category_dict[naver_category]

    if params["time_range"] in time_range_dict:
        query_params["nso"] = f"p:{time_range_dict[params['time_range']]}"

    params["url"] = f"{base_url}/search.naver?{urlencode(query_params)}"
    return params


def response(resp):
    parsers = {'general': parse_general, 'images': parse_images, 'news': parse_news, 'videos': parse_videos}

    return parsers[naver_category](resp.text)


def parse_general(data):
    results = []

    dom = html.fromstring(data)

    for item in eval_xpath_list(dom, "//ul[@class='lst_total']/li[contains(@class, 'bx')]"):
        results.append(
            {
                "title": extract_text(eval_xpath(item, ".//a[@class='link_tit']")),
                "url": eval_xpath_getindex(item, ".//a[@class='link_tit']/@href", 0),
                "content": extract_text(
                    eval_xpath(item, ".//div[contains(@class, 'total_dsc_wrap')]//a[contains(@class, 'api_txt_lines')]")
                ),
            }
        )

    return results


def parse_images(data):
    results = []

    match = re.search(r'var imageSearchTabData\s*=\s*({.*?})\s*</script>', data, re.DOTALL)
    if match:
        json = js_variable_to_python(match.group(1))
        items = json.get('content', {}).get('items', [])

        for item in items:
            results.append(
                {
                    "template": "images.html",
                    "url": item.get('link'),
                    "thumbnail_src": item.get('thumb'),
                    "img_src": item.get('originalUrl'),
                    "title": html_to_text(item.get('title')),
                    "source": item.get('source'),
                    "resolution": f"{item.get('orgWidth')} x {item.get('orgHeight')}",
                }
            )

    return results


def parse_news(data):
    results = []

    dom = html.fromstring(data)

    for item in eval_xpath_list(dom, "//ul[contains(@class, 'list_news')]/li[contains(@class, 'bx')]"):
        thumbnail = None
        try:
            thumbnail = eval_xpath_getindex(item, ".//a[contains(@class, 'dsc_thumb')]/img/@data-lazysrc", 0)
        except (ValueError, TypeError):
            pass

        results.append(
            {
                "title": extract_text(eval_xpath(item, ".//a[contains(@class, 'news_tit')]")),
                "url": eval_xpath_getindex(item, ".//a[contains(@class, 'news_tit')]/@href", 0),
                "content": html_to_text(
                    extract_text(
                        eval_xpath(item, ".//div[contains(@class, 'news_dsc')]//a[contains(@class, 'api_txt_lines')]")
                    )
                ),
                "thumbnail": thumbnail,
            }
        )

    return results


def parse_videos(data):
    results = []

    dom = html.fromstring(data)

    for item in eval_xpath_list(dom, "//li[contains(@class, 'video_item _svp_item')]"):
        thumbnail = None
        try:
            thumbnail = eval_xpath_getindex(item, ".//img[@class='thumb api_get api_img']/@src", 0)
        except (ValueError, TypeError):
            pass

        length = None
        try:
            length = parse_duration_string(extract_text(eval_xpath(item, ".//span[@class='time']")))
        except (ValueError, TypeError):
            pass

        results.append(
            {
                "template": "videos.html",
                "title": extract_text(eval_xpath(item, ".//a[@class='info_title']")),
                "url": eval_xpath_getindex(item, ".//a[@class='info_title']/@href", 0),
                "thumbnail": thumbnail,
                'length': length,
            }
        )

    return results
