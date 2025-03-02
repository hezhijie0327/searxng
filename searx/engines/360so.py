# SPDX-License-Identifier: AGPL-3.0-or-later
# pylint: disable=invalid-name
"""360So search engine for searxng"""

from urllib.parse import urlencode
from lxml import html

from searx.utils import extract_text

# Metadata
about = {
    "website": "https://www.so.com/",
    "official_api_documentation": None,
    "use_official_api": False,
    "require_api_key": False,
    "results": "HTML",
}

# Engine Configuration
categories = ["general"]
paging = True
max_page = 10
time_range_support = True

time_range_dict = {'day': 'd', 'week': 'w', 'month': 'm', 'year': 'y'}

# Base URL
base_url = "https://www.so.com"


def request(query, params):
    query_params = {
        "pn": params["pageno"],
        "q": query,
    }

    if time_range_dict.get(params['time_range']):
        query_params["adv_t"] = time_range_dict.get(params['time_range'])

    params["url"] = f"{base_url}/s?{urlencode(query_params)}"
    return params


def response(resp):
    dom = html.fromstring(resp.text)
    results = []

    for item in dom.xpath('//li[contains(@class, "res-list")]'):
        title = extract_text(item.xpath('.//h3[contains(@class, "res-title")]/a'))

        url = extract_text(item.xpath('.//h3[contains(@class, "res-title")]/a/@data-mdurl'))
        if not url:
            url = extract_text(item.xpath('.//h3[contains(@class, "res-title")]/a/@href'))

        content = extract_text(item.xpath('.//p[@class="res-desc"]'))
        if not content:
            content = extract_text(item.xpath('.//span[@class="res-list-summary"]'))

        if title and url:
            results.append(
                {
                    "title": title,
                    "url": url,
                    "content": content,
                }
            )

    return results
