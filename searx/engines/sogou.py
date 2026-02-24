# SPDX-License-Identifier: AGPL-3.0-or-later
# pylint: disable=too-many-branches, line-too-long
"""Sogou search engine for searxng"""

import re
from datetime import datetime
from urllib.parse import urlencode
from lxml import html

from searx.utils import extract_text

# Metadata
about = {
    "website": "https://www.sogou.com/",
    "wikidata_id": "Q7554565",
    "use_official_api": False,
    "require_api_key": False,
    "results": "HTML",
    "language": "zh",
}

# Engine Configuration
categories = ["general"]
paging = True
time_range_support = True

time_range_dict = {
    "day": "inttime_day",
    "week": "inttime_week",
    "month": "inttime_month",
    "year": "inttime_year",
}

# Base URL
base_url = "https://www.sogou.com"


def request(query, params):
    query_params = {
        "query": query,
        "page": params["pageno"],
    }

    if time_range_dict.get(params["time_range"]):
        query_params["s_from"] = time_range_dict.get(params["time_range"])
        query_params["tsn"] = 1

    params["url"] = f"{base_url}/web?{urlencode(query_params)}"
    return params


def response(resp):
    dom = html.fromstring(resp.text)
    results = []

    for item in dom.xpath(
        '//div[contains(@class, "rb")] | //div[contains(@class, "vrwrap") and not(.//div[contains(@class, "special-wrap")])]'
    ):
        title = None
        url = None
        content = None
        publishedDate = None

        item_html = html.tostring(item, encoding="unicode")

        if item.xpath('.//h3[@class="pt"]/a'):
            title = extract_text(item.xpath('.//h3[@class="pt"]/a'))
            content = extract_text(item.xpath('.//div[@class="ft"]'))
            match = re.search(r'data-url="([^"]+)"', item_html)
            if match:
                url = match.group(1)
            if not url:
                url = extract_text(item.xpath('.//h3[@class="pt"]/a/@href'))
                if url and url.startswith("/link?url="):
                    url = f"{base_url}{url}"
            cite_text = extract_text(item.xpath(".//cite"))
            if cite_text:
                date_match = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", cite_text)
                if date_match:
                    try:
                        publishedDate = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                    except (ValueError, TypeError):
                        publishedDate = None
        elif item.xpath('.//h3[contains(@class, "vr-title")]/a'):
            title = extract_text(item.xpath('.//h3[contains(@class, "vr-title")]/a'))
            content = extract_text(item.xpath('.//div[contains(@class, "attribute-centent")]'))
            if not content:
                content = extract_text(item.xpath('.//div[contains(@class, "fz-mid space-txt")]'))
            match = re.search(r'data-url="([^"]+)"', item_html)
            if match:
                url = match.group(1)
            if not url:
                url = extract_text(item.xpath('.//h3[contains(@class, "vr-title")]/a/@href'))
                if url and url.startswith("/link?url="):
                    url = f"{base_url}{url}"
            cite_date = extract_text(item.xpath('.//span[@class="cite-date"]'))
            if cite_date:
                date_str = cite_date.strip().lstrip("-").strip()
                try:
                    publishedDate = datetime.strptime(date_str, "%Y-%m-%d")
                except (ValueError, TypeError):
                    publishedDate = None

        if title and url:
            results.append(
                {
                    "title": title,
                    "url": url,
                    "content": content,
                    "publishedDate": publishedDate,
                }
            )

    return results
