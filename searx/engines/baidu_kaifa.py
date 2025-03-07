# SPDX-License-Identifier: AGPL-3.0-or-later
"""Baidu-Kaifa: A search engine for retrieving coding / development from Baidu."""

from urllib.parse import urlencode

import json

from searx.exceptions import SearxEngineAPIException

about = {
    "website": "https://kaifa.baidu.com/",
    "use_official_api": False,
    "require_api_key": False,
    "results": "JSON",
}

paging = True
results_per_page = 10
categories = ["it"]

base_url = "https://kaifa.baidu.com"


def request(query, params):
    query_params = {
        "wd": query,
        "pageNum": params["pageno"],
        "pageSize": results_per_page,
    }

    params["url"] = f"{base_url}/rest/v1/search?{urlencode(query_params)}"
    return params


def response(resp):
    try:
        data = resp.json()
    except Exception as e:
        raise SearxEngineAPIException(f"Invalid response: {e}") from e

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
