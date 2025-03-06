# SPDX-License-Identifier: AGPL-3.0-or-later
"""Baidu-Images: A search engine for retrieving images from Baidu."""

from urllib.parse import urlencode
from datetime import datetime

import json

from searx.exceptions import SearxEngineAPIException
from searx.utils import html_to_text

about = {
    "website": "https://image.baidu.com/",
    "wikidata_id": "Q109342769",
    "use_official_api": False,
    "require_api_key": False,
    "results": "JSON",
}

paging = True
results_per_page = 10
categories = ["images"]

base_url = "https://image.baidu.com"


def request(query, params):
    query_params = {"tn": "resultjson_com", "word": query, "pn": params["pageno"], "rn": 30}

    params["url"] = f"{base_url}/search/acjson?{urlencode(query_params)}"
    return params


def response(resp):
    try:
        clean_text = resp.text.replace("\\", "\\\\")
        data = json.loads(clean_text, strict=False)
    except Exception as e:
        raise SearxEngineAPIException(f"Invalid response: {e}") from e
    results = []

    if "data" in data:
        for item in data["data"]:
            replace_url = item.get("replaceUrl", [])
            from_url = replace_url[0].get("FromURL", "") if replace_url else ""
            img_src = replace_url[0].get("ObjURL", "") if replace_url else ""

            results.append(
                {
                    "template": "images.html",
                    "url": from_url.replace("\\/", "/"),
                    "thumbnail_src": item.get("thumbURL", ""),
                    "img_src": img_src.replace("\\/", "/"),
                    "content": item.get("fromPageTitleEnc", ""),
                    "title": item.get("fromPageTitle", ""),
                    "source": item.get("fromURLHost", ""),
                }
            )

    return results
