# SPDX-License-Identifier: AGPL-3.0-or-later
# pylint: disable=invalid-name

import json
import re
from urllib.parse import urlencode
from lxml import html

about = {
    "website": "https://v.sogou.com/",
    "official_api_documentation": None,
    "use_official_api": False,
    "require_api_key": False,
    "results": "HTML",
}

categories = ["videos", "web"]

# Base URL
base_url = "https://v.sogou.com/v"


def request(query, params):
    query_params = {
        "query": query
    }

    params["url"] = f"{base_url}?{urlencode(query_params)}"
    return params

def response(resp):
    results = []
    match = re.search(r'window\.__INITIAL_STATE__\s*=\s*({.*?});', resp.text, re.S)
    if not match:
        return results

    data = json.loads(match.group(1))
    if "result" in data and "shortVideo" in data["result"]:
        for item in data["result"]["shortVideo"].get("list", []):
            video_url = item.get("url", "")
            if video_url.startswith("/vc/np"):
                video_url = f"https://v.sogou.com{video_url}"

            results.append(
                {
                    "url": video_url,
                    "thumbnail": item.get("picurl", ""),
                    "title": item.get("titleEsc", ""),
                    "content": f"{item.get('site', '')} | {item.get('duration', '')} | {item.get('dateTime', '')}",
                    "template": "videos.html",
                }
            )'

    return results
