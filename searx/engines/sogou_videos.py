# SPDX-License-Identifier: AGPL-3.0-or-later
# pylint: disable=invalid-name

import json
import re
import requests
from urllib.parse import urlencode
from lxml import html

about = {
    "website": "https://v.sogou.com/",
    "wikidata_id": "Q86699677",
    "official_api_documentation": None,
    "use_official_api": False,
    "require_api_key": False,
    "results": "HTML",
}

categories = ["videos", "web"]
paging = True
safesearch = True

def request(query, params):
    """Assemble a Sogou Video search request."""
    query_params = {"ie": "utf8", "query": query}
    params["url"] = "https://v.sogou.com/v?" + urlencode(query_params)
    return params

def response(resp):
    """Parse response from Sogou Video search."""
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
            )
    return results
