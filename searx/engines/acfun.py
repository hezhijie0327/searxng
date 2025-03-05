# SPDX-License-Identifier: AGPL-3.0-or-later
# pylint: disable=invalid-name
"""Acfun search engine for searxng"""

from urllib.parse import urlencode
import re
import json
from datetime import datetime, timedelta

from searx.utils import extract_text

# Metadata
about = {
    "website": "https://www.acfun.cn/",
    "wikidata_id": "Q3077675",
    "use_official_api": False,
    "require_api_key": False,
    "results": "HTML",
}

# Engine Configuration
categories = ["videos"]
paging = True

# Base URL
base_url = "https://www.acfun.cn"


def request(query, params):
    query_params = {
        "keyword": query,
        "pCursor": params["pageno"],
    }

    params["url"] = f"{base_url}/search?{urlencode(query_params)}"
    return params


def response(resp):
    results = []

    # Extract JSON data embedded in the HTML (bigPipe.onPageletArrive)
    matches = re.findall(r'bigPipe\.onPageletArrive\((\{.*?\})\);', resp.text, re.DOTALL)
    if not matches:
        return results

    for match in matches:
        try:
            json_data = json.loads(match)
            raw_html = json_data.get("html", "")
            if not raw_html:
                continue

            # Split into individual video blocks
            video_blocks = re.findall(r'<div class="search-video".*?</div>\s*</div>', raw_html, re.DOTALL)

            for video_block in video_blocks:
                # Extract title and URL from data-exposure-log attribute
                exposure_log_match = re.search(r'data-exposure-log=\'({.*?})\'', video_block)
                if not exposure_log_match:
                    continue

                video_data = json.loads(exposure_log_match.group(1))
                title = video_data.get("title", "")
                content_id = video_data.get("content_id", "")
                url = f"{base_url}/v/ac{content_id}" if content_id else ""

                description_match = re.search(r'<div class="video__main__intro ellipsis2">(.*?)</div>', video_block, re.DOTALL)
                description = description_match.group(1).strip() if description_match else ""

                # Extract additional details using regex within the current video block
                cover_match = re.search(r'<img src="(.*?)" alt=', video_block)
                cover_image = cover_match.group(1) if cover_match else ""

                publish_time_match = re.search(r'<span class="info__create-time">(.*?)</span>', video_block)
                publish_time = publish_time_match.group(1).strip() if publish_time_match else ""

                duration_match = re.search(r'<span class="video__duration">(.*?)</span>', video_block)
                duration = duration_match.group(1).strip() if duration_match else ""

                iframe_url = f"{base_url}/player/ac{content_id}"

                published_date = None
                if publish_time:
                    try:
                        published_date = datetime.strptime(publish_time, "%Y-%m-%d")
                    except (ValueError, TypeError):
                        pass

                length = None
                if duration:
                    try:
                        timediff = datetime.strptime(duration, "%M:%S")
                        length = timedelta(minutes=timediff.minute, seconds=timediff.second)
                    except (ValueError, TypeError):
                        pass

                if title and url:
                    results.append(
                        {
                            "title": title,
                            "url": url,
                            "content": description,
                            "thumbnail": cover_image,
                            "length": length,
                            "publishedDate": published_date,
                            "iframe_src": iframe_url,
                        }
                    )
        except json.JSONDecodeError:
            continue

    return results
