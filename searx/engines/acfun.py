# SPDX-License-Identifier: AGPL-3.0-or-later
"""Acfun search engine for searxng"""

from urllib.parse import urlencode
import re
import json
from datetime import datetime, timedelta

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
    query_params = {"keyword": query, "pCursor": params["pageno"]}
    params["url"] = f"{base_url}/search?{urlencode(query_params)}"
    return params


def response(resp):
    results = []

    matches = re.findall(r'bigPipe\.onPageletArrive\((\{.*?\})\);', resp.text, re.DOTALL)
    if not matches:
        return results

    for match in matches:
        try:
            json_data = json.loads(match)
            raw_html = json_data.get("html", "")
            if not raw_html:
                continue

            video_blocks = re.findall(r'<div class="search-video".*?</div>\s*</div>', raw_html, re.DOTALL)

            for video_block in video_blocks:
                video_info = extract_video_data(video_block)
                if video_info and video_info["title"] and video_info["url"]:
                    results.append(video_info)
        except json.JSONDecodeError:
            continue

    return results


def extract_video_data(video_block):
    """Extract video data from a single video block."""
    try:
        # Extract title and content ID from data-exposure-log
        exposure_log_match = re.search(r'data-exposure-log=\'({.*?})\'', video_block)

        video_data = json.loads(exposure_log_match.group(1))
        title = video_data.get("title", "")
        content_id = video_data.get("content_id", "")

        # Extract description, cover image, publish time, and duration
        description = re.search(r'<div class="video__main__intro ellipsis2">(.*?)</div>', video_block, re.DOTALL)
        cover_image = re.search(r'<img src="(.*?)" alt=', video_block)
        publish_time = re.search(r'<span class="info__create-time">(.*?)</span>', video_block)
        duration = re.search(r'<span class="video__duration">(.*?)</span>', video_block)

        # Parse url and iframe_url
        url = f"{base_url}/v/ac{content_id}"
        iframe_url = f"{base_url}/player/ac{content_id}"

        # Parse publish_time and duration
        published_date = None
        if publish_time:
            try:
                published_date = datetime.strptime(publish_time.group(1).strip(), "%Y-%m-%d")
            except (ValueError, TypeError):
                pass

        length = None
        if duration:
            try:
                timediff = datetime.strptime(duration.group(1).strip(), "%M:%S")
                length = timedelta(minutes=timediff.minute, seconds=timediff.second)
            except (ValueError, TypeError):
                pass

        return {
            "title": title,
            "url": url,
            "content": description.group(1).strip(),
            "thumbnail": cover_image.group(1),
            "length": length,
            "publishedDate": published_date,
            "iframe_src": iframe_url,
        }
    except json.JSONDecodeError:
        return None
