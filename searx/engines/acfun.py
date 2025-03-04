# SPDX-License-Identifier: AGPL-3.0-or-later
"""Acfun search engine for searxng"""

from urllib.parse import urlencode
import re
import json
import codecs
import html
from lxml import html as lxml_html

# Metadata
about = {
    "website": "https://www.acfun.cn/",
    "wikidata_id": None,
    "use_official_api": False,
    "require_api_key": False,
    "results": "HTML",
}

# Engine Configuration
categories = ["videos"]
paging = True

# Base URL
base_url = "https://www.acfun.cn/search"


def request(query, params):
    query_params = {
        "keyword": query,
        "page": params["pageno"],
    }

    params["url"] = f"{base_url}?{urlencode(query_params)}"
    return params


def response(resp):
    # Force decode response content to UTF-8
    try:
        resp_text = resp.content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            resp_text = resp.content.decode('gbk')  # Try GBK encoding if UTF-8 fails
        except UnicodeDecodeError:
            resp_text = resp.text  # Fallback to default decoding

    # Debugging: Print raw response text
    print("Raw Response Text:")
    print(resp_text[:500])

    dom = lxml_html.fromstring(resp_text)
    results = []

    # Extract JSON data embedded in the HTML (bigPipe.onPageletArrive)
    matches = re.findall(r'bigPipe\.onPageletArrive\((\{.*?\})\);', resp_text, re.DOTALL)
    if not matches:
        return results

    for match in matches:
        try:
            json_data = json.loads(match)
            raw_html = json_data.get("html", "")
            if not raw_html:
                continue

            # Decode Unicode escape sequences in raw_html
            raw_html = codecs.decode(raw_html.encode('utf-8'), 'unicode_escape')

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
                url = f"https://www.acfun.cn/v/ac{content_id}" if content_id else ""

                # Extract additional details using regex within the current video block
                cover_match = re.search(r'<img src="(.*?)" alt=', video_block)
                cover_image = cover_match.group(1) if cover_match else ""

                duration_match = re.search(r'<span class="video__duration">(.*?)</span>', video_block)
                duration = duration_match.group(1).strip() if duration_match else ""

                publish_time_match = re.search(r'<span class="info__create-time">(.*?)</span>', video_block)
                publish_time = publish_time_match.group(1).strip() if publish_time_match else ""

                description_match = re.search(r'<div class="video__main__intro ellipsis2">(.*?)</div>', video_block, re.DOTALL)
                description = description_match.group(1).strip() if description_match else ""

                # Decode HTML entities in extracted text
                title = html.unescape(title)
                description = html.unescape(description)

                if title and url:
                    results.append(
                        {
                            "title": title,
                            "url": url,
                            "content": description,
                            "thumbnail": cover_image,
                            "duration": duration,
                            "published_date": publish_time,
                        }
                    )
        except json.JSONDecodeError:
            continue

    return results
