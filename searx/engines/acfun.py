# SPDX-License-Identifier: AGPL-3.0-or-later
"""Acfun search engine for searxng"""

from urllib.parse import urlencode
import json
import re
from datetime import datetime, timedelta

from lxml import html

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
    """Prepare the search request."""
    query_params = {"keyword": query, "pCursor": params["pageno"]}
    params["url"] = f"{base_url}/search?{urlencode(query_params)}"
    return params


def response(resp):
    """Parse the search results."""
    results = []

    # Extract JSON data embedded in JavaScript
    matches = re.findall(r'bigPipe\.onPageletArrive\((\{.*?\})\);', resp.text, re.DOTALL)
    if not matches:
        return results

    for match in matches:
        try:
            json_data = json.loads(match)
            raw_html = json_data.get("html", "")
            if not raw_html:
                continue

            # Parse HTML content
            tree = html.fromstring(raw_html)

            # Find all video blocks
            video_blocks = tree.xpath('//div[contains(@class, "search-video")]')

            # Extract video data from each block
            for video_block in video_blocks:
                video_info = extract_video_data(video_block)
                if video_info and video_info["title"] and video_info["url"]:
                    results.append(video_info)

        except (json.JSONDecodeError, Exception) as e:
            # Log errors but continue processing
            print(f"Error parsing JSON or extracting video data: {e}")
            continue

    return results


def extract_video_data(video_block):
    """Extract video data from a single video block."""
    try:
        # Extract data-exposure-log attribute
        data_exposure_log = video_block.get('data-exposure-log')
        if not data_exposure_log:
            return None

        # Parse JSON data from data-exposure-log
        video_data = json.loads(data_exposure_log)
        title = video_data.get("title", "")
        content_id = video_data.get("content_id", "")

        # Extract fields using XPath and extract_text
        description = extract_text(video_block.xpath('.//div[@class="video__main__intro"]'), allow_none=True)
        cover_image = video_block.xpath('.//div[@class="video__cover"]/a/img/@src')[0] if video_block.xpath('.//div[@class="video__cover"]/a/img/@src') else None
        publish_time = extract_text(video_block.xpath('.//span[@class="info__create-time"]'))
        duration = extract_text(video_block.xpath('.//span[@class="video__duration"]'))

        # Parse URL and iframe URL
        url = f"{base_url}/v/ac{content_id}"
        iframe_url = f"{base_url}/player/ac{content_id}"

        # Parse publish_time and duration
        published_date = None
        if publish_time:
            try:
                published_date = datetime.strptime(publish_time.strip(), "%Y-%m-%d")
            except (ValueError, TypeError):
                pass

        length = None
        if duration:
            try:
                timediff = datetime.strptime(duration.strip(), "%M:%S")
                length = timedelta(minutes=timediff.minute, seconds=timediff.second)
            except (ValueError, TypeError):
                pass

        # Return structured video data
        return {
            "title": title,
            "url": url,
            "content": description,
            "thumbnail": cover_image,
            "length": length,
            "publishedDate": published_date,
            "iframe_src": iframe_url,
        }

    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as e:
        # Log errors but return None
        print(f"Error extracting video data: {e}")
        return None
