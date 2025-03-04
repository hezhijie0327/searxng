# SPDX-License-Identifier: AGPL-3.0-or-later
"""ChinaSo: A search engine from ChinaSo."""

from urllib.parse import urlencode
from datetime import datetime

from searx.exceptions import SearxEngineAPIException

about = {
    "website": "https://www.chinaso.com/",
    "wikidata_id": "Q10846064",
    "use_official_api": False,
    "require_api_key": False,
    "results": "JSON",
}

paging = True
time_range_support = True
results_per_page = 10
categories = []
chinaso_category = 'news'
"""ChinaSo supports news, videos, images search.

- ``news``: search for news
- ``videos``: search for videos
- ``images``: search for images
"""

time_range_dict = {'day': '24h', 'week': '1w', 'month': '1m', 'year': '1y'}

base_url = "https://www.chinaso.com"


def request(query, params):
    if time_range_dict.get(params['time_range']):
      query_params["stime"] = time_range_dict.get(params['time_range'])
      query_params["etime"] = 'now'

    if chinaso_category == 'news':
        query_params = {
          "pn": params["pageno"],
          "ps": 10,
          "q": query
        }

        params["url"] = f"{base_url}/v5/general/v1/web/search?{urlencode(query_params)}"

    if chinaso_category == 'images':
        query_params = {
          "start_index": (params["pageno"] - 1) * 10,
          "rn": 10,
          "q": query
        }

        params["url"] = f"{base_url}/v5/general/v1/search/image?{urlencode(query_params)}"

    if chinaso_category == 'videos':
        query_params = {
          "start_index": (params["pageno"] - 1) * 10,
          "rn": 10,
          "q": query
        }
        
        params["url"] = f"{base_url}/v5/general/v1/search/video?{urlencode(query_params)}"

    return params


def response(resp):
    try:
        data = resp.json()
    except Exception as e:
        raise SearxEngineAPIException(f"Invalid response: {e}") from e
    results = []

    if chinaso_category == 'news':
      if "data" not in data or "data" not in data["data"]:
          raise SearxEngineAPIException("Invalid response")

      for entry in data["data"]["data"]:
          results.append(
              {
                  "title": entry.get("title"),
                  "url": entry.get("url"),
                  "content": entry.get("content"),
              }
          )

    if chinaso_category == 'images':
      if "data" not in data or "arrRes" not in data["data"]:
          raise SearxEngineAPIException("Invalid response")

      for entry in data["data"]["arrRes"]:
          results.append(
              {
                  'url': entry.get("web_url"),
                  'title': entry.get("title"),
                  'content': entry.get("summary"),
                  'template': 'images.html',
                  'img_src': entry.get("largeimage"),
                  'thumbnail_src': entry.get("smallimage"),
              }
          )

    if chinaso_category == 'videos':
      if "data" not in data or "arrRes" not in data["data"]:
          raise SearxEngineAPIException("Invalid response")

      for entry in data["data"]["arrRes"]:
          published_date = None
          if entry.get("VideoPubDate"):
            try:
                published_date = datetime.fromtimestamp(int(entry["VideoPubDate"]))
            except (ValueError, TypeError):
                published_date = None

          results.append(
              {
                  'url': entry["url"],
                  'title': entry["raw_title"],
                  'template': 'videos.html',
                  'publishedDate': published_date,
                  'thumbnail': entry["image_src"],
              }
          )

      return results
