# SPDX-License-Identifier: AGPL-3.0-or-later
"""Niconico search engine for searxng"""

import time

from urllib.parse import urlencode
from datetime import datetime, timedelta
from lxml import html

from searx.utils import extract_text

about = {
    "website": "https://www.nicovideo.jp/",
    "wikidata_id": "Q697233",
    "use_official_api": False,
    "require_api_key": False,
    "results": "HTML",
    "language": "ja",
}

categories = ["videos"]
paging = True

time_range_support = True
time_range_dict = {"day": 86400, "week": 604800, "month": 2592000, "year": 31536000}

base_url = "https://www.nicovideo.jp"


def request(query, params):
    query_params = {"page": params['pageno']}

    if time_range_dict.get(params['time_range']):
        now = int(time.time())
        past = now - time_range_dict[params["time_range"]]

        query_params['end'] = time.strftime('%Y-%m-%d', time.gmtime(now))
        query_params['start'] = time.strftime('%Y-%m-%d', time.gmtime(past))

    params['url'] = f"{base_url}/search/{query}?{urlencode(query_params)}"
    return params


def response(resp):
    results = []
    dom = html.fromstring(resp.text)

    for item in dom.xpath('//li[@data-video-item]'):
        relative_url = extract_text(item.xpath('.//a[@class="itemThumbWrap"]/@href')[0])
        video_id = relative_url.rsplit('?', maxsplit=1)[0].split('/')[-1]

        url = f"{base_url}/watch/{video_id}"
        iframe_src = f"https://embed.nicovideo.jp/watch/{video_id}"

        length = None
        video_length = extract_text(item.xpath('.//span[@class="videoLength"]'))
        if video_length:
            try:
                timediff = datetime.strptime(video_length, "%M:%S")
                length = timedelta(minutes=timediff.minute, seconds=timediff.second)
            except ValueError:
                pass

        published_date = None
        upload_time = extract_text(item.xpath('.//p[@class="itemTime"]//span[@class="time"]/text()'))
        if upload_time:
            try:
                published_date = datetime.strptime(upload_time, "%Y/%m/%d %H:%M")
            except ValueError:
                pass

        results.append(
            {
                'template': 'videos.html',
                'title': extract_text(item.xpath('.//p[@class="itemTitle"]/a')),
                'content': extract_text(item.xpath('.//p[@class="itemDescription"]/@title')[0]),
                'url': url,
                "iframe_src": iframe_src,
                'thumbnail': extract_text(item.xpath('.//img[@class="thumb"]/@src')[0]),
                'length': length,
                "publishedDate": published_date,
            }
        )

    return results
