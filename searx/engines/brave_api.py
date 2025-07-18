# SPDX-License-Identifier: AGPL-3.0-or-later
"""Brave (API) for SearXNG."""

from urllib.parse import urlencode

from searx.utils import html_to_text
from searx.result_types import EngineResults, MainResult

about = {
    "website": 'https://search.brave.com/',
    "wikidata_id": 'Q22906900',
    "official_api_documentation": 'https://api-dashboard.search.brave.com/app/documentation/web-search/get-started',
    "use_official_api": True,
    "require_api_key": True,
    "results": 'JSON',
}

categories = ['general']

paging = True
results_per_page = 10

safesearch = True
safesearch_map = {0: 'off', 1: 'moderate', 2: 'strict'}

time_range_support = True
time_range_dict = {'day': 'pd', 'week': 'pw', 'month': 'pm', 'year': 'py'}

api_key = None

base_url = "https://api.search.brave.com/res/v1"


def request(query, params):
    query_params = {
        "count": results_per_page,
        "q": query,
        "result_filter": "web",
        "safesearch": safesearch_map[params['safesearch']],
        "offset": params["pageno"] - 1,
    }

    if time_range_dict.get(params['time_range']):
        query_params["freshness"] = time_range_dict.get(params['time_range'])

    params['url'] = f'{base_url}/web/search?{urlencode(query_params)}'

    params['headers']['Accept'] = 'application/json'
    params['headers']['Accept-Encoding'] = 'gzip'
    params['headers']['X-Subscription-Token'] = api_key
    return params


def response(resp) -> EngineResults:
    results = EngineResults()
    search_results = resp.json()

    for item in search_results.get('web', {}).get('results', []):
        results.add(
            MainResult(
                title=item.get('title'),
                content=html_to_text(item.get('description')),
                url=item.get('url'),
            )
        )

    return results
