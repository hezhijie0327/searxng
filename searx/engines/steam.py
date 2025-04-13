# SPDX-License-Identifier: AGPL-3.0-or-later
"""Steam engine for searxng"""

from urllib.parse import urlencode

from searx.network import get
from searx.utils import html_to_text

about = {
    "website": 'https://store.steampowered.com/',
    "wikidata_id": 'Q337535',
    "use_official_api": True,
    "require_api_key": False,
    "results": 'JSON',
}

categories = []

base_url = "https://store.steampowered.com"

steam_app_details = False
"""
Whether to fetch detailed game info (e.g., descriptions) from the Steam API.

When True:
1. Makes an extra /api/appdetails request for each search result to enrich the data.
2. May impact performance due to:
   - Increased response time
   - Potential API rate limits

Default: False
"""


def request(query, params):
    query_params = {"term": query, "cc": "us", "l": "en"}
    params['url'] = f'{base_url}/api/storesearch/?{urlencode(query_params)}'
    return params


def response(resp):
    results = []
    search_results = resp.json()

    for item in search_results.get('items', []):
        app_id = item.get('id')

        currency = item.get('price', {}).get('currency', 'USD')
        price = item.get('price', {}).get('final', 0) / 100

        platforms = ', '.join([platform for platform, supported in item.get('platforms', {}).items() if supported])

        content = [f'Price: {price:.2f} {currency}', f'Platforms: {platforms}']

        if steam_app_details:
            app_data = get(f'{base_url}/api/appdetails/?appids={app_id}').json().get(str(app_id), {})

            description = app_data.get('data', {}).get('short_description') if app_data.get('success') else None
            if description:
                content.append(f'Description: {description}')

        results.append(
            {
                'title': item.get('name'),
                'content': html_to_text(' | '.join(content)),
                'url': f'{base_url}/app/{app_id}',
                'thumbnail': item.get('tiny_image', ''),
            }
        )

    return results
