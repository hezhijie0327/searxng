# SPDX-License-Identifier: AGPL-3.0-or-later
"""This module implements functions needed for the autocompleter.

"""
# pylint: disable=use-dict-literal

import json
import html
from urllib.parse import urlencode, quote_plus

import lxml
from httpx import HTTPError

from searx import settings
from searx.engines import (
    engines,
    google,
)
from searx.network import get as http_get, post as http_post
from searx.exceptions import SearxEngineResponseException

import bm25s
import bm25s.stopwords as stopwords_module


def update_kwargs(**kwargs):
    if 'timeout' not in kwargs:
        kwargs['timeout'] = settings['outgoing']['request_timeout']
    kwargs['raise_for_httperror'] = True


def get(*args, **kwargs):
    update_kwargs(**kwargs)
    return http_get(*args, **kwargs)


def post(*args, **kwargs):
    update_kwargs(**kwargs)
    return http_post(*args, **kwargs)


def baidu(query, _lang):
    # baidu search autocompleter
    base_url = "https://www.baidu.com/sugrec?"
    response = get(base_url + urlencode({'ie': 'utf-8', 'json': 1, 'prod': 'pc', 'wd': query}))

    results = []

    if response.ok:
        data = response.json()
        if 'g' in data:
            for item in data['g']:
                results.append(item['q'])
    return results


def brave(query, _lang):
    # brave search autocompleter
    url = 'https://search.brave.com/api/suggest?'
    url += urlencode({'q': query})
    country = 'all'
    # if lang in _brave:
    #    country = lang
    kwargs = {'cookies': {'country': country}}
    resp = get(url, **kwargs)

    results = []

    if resp.ok:
        data = resp.json()
        for item in data[1]:
            results.append(item)
    return results


def dbpedia(query, _lang):
    # dbpedia autocompleter, no HTTPS
    autocomplete_url = 'https://lookup.dbpedia.org/api/search.asmx/KeywordSearch?'

    response = get(autocomplete_url + urlencode(dict(QueryString=query)))

    results = []

    if response.ok:
        dom = lxml.etree.fromstring(response.content)
        results = dom.xpath('//Result/Label//text()')

    return results


def duckduckgo(query, sxng_locale):
    """Autocomplete from DuckDuckGo. Supports DuckDuckGo's languages"""

    traits = engines['duckduckgo'].traits
    args = {
        'q': query,
        'kl': traits.get_region(sxng_locale, traits.all_locale),
    }

    url = 'https://duckduckgo.com/ac/?type=list&' + urlencode(args)
    resp = get(url)

    ret_val = []
    if resp.ok:
        j = resp.json()
        if len(j) > 1:
            ret_val = j[1]
    return ret_val


def google_complete(query, sxng_locale):
    """Autocomplete from Google.  Supports Google's languages and subdomains
    (:py:obj:`searx.engines.google.get_google_info`) by using the async REST
    API::

        https://{subdomain}/complete/search?{args}

    """

    google_info = google.get_google_info({'searxng_locale': sxng_locale}, engines['google'].traits)

    url = 'https://{subdomain}/complete/search?{args}'
    args = urlencode(
        {
            'q': query,
            'client': 'gws-wiz',
            'hl': google_info['params']['hl'],
        }
    )
    results = []
    resp = get(url.format(subdomain=google_info['subdomain'], args=args))
    if resp.ok:
        json_txt = resp.text[resp.text.find('[') : resp.text.find(']', -3) + 1]
        data = json.loads(json_txt)
        for item in data[0]:
            results.append(lxml.html.fromstring(item[0]).text_content())
    return results


def mwmbl(query, _lang):
    """Autocomplete from Mwmbl_."""

    # mwmbl autocompleter
    url = 'https://api.mwmbl.org/search/complete?{query}'

    results = get(url.format(query=urlencode({'q': query}))).json()[1]

    # results starting with `go:` are direct urls and not useful for auto completion
    return [result for result in results if not result.startswith("go: ") and not result.startswith("search: ")]


def seznam(query, _lang):
    # seznam search autocompleter
    url = 'https://suggest.seznam.cz/fulltext/cs?{query}'

    resp = get(
        url.format(
            query=urlencode(
                {'phrase': query, 'cursorPosition': len(query), 'format': 'json-2', 'highlight': '1', 'count': '6'}
            )
        )
    )

    if not resp.ok:
        return []

    data = resp.json()
    return [
        ''.join([part.get('text', '') for part in item.get('text', [])])
        for item in data.get('result', [])
        if item.get('itemType', None) == 'ItemType.TEXT'
    ]


def stract(query, _lang):
    # stract autocompleter (beta)
    url = f"https://stract.com/beta/api/autosuggest?q={quote_plus(query)}"

    resp = post(url)

    if not resp.ok:
        return []

    return [html.unescape(suggestion['raw']) for suggestion in resp.json()]


def startpage(query, sxng_locale):
    """Autocomplete from Startpage. Supports Startpage's languages"""
    lui = engines['startpage'].traits.get_language(sxng_locale, 'english')
    url = 'https://startpage.com/suggestions?{query}'
    resp = get(url.format(query=urlencode({'q': query, 'segment': 'startpage.udog', 'lui': lui})))
    data = resp.json()
    return [e['text'] for e in data.get('suggestions', []) if 'text' in e]


def swisscows(query, _lang):
    # swisscows autocompleter
    url = 'https://swisscows.ch/api/suggest?{query}&itemsCount=5'

    resp = json.loads(get(url.format(query=urlencode({'query': query}))).text)
    return resp


def qwant(query, sxng_locale):
    """Autocomplete from Qwant. Supports Qwant's regions."""
    results = []

    locale = engines['qwant'].traits.get_region(sxng_locale, 'en_US')
    url = 'https://api.qwant.com/v3/suggest?{query}'
    resp = get(url.format(query=urlencode({'q': query, 'locale': locale, 'version': '2'})))

    if resp.ok:
        data = resp.json()
        if data['status'] == 'success':
            for item in data['data']['items']:
                results.append(item['value'])

    return results


def wikipedia(query, sxng_locale):
    """Autocomplete from Wikipedia. Supports Wikipedia's languages (aka netloc)."""
    results = []
    eng_traits = engines['wikipedia'].traits
    wiki_lang = eng_traits.get_language(sxng_locale, 'en')
    wiki_netloc = eng_traits.custom['wiki_netloc'].get(wiki_lang, 'en.wikipedia.org')

    url = 'https://{wiki_netloc}/w/api.php?{args}'
    args = urlencode(
        {
            'action': 'opensearch',
            'format': 'json',
            'formatversion': '2',
            'search': query,
            'namespace': '0',
            'limit': '10',
        }
    )
    resp = get(url.format(args=args, wiki_netloc=wiki_netloc))
    if resp.ok:
        data = resp.json()
        if len(data) > 1:
            results = data[1]

    return results


def yandex(query, _lang):
    # yandex autocompleter
    url = "https://suggest.yandex.com/suggest-ff.cgi?{0}"

    resp = json.loads(get(url.format(urlencode(dict(part=query)))).text)
    if len(resp) > 1:
        return resp[1]
    return []


backends = {
    'baidu': baidu,
    'brave': brave,
    'dbpedia': dbpedia,
    'duckduckgo': duckduckgo,
    'google': google_complete,
    'mwmbl': mwmbl,
    'qwant': qwant,
    'seznam': seznam,
    'startpage': startpage,
    'stract': stract,
    'swisscows': swisscows,
    'wikipedia': wikipedia,
    'yandex': yandex,
    'all': 'all',
    'custom': 'custom',
}


def deduplicate_results(results):
    seen = set()
    unique_results = []
    for result in results:
        if result not in seen:
            unique_results.append(result)
            seen.add(result)
    return unique_results


def rerank_results(results_list, query):
    corpus = deduplicate_results([result for results in results_list for result in results])

    stopwords = {
        word for name, value in stopwords_module.__dict__.items()
        if name.startswith("STOPWORDS_") and isinstance(value, tuple) for word in value
    }

    corpus_tokens = bm25s.tokenize(corpus, stopwords=stopwords)
    query_tokens = bm25s.tokenize(query, stopwords=stopwords)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    documents, scores = retriever.retrieve(query_tokens, k=len(corpus), return_as='tuple', show_progress=False)

    ranked_results = [
        corpus[index] for index, _ in sorted(zip(documents[0], scores[0]), key=lambda x: x[1], reverse=True)
    ]

    return ranked_results


def search_autocomplete(backend_name, query, sxng_locale):
    excluded_backends = ['all', 'custom']

    if backend_name == 'all':
        results_list = []
        for backend_key, backend in backends.items():
            if backend_key not in excluded_backends:
                try:
                    results_list.append(backend(query, sxng_locale))
                except (HTTPError, SearxEngineResponseException, ValueError):
                    results_list.append([])
        return rerank_results(results_list, query)

    elif backend_name == 'custom':
        custom_backends = settings.get('search', {}).get('autocomplete_engines', [])

        custom_backends = [backend.strip() for backend in custom_backends if backend.strip() in backends]

        results_list = []
        for backend_key in custom_backends:
            backend = backends.get(backend_key)
            if backend is not None:
                try:
                    results_list.append(backend(query, sxng_locale))
                except (HTTPError, SearxEngineResponseException, ValueError):
                    results_list.append([])
        return rerank_results(results_list, query)

    else:
        backend = backends.get(backend_name)
        if backend is None:
            return []
        try:
            return backend(query, sxng_locale)
        except (HTTPError, SearxEngineResponseException, ValueError):
            return []
