# SPDX-License-Identifier: AGPL-3.0-or-later
# pylint: disable=protected-access
"""Plugin which reranks the search results using the Okapi BM25 algorithm.

Enable in ``settings.yml``:

.. code:: yaml

  enabled_plugins:
    ..
    - 'Rerank plugin'
"""

from searx import settings

import bm25s

name = 'Rerank plugin'
description = 'Rerank search results, ignoring original engine ranking'
default_on = True
preference_section = 'general'


def post_search(_request, search):
    results = search.result_container._merged_results
    query = search.search_query.query

    retriever = bm25s.BM25()
    result_tokens = bm25s.tokenize([f"{result.get('content', '')} | {result.get('title', '')} | {result.get('url', '')}" for result in results])
    retriever.index(result_tokens)

    query_tokens = bm25s.tokenize(query)

    indices = retriever.retrieve(query_tokens, k=len(results), return_as='documents', show_progress=False)

    for position, index in enumerate(indices[0], start=1):
        if 'positions' in results[index]:
            results[index]['positions'] = [position] * ( len(results[index]['positions']) * 0.25 + len(results[index]['engines']) * 0.75 )

    return True
