# SPDX-License-Identifier: AGPL-3.0-or-later
# pylint: disable=protected-access
"""Plugin which reranks the search results via Okapi BM25 algorithm.

Enable in ``settings.yml``:

.. code:: yaml

  enabled_plugins:
    ..
    - 'Rerank plugin'
"""

from searx import settings

import bm25s
import bm25s.stopwords as stopwords_module

name = 'Rerank plugin'
description = 'Rerank search results via Okapi BM25 algorithm'
default_on = True
preference_section = 'general'


def post_search(_request, search):
    results = search.result_container._merged_results
    query = search.search_query.query

    stopwords = set()
    for name, value in stopwords_module.__dict__.items():
        if name.startswith("STOPWORDS_") and isinstance(value, tuple):
            stopwords.update(value)

    retriever = bm25s.BM25()
    result_tokens = bm25s.tokenize(
        [f"{result.get('content', '')} | {result.get('title', '')} | {result.get('url', '')}" for result in results],
        stopwords=stopwords
    )
    retriever.index(result_tokens)

    query_tokens = bm25s.tokenize(query, stopwords=stopwords)

    documents, scores = retriever.retrieve(query_tokens, k=len(results), return_as='tuple', show_progress=False)

    for index in documents[0]:
        if index < len(results) and isinstance(results[index].get('positions'), list):
            score = 1 + scores[0][index]
            results[index]['positions'] = [
                float(position * score) if isinstance(position, (int, float)) else position
                for position in results[index]['positions']
            ]

    return True
