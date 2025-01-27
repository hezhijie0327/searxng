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

    corpus = [f"{result.get('content', '')} | {result.get('title', '')} | {result.get('url', '')}" for result in results]

    stopwords = {
        word for name, value in stopwords_module.__dict__.items()
        if name.startswith("STOPWORDS_") and isinstance(value, tuple) for word in value
    }

    corpus_tokens = bm25s.tokenize(corpus, stopwords=stopwords)
    query_tokens = bm25s.tokenize(query, stopwords=stopwords)

    retriever = bm25s.BM25(corpus=corpus, backend="numba")
    retriever.index(corpus_tokens)

    documents, scores = retriever.retrieve(query_tokens, k=len(results), return_as='tuple', show_progress=False)

    for idx, doc in enumerate(documents[0]):
        if idx < len(results) and isinstance(results[idx].get('positions'), list):
            score = 1 + scores[0][idx]
            results[idx]['positions'] = [
                float(position * score) if isinstance(position, (int, float)) else position
                for position in results[idx]['positions']
            ]

    return True
