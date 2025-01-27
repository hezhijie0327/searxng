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
    doc_to_index = {doc: idx for idx, doc in enumerate(corpus)}

    stopwords = {
        word for name, value in stopwords_module.__dict__.items()
        if name.startswith("STOPWORDS_") and isinstance(value, tuple) for word in value
    }

    corpus_tokens = bm25s.tokenize(corpus, stopwords=stopwords)
    query_tokens = bm25s.tokenize(query, stopwords=stopwords)

    retriever = bm25s.BM25(corpus=corpus, backend="numba")
    retriever.index(corpus_tokens)

    documents, scores = retriever.retrieve(query_tokens, k=len(results), return_as='tuple', show_progress=False)

    for doc in documents[0]:
        index = doc_to_index.get(doc)
        if index is not None and isinstance(results[index].get('positions'), list):
            score = 1 + scores[0][doc_to_index[doc]]
            results[index]['positions'] = [
                float(position * score) if isinstance(position, (int, float)) else position
                for position in results[index]['positions']
            ]

    return True
