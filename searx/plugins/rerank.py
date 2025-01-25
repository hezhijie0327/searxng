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

    # 获取 documents 和 scores
    documents, scores = retriever.retrieve(query_tokens, k=len(results), return_as='tuple', show_progress=False)

    # 遍历文档和分数
    for position, (doc, score) in enumerate(zip(documents[0], scores[0]), start=1):
        # 在 `results` 中找到与 `doc` 对应的索引
        index = next(
            (i for i, result in enumerate(results) 
             if doc.startswith(result.get('content', ''))), 
            None
        )
        if index is not None:
            # 确保 `positions` 字段存在，并更新
            if 'positions' in results[index]:
                results[index]['positions'] = [
                    pos / score if score > 0 else pos  # 防止除以 0
                    for pos in results[index]['positions']
                ]
            # 添加排名位置和分数
            results[index]['position'] = position
            results[index]['score'] = score  # 将分数附加到结果中

    return True
