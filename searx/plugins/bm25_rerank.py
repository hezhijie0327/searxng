# SPDX-License-Identifier: AGPL-3.0-or-later
# pylint: disable=missing-module-docstring, missing-class-docstring, protected-access
from __future__ import annotations
import typing

import bm25s
import bm25s.stopwords as stopwords_module

from searx.plugins import Plugin, PluginInfo
from searx.result_types import EngineResults

if typing.TYPE_CHECKING:
    from searx.search import SearchWithPlugins
    from searx.extended_types import SXNG_Request
    from searx.plugins import PluginCfg


class SXNGPlugin(Plugin):
    """Rerank search results using the Okapi BM25 algorithm. The
    results are reordered to improve relevance based on the query.
    """

    id = "bm25_rerank"
    default_on = True

    def __init__(self, plg_cfg: "PluginCfg") -> None:
        super().__init__(plg_cfg)

        self.info = PluginInfo(
            id=self.id,
            name="BM25 Rerank",
            description="Rerank search results using the Okapi BM25 algorithm",
            preference_section="general",
        )

    def post_search(self, request: "SXNG_Request", search: "SearchWithPlugins") -> EngineResults:
        results = search.result_container.get_ordered_results()
        query = search.search_query.query

        corpus = [
            f"{result.content} | {result.title} | {result.url}" for result in results
        ]

        stopwords = {
            word
            for name, value in stopwords_module.__dict__.items()
            if name.startswith("STOPWORDS_") and isinstance(value, tuple)
            for word in value
        }

        corpus_tokens = bm25s.tokenize(corpus, stopwords=stopwords)
        query_tokens = bm25s.tokenize(query, stopwords=stopwords)

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        documents, scores = retriever.retrieve(query_tokens, k=len(corpus), return_as="tuple", show_progress=False)

        for index in documents[0]:
            score = scores[0][index]
            for i, position in enumerate(results[index].positions):
                if isinstance(position, (int, float)):
                    results[index]["positions"][i] = float(position / (score + 1))
