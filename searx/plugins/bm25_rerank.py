# SPDX-License-Identifier: AGPL-3.0-or-later
# pylint: disable=missing-module-docstring, missing-class-docstring, protected-access
from __future__ import annotations
import re
import typing

import bm25s
import tiktoken

from searx.plugins import Plugin, PluginInfo
from searx.result_types import EngineResults

if typing.TYPE_CHECKING:
    from searx.search import SearchWithPlugins
    from searx.extended_types import SXNG_Request
    from searx.plugins import PluginCfg

# 常量定义
PRIMARY_MODEL = "gpt-oss-120b"
FALLBACK_ENCODING = "o200k_harmony"


class SXNGPlugin(Plugin):
    """Rerank search results using BM25 algorithm with tiktoken tokenizer."""

    id = "bm25_rerank"
    default_on = True

    def __init__(self, plg_cfg: "PluginCfg") -> None:
        super().__init__(plg_cfg)
        self.info = PluginInfo(
            id=self.id,
            name="BM25 Rerank",
            description="Rerank search results using BM25 algorithm with tiktoken tokenizer",
            preference_section="general",
        )
        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        """初始化tiktoken编码器"""
        try:
            self.tokenizer = tiktoken.encoding_for_model(PRIMARY_MODEL)
        except (KeyError, ValueError):
            self.tokenizer = tiktoken.get_encoding(FALLBACK_ENCODING)

    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        return text.strip()

    def _tokenize_text(self, text: str) -> list[str]:
        """使用tiktoken进行分词"""
        if not text:
            return []

        preprocessed_text = self._preprocess_text(text)
        if not preprocessed_text:
            return []

        try:
            tokens = self.tokenizer.encode(preprocessed_text)
            token_strings = []
            for token in tokens:
                try:
                    token_str = self.tokenizer.decode([token]).strip()
                    if token_str:
                        token_strings.append(token_str)
                except (ValueError, UnicodeDecodeError):
                    continue
            return token_strings
        except (ValueError, TypeError):
            return preprocessed_text.split()

    def _extract_result_text(self, result) -> str:
        """提取搜索结果的文本内容"""
        text_parts = []

        for attr in ['title', 'content']:
            if hasattr(result, attr) and getattr(result, attr):
                text_parts.append(str(getattr(result, attr)))

        if hasattr(result, 'url') and result.url:
            url_words = re.findall(r'[a-zA-Z\u4e00-\u9fff]+', str(result.url))
            if url_words:
                text_parts.append(' '.join(url_words))

        return " ".join(text_parts)

    def _get_bm25_scores(self, results: list, query: str) -> list[float]:
        """获取所有文档的BM25分数"""
        if not results or not query:
            return [0.0] * len(results)

        try:
            # 构建语料库和分词
            corpus_tokens = []
            for result in results:
                text = self._extract_result_text(result)
                tokens = self._tokenize_text(text)
                corpus_tokens.append(tokens)

            query_tokens = self._tokenize_text(query)
            if not query_tokens or not any(corpus_tokens):
                return [0.0] * len(results)

            # 计算BM25分数
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)
            scores = retriever.get_scores(query_tokens)

            # 转换为列表格式
            if hasattr(scores, 'tolist'):
                return scores.tolist()

            return [float(score) for score in scores]

        except (ValueError, TypeError, AttributeError, ImportError):
            return [0.0] * len(results)

    def _update_positions(self, results: list, bm25_scores: list[float]) -> None:
        """将BM25分数更新到结果的positions中"""
        # 确保分数数量与结果数量一致
        while len(bm25_scores) < len(results):
            bm25_scores.append(0.0)

        for i, result in enumerate(results):
            # BM25分数
            position_score = bm25_scores[i]

            try:
                # 获取原有positions
                original_positions = getattr(result, 'positions', [])

                # 将BM25分数插入第一位
                result.positions = [position_score] + list(original_positions)
            except AttributeError:
                try:
                    result.positions = [position_score]
                except AttributeError:
                    pass

    def post_search(self, request: "SXNG_Request", search: "SearchWithPlugins") -> EngineResults:
        """搜索后处理钩子"""
        try:
            # 获取搜索结果
            results = getattr(search.result_container, 'results', None)
            if not results and hasattr(search.result_container, 'get_ordered_results'):
                results = search.result_container.get_ordered_results()

            if results and len(results) > 0:
                query = search.search_query.query
                if query:
                    # 计算BM25分数并更新positions
                    bm25_scores = self._get_bm25_scores(results, query)
                    self._update_positions(results, bm25_scores)

        except AttributeError:
            pass

        return search.result_container
