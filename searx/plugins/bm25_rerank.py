# SPDX-License-Identifier: AGPL-3.0-or-later
# pylint: disable=missing-module-docstring, missing-class-docstring, protected-access
from __future__ import annotations
import re
import typing

import numpy as np
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
TITLE_PREVIEW_LENGTH = 100
CONTENT_PREVIEW_LENGTH = 150
DISPLAY_TOP_K = 5
MIN_POSITION_VALUE = 0.01
MAX_MULTIPLIER = 1.0
MIN_MULTIPLIER = 0.1
MULTIPLIER_SCALE = 0.8


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
        except (KeyError, ValueError, ImportError):
            # 捕获更具体的异常：模型不存在、参数错误、导入错误
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
                    token_str = self.tokenizer.decode([token])
                    if token_str.strip():
                        token_strings.append(token_str.strip())
                except (UnicodeDecodeError, ValueError):
                    # 捕获解码相关的特定异常
                    continue
            return token_strings
        except (AttributeError, ValueError, TypeError):
            # 捕获编码相关的特定异常
            return preprocessed_text.split()

    def _extract_result_text(self, result) -> str:
        """提取搜索结果的文本内容"""
        text_parts = []

        if hasattr(result, 'title') and result.title:
            text_parts.append(str(result.title))

        if hasattr(result, 'content') and result.content:
            text_parts.append(str(result.content))

        if hasattr(result, 'url') and result.url:
            url_words = re.findall(r'[a-zA-Z\u4e00-\u9fff]+', str(result.url))
            if url_words:
                text_parts.append(' '.join(url_words))

        return " ".join(text_parts)

    def _build_corpus(self, results: list) -> list[str]:
        """构建语料库"""
        return [self._extract_result_text(result) for result in results]

    def _normalize_scores(self, scores: np.ndarray) -> list[float]:
        """标准化分数到0-1范围"""
        if len(scores) == 0:
            return []

        scores = scores.astype(float)
        min_score, max_score = float(np.min(scores)), float(np.max(scores))

        if max_score > min_score:
            normalized = (scores - min_score) / (max_score - min_score)
            return normalized.tolist()
        return [0.5] * len(scores)

    def _calculate_multiplier(self, bm25_score: float) -> float:
        """计算位置权重倍数"""
        multiplier = MAX_MULTIPLIER - bm25_score * MULTIPLIER_SCALE
        return max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, multiplier))

    def _perform_bm25_search(self, corpus_tokens: list, query_tokens: list) -> tuple:
        """执行BM25搜索"""
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        # 尝试多种检索方式
        for method_func in [
            lambda: self._try_get_scores(retriever, query_tokens),
            lambda: self._try_retrieve_batch(retriever, query_tokens),
            lambda: self._try_retrieve_fallback(retriever, query_tokens),
        ]:
            try:
                documents, scores_array = method_func()
                return documents, scores_array
            except (AttributeError, ValueError, RuntimeError, IndexError):
                # 捕获BM25检索可能出现的具体异常
                continue

        raise RuntimeError("所有BM25检索方法都失败")

    def _try_get_scores(self, retriever, query_tokens: list) -> tuple:
        """尝试使用get_scores方法"""
        scores = retriever.get_scores(query_tokens)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [sorted_indices], [scores[sorted_indices]]

    def _try_retrieve_batch(self, retriever, query_tokens: list) -> tuple:
        """尝试使用retrieve方法（批处理）"""
        query_batch = [query_tokens]
        return retriever.retrieve(query_batch, k=len(query_tokens), return_as="tuple", show_progress=False)

    def _try_retrieve_fallback(self, retriever, query_tokens: list) -> tuple:
        """回退检索方法"""
        scores = retriever.get_scores(query_tokens)
        sorted_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [[pair[0] for pair in sorted_pairs]], [[pair[1] for pair in sorted_pairs]]

    def _update_result_positions(self, result, doc_index: int, multiplier: float, total_results: int) -> None:
        """更新单个结果的位置权重"""
        if hasattr(result, 'positions') and result.positions:
            for i, position in enumerate(result.positions):
                if isinstance(position, (int, float)):
                    result.positions[i] = float(max(MIN_POSITION_VALUE, position * multiplier))
        else:
            initial_position = float(doc_index + 1.0) / total_results
            result.positions = [float(initial_position * multiplier)]

    def _apply_rerank(self, results: list, documents: list, normalized_scores: list[float]) -> None:
        """应用重排序到搜索结果"""
        if not documents or not documents[0] or not normalized_scores:
            return

        for idx, doc_index in enumerate(documents[0]):
            if doc_index >= len(results) or idx >= len(normalized_scores):
                continue

            result = results[doc_index]
            bm25_score = normalized_scores[idx]
            multiplier = self._calculate_multiplier(bm25_score)

            self._update_result_positions(result, doc_index, multiplier, len(results))

    def _process_results(self, results: list, query: str) -> None:
        """处理搜索结果并重新排序"""
        if len(results) < 2:
            return

        try:
            # 构建语料库和分词
            corpus = self._build_corpus(results)
            if not corpus:
                return

            corpus_tokens = [self._tokenize_text(doc) for doc in corpus]
            query_tokens = self._tokenize_text(query)

            if not query_tokens:
                return

            # 执行BM25搜索
            documents, scores_array = self._perform_bm25_search(corpus_tokens, query_tokens)

            # 标准化分数并重排
            if documents and scores_array:
                normalized_scores = self._normalize_scores(np.array(scores_array[0]))
                if normalized_scores:
                    self._apply_rerank(results, documents, normalized_scores)

        except (RuntimeError, ValueError, TypeError, AttributeError):
            # 捕获处理过程中可能出现的具体异常
            pass

    def post_search(self, request: "SXNG_Request", search: "SearchWithPlugins") -> EngineResults:
        """搜索后处理钩子"""
        try:
            results = search.result_container.get_ordered_results()

            if len(results) >= 2:
                query = search.search_query.query
                if query:
                    self._process_results(results, query)

        except (AttributeError, TypeError):
            # 捕获访问搜索对象属性时可能出现的具体异常
            pass

        return search.result_container
