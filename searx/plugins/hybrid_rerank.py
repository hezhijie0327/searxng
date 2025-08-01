# SPDX-License-Identifier: AGPL-3.0-or-later
# pylint: disable=missing-module-docstring, missing-class-docstring, protected-access
from __future__ import annotations
import math
import re
import typing
import hashlib
import time
import os
from functools import lru_cache
from threading import Lock
from pathlib import Path
import logging

import numpy as np
import bm25s
import bm25s.stopwords as stopwords_module

from searx.plugins import Plugin, PluginInfo
from searx.result_types import EngineResults

# Lazy import for sentence-transformers to avoid startup overhead
sentence_transformers = None
torch = None

if typing.TYPE_CHECKING:
    from searx.search import SearchWithPlugins
    from searx.extended_types import SXNG_Request
    from searx.plugins import PluginCfg

logger = logging.getLogger(__name__)

# 🚀 预编译正则表达式以提升性能
HTML_TAG_RE = re.compile(r'<[^>]+>')
WHITESPACE_RE = re.compile(r'\s+')
URL_WORD_RE = re.compile(r'[a-zA-Z]+')

# 📁 模型路径配置
DEFAULT_MODEL_DIR = os.environ.get('SEARXNG_MODEL_DIR', '/var/cache/searxng/models')
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

class ScoreDetails:
    """轻量级分数详情类"""
    __slots__ = [
        'doc_index',
        'title',
        'bm25_raw',
        'bm25_normalized',
        'semantic_raw',
        'semantic_normalized',
        'combined_score',
        'final_position',
        'position_multiplier',
    ]

    def __init__(self, doc_index: int, title: str = ""):
        self.doc_index = doc_index
        self.title = title[:40] + "..." if len(title) > 40 else title  # 减少字符数
        self.bm25_raw = 0.0
        self.bm25_normalized = 0.0
        self.semantic_raw = 0.0
        self.semantic_normalized = 0.0
        self.combined_score = 0.0
        self.final_position = 0.0
        self.position_multiplier = 1.0


class FastSemanticCache:
    """高性能语义向量缓存"""

    def __init__(self, max_size: int = 800, ttl: int = 3600):
        self.cache = {}
        self.access_times = {}  # 简化时间戳管理
        self.max_size = max_size
        self.ttl = ttl
        self.lock = Lock()
        self.hit_count = 0
        self.miss_count = 0

        logger.info(f"⚡ FastSemanticCache initialized: {max_size} entries")

    @lru_cache(maxsize=2048)
    def _generate_key(self, text: str) -> str:
        """缓存的键生成"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]  # 缩短key长度

    def get(self, text: str) -> np.ndarray | None:
        """快速获取缓存的向量"""
        key = self._generate_key(text)
        current_time = time.time()

        with self.lock:
            if key in self.cache:
                # 简化过期检查
                if current_time - self.access_times.get(key, 0) < self.ttl:
                    self.hit_count += 1
                    self.access_times[key] = current_time
                    return self.cache[key]
                else:
                    # 快速清理
                    del self.cache[key]
                    del self.access_times[key]

        self.miss_count += 1
        return None

    def set(self, text: str, vector: np.ndarray) -> None:
        """快速设置缓存"""
        key = self._generate_key(text)
        current_time = time.time()

        with self.lock:
            # 简化缓存满处理
            if len(self.cache) >= self.max_size:
                # 删除最旧的3个条目（批量清理）
                old_keys = sorted(self.access_times.items(), key=lambda x: x[1])[:3]
                for old_key, _ in old_keys:
                    self.cache.pop(old_key, None)
                    self.access_times.pop(old_key, None)

            self.cache[key] = vector
            self.access_times[key] = current_time

    def get_hit_rate(self) -> float:
        """快速获取命中率"""
        total = self.hit_count + self.miss_count
        return (self.hit_count / total * 100) if total > 0 else 0


class ModelManager:
    """优化的模型管理器 - 简化逻辑：本地不存在就下载"""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, model_name: str = DEFAULT_MODEL_NAME):
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.local_model_path = self.model_dir / model_name

        # 确保模型目录存在
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Model manager initialized:")
        logger.info(f"   - Model dir: {self.model_dir}")
        logger.info(f"   - Model name: {self.model_name}")
        logger.info(f"   - Local path: {self.local_model_path}")

    def _is_model_valid(self) -> bool:
        """检查本地模型是否完整有效"""
        if not self.local_model_path.exists():
            return False

        # 检查必要文件是否存在
        required_files = ['config.json']
        model_files = ['pytorch_model.bin', 'model.safetensors']

        # 检查配置文件
        if not (self.local_model_path / 'config.json').exists():
            return False

        # 检查模型文件（至少有一个）
        has_model_file = any((self.local_model_path / f).exists() for f in model_files)
        if not has_model_file:
            return False

        logger.debug(f"✅ Local model validated: {self.local_model_path}")
        return True

    def _download_model_to_local(self) -> bool:
        """下载模型到本地目录"""
        try:
            logger.info(f"⬇️ Downloading model '{self.model_name}' to {self.local_model_path}")

            # 导入sentence_transformers
            global sentence_transformers
            if sentence_transformers is None:
                import sentence_transformers

            # 从HuggingFace下载模型
            model = sentence_transformers.SentenceTransformer(self.model_name)

            # 保存到本地目录
            model.save(str(self.local_model_path))

            # 验证下载结果
            if self._is_model_valid():
                logger.info(f"✅ Model successfully downloaded and saved to {self.local_model_path}")
                return True
            else:
                logger.error(f"❌ Downloaded model validation failed")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
            return False

    def ensure_model_available(self) -> str:
        """确保模型可用，返回本地模型路径"""
        # 1. 检查本地模型是否存在且有效
        if self._is_model_valid():
            logger.debug(f"📁 Using existing local model: {self.local_model_path}")
            return str(self.local_model_path)

        # 2. 本地不存在，下载到本地
        logger.info(f"📥 Local model not found, downloading...")
        if self._download_model_to_local():
            return str(self.local_model_path)

        # 3. 下载失败，抛出异常
        raise RuntimeError(f"Failed to ensure model availability: {self.model_name}")

    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = {
            'model_name': self.model_name,
            'model_dir': str(self.model_dir),
            'local_path': str(self.local_model_path),
            'exists': self.local_model_path.exists(),
            'valid': self._is_model_valid(),
        }

        if self.local_model_path.exists():
            try:
                # 获取模型大小
                total_size = sum(f.stat().st_size for f in self.local_model_path.rglob('*') if f.is_file())
                info['size_mb'] = round(total_size / (1024 * 1024), 2)
                info['file_count'] = len(list(self.local_model_path.rglob('*')))
            except Exception:
                info['size_mb'] = 0
                info['file_count'] = 0

        return info


class SXNGPlugin(Plugin):
    """Hybrid Rerank: CPU-optimized semantic + BM25 fusion plugin"""

    id = "hybrid_rerank"
    default_on = True

    def __init__(self, plg_cfg: "PluginCfg") -> None:
        super().__init__(plg_cfg)

        # 🚀 混合权重配置：语义优先
        self.semantic_weight = 0.75
        self.bm25_weight = 0.25

        # 📁 模型管理 - 使用优化的逻辑
        self.model_manager = ModelManager(
            model_dir=os.environ.get('SEARXNG_MODEL_DIR', DEFAULT_MODEL_DIR),
            model_name=os.environ.get('SEARXNG_MODEL_NAME', DEFAULT_MODEL_NAME),
        )

        # 🔥 CPU性能优化的批处理配置
        self.cpu_batch_size = 16  # 降低到CPU友好的大小
        self.cpu_max_batch_size = 32  # 最大批量限制
        self.cpu_chunk_size = 8  # 内存友好的块大小
        self.max_results_limit = 200  # 降低处理限制
        self.max_length = 200  # 减少文本长度

        # 🚀 性能控制
        self.cpu_timeout = 20.0  # 更严格的超时
        self.performance_threshold = 8.0  # 更严格的性能阈值
        self.enable_fast_mode = True  # 启用快速模式

        # 📊 简化显示配置
        self.show_detailed_scores = True
        self.show_top_n_scores = 8  # 减少显示数量
        self.show_score_distribution = False  # 关闭分布统计以节省CPU
        self.enable_debug_logs = False  # 关闭详细调试日志

        # 运行时状态
        self._model = None
        self._model_load_failed = False
        self._semantic_cache = FastSemanticCache(max_size=800, ttl=3600)
        self._performance_stats = {
            "total_calls": 0,
            "avg_time": 0,
            "cpu_optimized_calls": 0,
            "fast_mode_calls": 0,
            "timeout_fallbacks": 0,
        }

        # 显示模型信息
        model_info = self.model_manager.get_model_info()

        logger.info(f"🔄 HYBRID RERANK Plugin initialized:")
        logger.info(f"   - Semantic weight: {self.semantic_weight} ({self.semantic_weight*100:.0f}%)")
        logger.info(f"   - BM25 weight: {self.bm25_weight} ({self.bm25_weight*100:.0f}%)")
        logger.info(f"   - CPU batch size: {self.cpu_batch_size}")
        logger.info(f"   - Max results: {self.max_results_limit}")
        logger.info(f"   - Fast mode: {self.enable_fast_mode}")
        logger.info(f"   - Model status: {'✅ Valid' if model_info['valid'] else '⏳ Will download on first use'}")

        if model_info['valid']:
            logger.info(f"   - Model size: {model_info.get('size_mb', 0):.1f} MB")

        self.info = PluginInfo(
            id=self.id,
            name="Hybrid Rerank",
            description="A hybrid reranking method that combines traditional information retrieval with modern semantic understanding.",
            preference_section="general",
        )

    def _lazy_load_model(self) -> bool:
        """优化的模型加载 - 始终从本地加载"""
        if self._model is not None:
            return True

        if self._model_load_failed:
            return False

        try:
            global sentence_transformers, torch
            if sentence_transformers is None:
                import sentence_transformers
                import torch

            # CPU优化设置
            torch.set_num_threads(2)  # 保守的线程数
            torch.set_grad_enabled(False)  # 禁用梯度计算

            # 确保模型可用并获取本地路径
            model_path = self.model_manager.ensure_model_available()

            logger.info(f"📥 Loading hybrid rerank model from: {model_path}")

            self._model = sentence_transformers.SentenceTransformer(model_path, device='cpu')

            # 快速预热
            _ = self._model.encode(["hybrid rerank warmup"], show_progress_bar=False)

            logger.info(f"⚡ Hybrid rerank model loaded successfully with CPU optimizations")
            return True

        except Exception as e:
            logger.error(f"❌ Hybrid rerank model loading failed: {e}")
            self._model_load_failed = True
            return False

    def _fast_preprocess_text(self, text: str) -> str:
        """高性能文本预处理"""
        if not text:
            return ""

        # 使用预编译的正则表达式
        text = HTML_TAG_RE.sub(' ', text)
        text = WHITESPACE_RE.sub(' ', text.strip())

        # 快速长度限制
        return text[: self.max_length] if len(text) > self.max_length else text

    def _compute_embeddings_cpu_optimized(self, texts: list) -> np.ndarray | None:
        """CPU优化的嵌入计算"""
        if not texts:
            return None

        embeddings = []
        texts_to_compute = []
        indices_map = {}

        # 快速缓存检查
        for i, text in enumerate(texts):
            cached = self._semantic_cache.get(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                texts_to_compute.append(text)
                indices_map[len(texts_to_compute) - 1] = i

        if not texts_to_compute:
            return np.array(embeddings)

        try:
            # CPU友好的小批量处理
            computed_embeddings = []
            batch_size = min(self.cpu_batch_size, len(texts_to_compute))

            for i in range(0, len(texts_to_compute), batch_size):
                batch = texts_to_compute[i : i + batch_size]
                batch_emb = self._model.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # 预先标准化
                )
                computed_embeddings.extend(batch_emb)

            # 更新缓存和结果
            for comp_idx, embedding in enumerate(computed_embeddings):
                orig_idx = indices_map[comp_idx]
                embeddings[orig_idx] = embedding
                self._semantic_cache.set(texts[orig_idx], embedding)

            return np.array(embeddings)

        except Exception as e:
            logger.error(f"❌ Hybrid embedding computation failed: {e}")
            return None

    def _calculate_cpu_optimized_semantic_scores(self, query: str, corpus: list) -> np.ndarray | None:
        """CPU优化的语义分数计算"""
        if not self._lazy_load_model():
            return None

        start_time = time.time()
        corpus_size = len(corpus)

        # 严格的大小限制
        if corpus_size > self.max_results_limit:
            logger.warning(f"⚠️ Truncating {corpus_size} to {self.max_results_limit} for CPU performance")
            corpus = corpus[: self.max_results_limit]
            corpus_size = len(corpus)

        try:
            # 快速预处理
            processed_query = self._fast_preprocess_text(query)
            processed_corpus = [self._fast_preprocess_text(text) for text in corpus]

            # 检查超时
            if time.time() - start_time > self.cpu_timeout:
                logger.warning("⏱️ CPU timeout during preprocessing")
                return None

            # 获取查询嵌入
            query_embedding = self._semantic_cache.get(processed_query)
            if query_embedding is None:
                query_embedding = self._model.encode(
                    [processed_query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
                )[0]
                self._semantic_cache.set(processed_query, query_embedding)

            # 检查超时
            if time.time() - start_time > self.cpu_timeout:
                logger.warning("⏱️ CPU timeout during query embedding")
                return None

            # 获取文档嵌入
            doc_embeddings = self._compute_embeddings_cpu_optimized(processed_corpus)
            if doc_embeddings is None:
                return None

            # 快速相似度计算（已经预先标准化）
            similarities = np.dot(doc_embeddings, query_embedding)

            elapsed = time.time() - start_time
            throughput = corpus_size / elapsed if elapsed > 0 else 0

            if self.enable_debug_logs:
                logger.info(f"⚡ Hybrid semantic processing: {elapsed:.2f}s, {throughput:.1f} docs/s")

            # 性能检查
            if elapsed > self.performance_threshold:
                logger.warning(f"🐌 Slow hybrid processing: {elapsed:.2f}s")

            return similarities

        except Exception as e:
            logger.error(f"❌ Hybrid semantic processing failed: {e}")
            return None

    def _fast_normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """快速分数标准化"""
        if len(scores) == 0:
            return scores

        min_val, max_val = np.min(scores), np.max(scores)
        if max_val > min_val:
            return (scores - min_val) / (max_val - min_val)
        return np.full_like(scores, 0.5)

    def _hybrid_score_combination(self, bm25_scores: np.ndarray, semantic_scores: np.ndarray) -> np.ndarray:
        """混合分数组合：语义优先的加权融合"""
        if semantic_scores is None:
            logger.warning("⚠️ Semantic scores unavailable, using BM25 only")
            return bm25_scores

        # 快速标准化
        bm25_norm = self._fast_normalize_scores(bm25_scores)
        semantic_norm = self._fast_normalize_scores(semantic_scores)

        # 混合加权组合：语义优先
        combined = self.bm25_weight * bm25_norm + self.semantic_weight * semantic_norm

        if self.enable_debug_logs:
            logger.info(
                f"🔄 Hybrid scores: avg={np.mean(combined):.3f}, "
                f"BM25_avg={np.mean(bm25_norm):.3f}, "
                f"Semantic_avg={np.mean(semantic_norm):.3f}"
            )

        return combined

    def _create_fast_score_details(
        self,
        results: list,
        documents: list,
        bm25_scores: np.ndarray,
        semantic_scores: np.ndarray,
        combined_scores: np.ndarray
    ) -> list:
        """快速创建分数详情"""
        score_details = []
        doc_mapping = {doc_idx: rank for rank, doc_idx in enumerate(documents[0])}

        for doc_idx in range(min(len(results), self.max_results_limit)):
            result = results[doc_idx]
            title = getattr(result, 'title', f'Doc{doc_idx}') or f'Doc{doc_idx}'

            detail = ScoreDetails(doc_idx, title)

            if doc_idx in doc_mapping:
                rank = doc_mapping[doc_idx]
                detail.bm25_raw = float(bm25_scores[rank])
                detail.combined_score = float(combined_scores[rank])
                if semantic_scores is not None:
                    detail.semantic_raw = float(semantic_scores[rank])

            score_details.append(detail)

        return score_details

    def _display_fast_scores(self, score_details: list, query: str):
        """快速分数显示 - 混合排序结果"""
        if not self.show_detailed_scores:
            return

        # 🔄 混合排序标题
        logger.info(f"🔄 HYBRID RERANK TOP {self.show_top_n_scores} RESULTS:")
        logger.info(f"📊 Weights: Semantic {self.semantic_weight*100:.0f}% + BM25 {self.bm25_weight*100:.0f}%")
        logger.info("-" * 80)

        # 快速排序和显示
        sorted_details = sorted(score_details, key=lambda x: x.combined_score, reverse=True)

        for i, detail in enumerate(sorted_details[: self.show_top_n_scores]):
            logger.info(
                f"#{i+1:2d}: {detail.title}: "
                f"BM25={detail.bm25_raw:.3f}, "
                f"Semantic={detail.semantic_raw:.3f}, "
                f"Hybrid={detail.combined_score:.3f}"
            )

    def _cpu_optimized_position_multiplier(self, score: float) -> float:
        """简化的位置倍数计算"""
        if score > 0.85:
            return 0.05
        elif score > 0.7:
            return 0.15
        elif score > 0.5:
            return 0.3
        else:
            return 0.8

    def _apply_hybrid_rerank(self, results: list, documents: list, normalized_scores: list):
        """应用混合重排序"""
        for idx, doc_index in enumerate(documents[0][: len(results)]):
            if doc_index >= len(results):
                continue

            score = float(normalized_scores[idx])
            multiplier = self._cpu_optimized_position_multiplier(score)
            result = results[doc_index]

            if hasattr(result, 'positions') and result.positions:
                result.positions[0] = float(max(0.01, result.positions[0] * multiplier))
            else:
                result.positions = [float((doc_index + 1.0) / len(results) * multiplier)]

    def _process_hybrid_results(self, results: list, query: str) -> None:
        """混合重排序的结果处理主流程"""
        if len(results) < 2:
            return

        # 严格限制处理数量
        if len(results) > self.max_results_limit:
            results = results[: self.max_results_limit]
            logger.warning(f"⚠️ Limited to {self.max_results_limit} results for CPU performance")

        start_time = time.time()

        # 快速构建语料库
        corpus = []
        for result in results:
            text_parts = []
            if hasattr(result, 'title') and result.title:
                text_parts.append(result.title)
            if hasattr(result, 'content') and result.content:
                text_parts.append(result.content[:100])  # 限制content长度
            corpus.append(" ".join(text_parts))

        # 获取停用词（缓存）
        if not hasattr(self, '_cached_stopwords'):
            self._cached_stopwords = {
                word
                for name, value in stopwords_module.__dict__.items()
                if name.startswith("STOPWORDS_") and isinstance(value, tuple)
                for word in value
            }
        stopwords = self._cached_stopwords

        # BM25处理
        corpus_tokens = bm25s.tokenize(corpus, stopwords=stopwords)
        query_tokens = bm25s.tokenize(query, stopwords=stopwords)

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        documents, bm25_scores = retriever.retrieve(query_tokens, k=len(corpus), return_as="tuple", show_progress=False)

        bm25_time = time.time() - start_time

        # 语义处理
        semantic_start = time.time()
        semantic_scores = self._calculate_cpu_optimized_semantic_scores(query, corpus)
        semantic_time = time.time() - semantic_start

        if bm25_scores is not None and len(bm25_scores) > 0:
            # 混合分数组合
            combined_scores = self._hybrid_score_combination(bm25_scores[0], semantic_scores)
            normalized_scores = self._fast_normalize_scores(combined_scores).tolist()

            if normalized_scores:
                # 应用混合重排序
                self._apply_hybrid_rerank(results, documents, normalized_scores)

                # 如果启用，显示分数
                if self.show_detailed_scores:
                    score_details = self._create_fast_score_details(
                        results, documents, bm25_scores[0], semantic_scores, combined_scores
                    )
                    self._display_fast_scores(score_details, query)

                total_time = time.time() - start_time

                logger.info(
                    f"🔄 Hybrid rerank processing: {total_time:.2f}s total "
                    f"(BM25: {bm25_time:.2f}s, Semantic: {semantic_time:.2f}s)"
                )

                self._performance_stats["cpu_optimized_calls"] += 1
                if self.enable_fast_mode:
                    self._performance_stats["fast_mode_calls"] += 1

    def post_search(self, request: "SXNG_REQUEST", search: "SearchWithPlugins") -> EngineResults:
        results = search.result_container.get_ordered_results()

        if len(results) < 2:
            return search.result_container

        query = search.search_query.query

        logger.info(f"🔄 Hybrid rerank processing: {len(results)} results")

        self._process_hybrid_results(results, query)

        # 简化性能统计
        self._performance_stats["total_calls"] += 1
        if self._performance_stats["total_calls"] % 10 == 0:
            hit_rate = self._semantic_cache.get_hit_rate()
            logger.info(
                f"📊 Hybrid rerank stats: {self._performance_stats['total_calls']} calls, "
                f"cache hit rate: {hit_rate:.1f}%"
            )

        return search.result_container

    def clear_cache(self) -> None:
        """清理缓存"""
        self._semantic_cache.cache.clear()
        self._semantic_cache.access_times.clear()
        logger.info("🧹 Hybrid rerank cache cleared")

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return self.model_manager.get_model_info()
