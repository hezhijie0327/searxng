# SPDX-License-Identifier: AGPL-3.0-or-later
"""This module implements functions needed for the autocompleter."""
# pylint: disable=use-dict-literal,too-many-locals

import json
import html
import re
import typing as t
from urllib.parse import urlencode, quote_plus

import numpy as np
import bm25s
import tiktoken

import lxml.etree
import lxml.html
from httpx import HTTPError

from searx import settings
from searx.engines import (
    engines,
    google,
)
from searx.network import get as http_get, post as http_post  # pyright: ignore[reportUnknownVariableType]
from searx.exceptions import SearxEngineResponseException
from searx.utils import extr, gen_useragent

if t.TYPE_CHECKING:
    from searx.extended_types import SXNG_Response

# Tiktoken 配置常量
PRIMARY_MODEL = "gpt-oss-120b"
FALLBACK_ENCODING = "o200k_harmony"

# 初始化 tiktoken 编码器
try:
    _tokenizer = tiktoken.encoding_for_model(PRIMARY_MODEL)
except (KeyError, ImportError, ValueError) as e:
    _tokenizer = tiktoken.get_encoding(FALLBACK_ENCODING)


def _preprocess_text(text: str) -> str:
    """预处理文本：统一大小写、清理空白符、移除特殊字符"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # 合并多个空白符
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)  # 保留字母数字中文，其他替换为空格
    return text.strip()


def _tokenize_for_bm25(text: str) -> list[str]:
    """使用 tiktoken 对文本进行分词，为 BM25 算法准备 token 列表"""
    if not text:
        return []

    preprocessed_text = _preprocess_text(text)
    if not preprocessed_text:
        return []

    try:
        # 使用 tiktoken 编码后解码，获得高质量的 token
        tokens = _tokenizer.encode(preprocessed_text)
        token_strings = []
        for token in tokens:
            try:
                token_str = _tokenizer.decode([token])
                if token_str.strip():
                    token_strings.append(token_str.strip())
            except (UnicodeDecodeError, ValueError):
                continue

        # 如果 tiktoken 分词失败，回退到简单空格分词
        if not token_strings:
            token_strings = preprocessed_text.split()

        return token_strings

    except (ValueError, TypeError, AttributeError):
        # 异常情况下使用简单分词
        return preprocessed_text.split()


def update_kwargs(**kwargs) -> None:  # type: ignore
    if 'timeout' not in kwargs:
        kwargs['timeout'] = settings['outgoing']['request_timeout']
    kwargs['raise_for_httperror'] = True


def get(*args, **kwargs) -> "SXNG_Response":  # type: ignore
    update_kwargs(**kwargs)  # pyright: ignore[reportUnknownArgumentType]
    return http_get(*args, **kwargs)  # pyright: ignore[reportUnknownArgumentType]


def post(*args, **kwargs) -> "SXNG_Response":  # type: ignore
    update_kwargs(**kwargs)  # pyright: ignore[reportUnknownArgumentType]
    return http_post(*args, **kwargs)  # pyright: ignore[reportUnknownArgumentType]


def baidu(query: str, _sxng_locale: str) -> list[str]:
    # baidu search autocompleter
    base_url = "https://www.baidu.com/sugrec?"
    response = get(base_url + urlencode({'ie': 'utf-8', 'json': 1, 'prod': 'pc', 'wd': query}))
    results: list[str] = []

    if response.ok:
        data: dict[str, t.Any] = response.json()
        if 'g' in data:
            for item in data['g']:
                results.append(item['q'])
    return results


def brave(query: str, _sxng_locale: str) -> list[str]:
    # brave search autocompleter
    url = 'https://search.brave.com/api/suggest?'
    url += urlencode({'q': query})
    country = 'all'
    kwargs = {'cookies': {'country': country}}
    resp = get(url, **kwargs)
    results: list[str] = []

    if resp.ok:
        data: list[list[str]] = resp.json()
        for item in data[1]:
            results.append(item)
    return results


def dbpedia(query: str, _sxng_locale: str) -> list[str]:
    autocomplete_url = 'https://lookup.dbpedia.org/api/search.asmx/KeywordSearch?'
    resp = get(autocomplete_url + urlencode(dict(QueryString=query)))
    results: list[str] = []

    if resp.ok:
        dom = lxml.etree.fromstring(resp.content)
        results = [str(x) for x in dom.xpath('//Result/Label//text()')]

    return results


def duckduckgo(query: str, sxng_locale: str) -> list[str]:
    """Autocomplete from DuckDuckGo. Supports DuckDuckGo's languages"""

    traits = engines['duckduckgo'].traits
    args: dict[str, str] = {
        'q': query,
        'kl': traits.get_region(sxng_locale, traits.all_locale),
    }

    url = 'https://duckduckgo.com/ac/?type=list&' + urlencode(args)
    resp = get(url)
    results: list[str] = []

    if resp.ok:
        j = resp.json()
        if len(j) > 1:
            results = j[1]
    return results


def google_complete(query: str, sxng_locale: str) -> list[str]:
    """Autocomplete from Google.  Supports Google's languages and subdomains
    (:py:obj:`searx.engines.google.get_google_info`) by using the async REST
    API::

        https://{subdomain}/complete/search?{args}

    """

    google_info: dict[str, t.Any] = google.get_google_info({'searxng_locale': sxng_locale}, engines['google'].traits)
    url = 'https://{subdomain}/complete/search?{args}'
    args = urlencode(
        {
            'q': query,
            'client': 'gws-wiz',
            'hl': google_info['params']['hl'],
        }
    )
    results: list[str] = []

    resp = get(url.format(subdomain=google_info['subdomain'], args=args))
    if resp and resp.ok:
        json_txt = resp.text[resp.text.find('[') : resp.text.find(']', -3) + 1]
        data = json.loads(json_txt)
        for item in data[0]:
            results.append(lxml.html.fromstring(item[0]).text_content())
    return results


def mwmbl(query: str, _sxng_locale: str) -> list[str]:
    """Autocomplete from Mwmbl_."""

    # mwmbl autocompleter
    url = 'https://api.mwmbl.org/search/complete?{query}'

    results: list[str] = get(url.format(query=urlencode({'q': query}))).json()[1]

    # results starting with `go:` are direct urls and not useful for auto completion
    return [result for result in results if not result.startswith("go: ") and not result.startswith("search: ")]


def naver(query: str, _sxng_locale: str) -> list[str]:
    # Naver search autocompleter
    url = f"https://ac.search.naver.com/nx/ac?{urlencode({'q': query, 'r_format': 'json', 'st': 0})}"
    response = get(url)
    results: list[str] = []

    if response.ok:
        data: dict[str, t.Any] = response.json()
        if data.get('items'):
            for item in data['items'][0]:
                results.append(item[0])
    return results


def qihu360search(query: str, _sxng_locale: str) -> list[str]:
    # 360Search search autocompleter
    url = f"https://sug.so.360.cn/suggest?{urlencode({'format': 'json', 'word': query})}"
    response = get(url)
    results: list[str] = []

    if response.ok:
        data: dict[str, t.Any] = response.json()
        if 'result' in data:
            for item in data['result']:
                results.append(item['word'])
    return results


def quark(query: str, _sxng_locale: str) -> list[str]:
    # Quark search autocompleter
    url = f"https://sugs.m.sm.cn/web?{urlencode({'q': query})}"
    response = get(url)
    results: list[str] = []

    if response.ok:
        data = response.json()
        for item in data.get('r', []):
            results.append(item['w'])
    return results


def seznam(query: str, _sxng_locale: str) -> list[str]:
    # seznam search autocompleter
    url = 'https://suggest.seznam.cz/fulltext/cs?{query}'
    resp = get(
        url.format(
            query=urlencode(
                {'phrase': query, 'cursorPosition': len(query), 'format': 'json-2', 'highlight': '1', 'count': '6'}
            )
        )
    )
    results: list[str] = []

    if resp.ok:
        data = resp.json()
        results = [
            ''.join([part.get('text', '') for part in item.get('text', [])])
            for item in data.get('result', [])
            if item.get('itemType', None) == 'ItemType.TEXT'
        ]
    return results


def sogou(query: str, _sxng_locale: str) -> list[str]:
    # Sogou search autocompleter
    base_url = "https://sor.html5.qq.com/api/getsug?"
    resp = get(base_url + urlencode({'m': 'searxng', 'key': query}))
    results: list[str] = []

    if resp.ok:
        raw_json = extr(resp.text, "[", "]", default="")
        try:
            data = json.loads(f"[{raw_json}]]")
            results = data[1]
        except json.JSONDecodeError:
            pass
    return results


def startpage(query: str, sxng_locale: str) -> list[str]:
    """Autocomplete from Startpage's Firefox extension.
    Supports the languages specified in lang_map.
    """

    lang_map = {
        'da': 'dansk',
        'de': 'deutsch',
        'en': 'english',
        'es': 'espanol',
        'fr': 'francais',
        'nb': 'norsk',
        'nl': 'nederlands',
        'pl': 'polski',
        'pt': 'portugues',
        'sv': 'svenska',
    }

    base_lang = sxng_locale.split('-')[0]
    lui = lang_map.get(base_lang, 'english')

    url_params = {
        'q': query,
        'format': 'opensearch',
        'segment': 'startpage.defaultffx',
        'lui': lui,
    }
    url = f'https://www.startpage.com/suggestions?{urlencode(url_params)}'

    # Needs user agent, returns a 204 otherwise
    h = {'User-Agent': gen_useragent()}

    resp = get(url, headers=h)
    results: list[str] = []

    if resp.ok:
        try:
            data = resp.json()
            if len(data) >= 2 and isinstance(data[1], list):
                results = data[1]
        except json.JSONDecodeError:
            pass

    return results


def stract(query: str, _sxng_locale: str) -> list[str]:
    # stract autocompleter (beta)
    url = f"https://stract.com/beta/api/autosuggest?q={quote_plus(query)}"
    resp = post(url)
    results: list[str] = []

    if resp.ok:
        results = [html.unescape(suggestion['raw']) for suggestion in resp.json()]

    return results


def swisscows(query: str, _sxng_locale: str) -> list[str]:
    # swisscows autocompleter
    url = 'https://swisscows.ch/api/suggest?{query}&itemsCount=5'
    results: list[str] = json.loads(get(url.format(query=urlencode({'query': query}))).text)
    return results


def qwant(query: str, sxng_locale: str) -> list[str]:
    """Autocomplete from Qwant. Supports Qwant's regions."""
    locale = engines['qwant'].traits.get_region(sxng_locale, 'en_US')
    url = 'https://api.qwant.com/v3/suggest?{query}'
    resp = get(url.format(query=urlencode({'q': query, 'locale': locale, 'version': '2'})))
    results: list[str] = []

    if resp.ok:
        data = resp.json()
        if data['status'] == 'success':
            for item in data['data']['items']:
                results.append(item['value'])

    return results


def wikipedia(query: str, sxng_locale: str) -> list[str]:
    """Autocomplete from Wikipedia. Supports Wikipedia's languages (aka netloc)."""
    eng_traits = engines['wikipedia'].traits
    wiki_lang = eng_traits.get_language(sxng_locale, 'en')
    wiki_netloc: str = eng_traits.custom['wiki_netloc'].get(wiki_lang, 'en.wikipedia.org')  # type: ignore

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
    resp = get(f'https://{wiki_netloc}/w/api.php?{args}')
    results: list[str] = []

    if resp.ok:
        data = resp.json()
        if len(data) > 1:
            results = data[1]

    return results


def yandex(query: str, _sxng_locale: str) -> list[str]:
    # yandex autocompleter
    url = "https://suggest.yandex.com/suggest-ff.cgi?{0}"
    resp = json.loads(get(url.format(urlencode(dict(part=query)))).text)
    results: list[str] = []

    if len(resp) > 1:
        results = resp[1]
    return results


backends: dict[str, t.Callable[[str, str], list[str]]] = {
    '360search': qihu360search,
    'baidu': baidu,
    'brave': brave,
    'dbpedia': dbpedia,
    'duckduckgo': duckduckgo,
    'google': google_complete,
    'mwmbl': mwmbl,
    'naver': naver,
    'quark': quark,
    'qwant': qwant,
    'seznam': seznam,
    'sogou': sogou,
    'startpage': startpage,
    'stract': stract,
    'swisscows': swisscows,
    'wikipedia': wikipedia,
    'yandex': yandex,
    'custom': 'custom',
}


def deduplicate_results(results):
    """去除重复的自动补全结果，保持原有顺序"""
    seen = set()
    unique_results = []
    for result in results:
        if result not in seen:
            unique_results.append(result)
            seen.add(result)
    return unique_results


def _get_strategy_params(query_length):
    """根据查询长度选择自动补全策略参数"""
    if query_length <= 2:
        # 极短查询：强调前缀匹配，降低 BM25 权重
        return {
            'prefix_bonus': 2.0,
            'length_penalty_rate': 0.1,
            'exact_match_bonus': 3.0,
            'bm25_weight': 0.3,
        }
    # 修复：移除不必要的 elif，改为 if
    if query_length <= 5:
        # 中等查询：平衡前缀匹配和语义相关性
        return {
            'prefix_bonus': 1.5,
            'length_penalty_rate': 0.05,
            'exact_match_bonus': 2.0,
            'bm25_weight': 0.6,
        }

    # 长查询：依赖 BM25 语义理解能力
    return {
        'prefix_bonus': 1.2,
        'length_penalty_rate': 0.02,
        'exact_match_bonus': 1.5,
        'bm25_weight': 0.8,
    }


def _normalize_bm25_scores(raw_scores):
    """标准化 BM25 分数到 [0,1] 区间"""
    if len(raw_scores) == 0:
        return []

    min_score, max_score = float(np.min(raw_scores)), float(np.max(raw_scores))
    if max_score > min_score:
        return ((raw_scores - min_score) / (max_score - min_score)).tolist()

    return [0.5] * len(raw_scores)


def _calculate_suggestion_score(suggestion, query_lower, bm25_score, params):
    """计算单个建议的综合分数"""
    suggestion_lower = suggestion.lower()

    # 前缀匹配加分：自动补全的核心特性
    prefix_boost = params['prefix_bonus'] if suggestion_lower.startswith(query_lower) else 1.0

    # 完全匹配加分：用户可能已经知道想要的结果
    exact_match_boost = params['exact_match_bonus'] if suggestion_lower == query_lower else 1.0

    # 长度惩罚：自动补全偏好简洁的建议
    length_penalty = 1.0
    if len(suggestion) > len(query_lower) * 3:
        excess_length = len(suggestion) - len(query_lower) * 2
        length_penalty = 1.0 - (excess_length * params['length_penalty_rate'])
        length_penalty = max(0.1, length_penalty)  # 避免过度惩罚

    # 综合计算最终分数
    final_score = (
        (bm25_score * params['bm25_weight'] + (1.0 - params['bm25_weight']))
        * prefix_boost
        * exact_match_boost
        * length_penalty
    )

    # 添加微小随机因子避免相同分数的不稳定排序
    final_score += hash(suggestion) % 1000 * 0.000001

    return final_score


def rerank_results(results_list, query):
    """使用 BM25 算法和 tiktoken 分词对自动补全结果进行重排

    结合 BM25 语义相关性和自动补全特有的前缀匹配、长度偏好等特性，
    根据查询长度动态调整各评分因子的权重。
    """
    # 合并并去重所有后端返回的结果
    corpus = deduplicate_results([result for results in results_list for result in results])

    if len(corpus) < 2:
        return corpus

    try:
        # 使用 tiktoken 进行高质量分词，支持中英文混合
        corpus_tokens = [_tokenize_for_bm25(doc) for doc in corpus]
        query_tokens = _tokenize_for_bm25(query)

        if not query_tokens:
            return corpus

        # 构建 BM25 索引并检索
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        documents, scores = retriever.retrieve([query_tokens], k=len(corpus), return_as='tuple', show_progress=False)

        # 标准化 BM25 分数到 [0,1] 区间
        raw_scores = scores[0]
        normalized_scores = _normalize_bm25_scores(raw_scores)

        if not normalized_scores:
            return corpus

        # 获取策略参数
        query_length = len(query.strip())
        params = _get_strategy_params(query_length)

        # 计算每个建议的综合分数
        final_scores = []
        query_lower = query.lower()

        for idx, doc_index in enumerate(documents[0]):
            if doc_index >= len(corpus):
                continue

            suggestion = corpus[doc_index]
            bm25_score = float(normalized_scores[idx])

            final_score = _calculate_suggestion_score(suggestion, query_lower, bm25_score, params)
            final_scores.append((doc_index, final_score))

        # 按最终分数降序排列
        final_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回重排后的建议列表
        return [corpus[doc_index] for doc_index, _ in final_scores]

    except (ValueError, TypeError, AttributeError, ImportError):
        # 异常情况下返回原始结果
        return corpus


def search_autocomplete(backend_name: str, query: str, sxng_locale: str) -> list[str]:
    if backend_name == 'custom':
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

    backend = backends.get(backend_name)
    if backend is None:
        return []

    # 修复：移除不必要的 else，直接执行代码
    try:
        return backend(query, sxng_locale)
    except (HTTPError, SearxEngineResponseException, ValueError):
        return []
