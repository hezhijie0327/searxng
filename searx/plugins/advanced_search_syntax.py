# SPDX-License-Identifier: AGPL-3.0-or-later
"""Advanced search syntax plugin for SearXNG.
Supports site filtering, exact phrase matching, &&/|| operations, word exclusion,
positional search, wildcard matching, and AI-powered query rewriting."""

import typing
import re
import asyncio
import logging
from urllib.parse import urlparse
from flask_babel import gettext
from werkzeug.datastructures import ImmutableMultiDict
from searx.extended_types import SXNG_Request
from searx.plugins import Plugin, PluginInfo
from searx.result_types import Result

# OpenAI SDK import
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

if typing.TYPE_CHECKING:
    from searx.plugins import PluginCfg
    from searx.search import SearchWithPlugins

# 设置日志
logger = logging.getLogger(__name__)


class SXNGPlugin(Plugin):
    """Plugin that enhances search with advanced syntax support and AI query rewriting."""

    id = "advanced_search_syntax"

    def __init__(self, plg_cfg: "PluginCfg") -> None:
        super().__init__(plg_cfg)
        self.info = PluginInfo(
            id=self.id,
            name=gettext("Advanced Search Syntax with AI"),
            description=gettext(
                "Enhanced search with site filtering, exact phrases, &&/|| operations, "
                "word exclusion, positional search, wildcards, and AI-powered query rewriting"
            ),
            preference_section="general",
        )

        # OpenAI 配置
        self.openai_config = {
            'base_url': plg_cfg.get('openai_base_url', 'https://api.openai.com/v1'),
            'api_key': plg_cfg.get('openai_api_key', ''),
            'model_name': plg_cfg.get('openai_model', 'gpt-3.5-turbo'),
            'max_tokens': plg_cfg.get('openai_max_tokens', 150),
            'temperature': plg_cfg.get('openai_temperature', 0.7),
            'timeout': plg_cfg.get('openai_timeout', 10),
            'enabled': plg_cfg.get('openai_enabled', True) and OPENAI_AVAILABLE
        }

        # 初始化 OpenAI 客户端
        self.openai_client = None
        if self.openai_config['enabled'] and self.openai_config['api_key']:
            try:
                self.openai_client = OpenAI(
                    base_url=self.openai_config['base_url'],
                    api_key=self.openai_config['api_key'],
                    timeout=self.openai_config['timeout']
                )
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.openai_config['enabled'] = False

        # Pre-compile regex patterns for better performance
        self._patterns = {
            'ai_rewrite': re.compile(r'@ai\s+(.+)', re.IGNORECASE | re.DOTALL),
            'site_include': re.compile(r'(?:^|\s)site:([^\s]+)', re.IGNORECASE),
            'site_exclude': re.compile(r'(?:^|\s)-site:([^\s]+)', re.IGNORECASE),
            'intitle_phrase': re.compile(r'intitle:"([^"]+)"', re.IGNORECASE),
            'intitle_word': re.compile(r'intitle:([^\s"]+)', re.IGNORECASE),
            'inurl_word': re.compile(r'inurl:([^\s"]+)', re.IGNORECASE),
            'intext_phrase': re.compile(r'intext:"([^"]+)"', re.IGNORECASE),
            'intext_word': re.compile(r'intext:([^\s"]+)', re.IGNORECASE),
            'exclude_phrase': re.compile(r'-"([^"]+)"'),
            'exact_phrase': re.compile(r'"([^"]+)"'),
            'wildcard': re.compile(r'(?:^|\s)(\w+\*)'),
            'or_group': re.compile(r'(\w+(?:\s*\|\|\s*\w+)+)'),
            'and_group': re.compile(r'(\w+(?:\s*\&\&\s*\w+)+)'),
            'exclude_word': re.compile(r'(?:^|\s)-((?!site:)[^\s"]+)'),
        }

    def _has_advanced_syntax(self, query: str) -> bool:
        """Check if the query contains any advanced search syntax."""
        # Quick check using any of our compiled patterns
        return any(pattern.search(query) for pattern in self._patterns.values())

    def _rewrite_query_with_ai(self, query: str) -> str:
        """使用 OpenAI API 重写查询语句，优化搜索效果。"""
        if not self.openai_config['enabled'] or not self.openai_client:
            logger.warning("OpenAI client not available for query rewriting")
            return query

        try:
            # 构建 prompt
            system_prompt = """You are a search query optimizer. Your task is to rewrite user queries to make them more effective for web search engines.

Guidelines:
1. Keep the core intent and meaning of the original query
2. Use more specific and searchable keywords
3. Remove unnecessary words and filler
4. Add relevant synonyms or alternative terms when helpful
5. Structure the query for better search results
6. Keep it concise and focused
7. Respond ONLY with the optimized query, no explanations

Examples:
Input: "how to fix my computer that won't start"
Output: "computer won't boot troubleshooting startup repair"

Input: "best restaurants near me for date night"
Output: "romantic restaurants date night dining"
"""

            user_prompt = f"Optimize this search query: {query}"

            # 调用 OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.openai_config['model_name'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.openai_config['max_tokens'],
                temperature=self.openai_config['temperature'],
                timeout=self.openai_config['timeout']
            )

            rewritten_query = response.choices[0].message.content.strip()

            # 基本验证重写结果
            if rewritten_query and len(rewritten_query) > 0 and len(rewritten_query) <= 500:
                logger.info(f"Query rewritten: '{query}' -> '{rewritten_query}'")
                return rewritten_query
            else:
                logger.warning(f"Invalid rewritten query, using original: {query}")
                return query

        except Exception as e:
            logger.error(f"Error rewriting query with AI: {e}")
            return query

    def _clean_query_for_engines(self, query: str) -> str:
        """Clean query by removing ALL advanced syntax patterns for external engines."""
        cleaned = query

        # Remove all advanced syntax patterns in order
        for pattern in self._patterns.values():
            cleaned = pattern.sub(' ', cleaned)

        # Additional cleanup for any remaining artifacts
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Remove empty quotes and other artifacts
        cleaned = re.sub(r'""', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    def _parse_advanced_syntax(self, query: str) -> tuple[str, dict]:
        """Parse and extract all advanced syntax patterns from query.

        Returns:
            tuple: (cleaned_query, syntax_dict)
        """
        syntax = {
            'ai_rewrite': False,
            'ai_query': '',
            'site_include': [],
            'site_exclude': [],
            'exact_phrases': [],
            'exclude_phrases': [],
            'or_groups': [],
            'and_groups': [],
            'exclude_words': [],
            'wildcard_terms': [],
            'intitle_words': [],
            'intitle_phrases': [],
            'inurl_words': [],
            'intext_words': [],
            'intext_phrases': [],
            'remaining_terms': [],
        }

        # 1. 首先检查 AI 重写语法
        ai_match = self._patterns['ai_rewrite'].search(query)
        if ai_match:
            syntax['ai_rewrite'] = True
            syntax['ai_query'] = ai_match.group(1).strip()
            # 从查询中移除 @ai 部分，继续处理其他高级语法
            query_without_ai = self._patterns['ai_rewrite'].sub('', query).strip()

            # 如果有AI重写，先重写查询
            if syntax['ai_query']:
                rewritten_query = self._rewrite_query_with_ai(syntax['ai_query'])
                # 将重写的查询与剩余的高级语法结合
                query = f"{rewritten_query} {query_without_ai}".strip()

        # 2. 继续处理其他高级语法模式

        # Site patterns
        syntax['site_include'] = [m.lower().strip() for m in self._patterns['site_include'].findall(query)]
        syntax['site_exclude'] = [m.lower().strip() for m in self._patterns['site_exclude'].findall(query)]

        # Positional patterns (phrases first, then words)
        syntax['intitle_phrases'] = [m.strip() for m in self._patterns['intitle_phrase'].findall(query)]
        intitle_words = self._patterns['intitle_word'].findall(query)
        syntax['intitle_words'] = [
            w.strip() for w in intitle_words if not any(w in phrase for phrase in syntax['intitle_phrases'])
        ]

        syntax['intext_phrases'] = [m.strip() for m in self._patterns['intext_phrase'].findall(query)]
        intext_words = self._patterns['intext_word'].findall(query)
        syntax['intext_words'] = [
            w.strip() for w in intext_words if not any(w in phrase for phrase in syntax['intext_phrases'])
        ]

        syntax['inurl_words'] = [m.strip() for m in self._patterns['inurl_word'].findall(query)]

        # Phrase patterns (exclude first)
        syntax['exclude_phrases'] = [m.strip() for m in self._patterns['exclude_phrase'].findall(query)]
        syntax['exact_phrases'] = [m.strip() for m in self._patterns['exact_phrase'].findall(query)]

        # Logic patterns
        or_matches = self._patterns['or_group'].findall(query)
        for or_group in or_matches:
            terms = [t.strip() for t in re.split(r'\s*\|\|\s*', or_group) if t.strip()]
            if len(terms) > 1:
                syntax['or_groups'].append(terms)

        and_matches = self._patterns['and_group'].findall(query)
        for and_group in and_matches:
            terms = [t.strip() for t in re.split(r'\s*\&\&\s*', and_group) if t.strip()]
            if len(terms) > 1:
                syntax['and_groups'].append(terms)

        # Wildcard and exclude patterns
        syntax['wildcard_terms'] = [m.strip() for m in self._patterns['wildcard'].findall(query)]
        syntax['exclude_words'] = [m.strip() for m in self._patterns['exclude_word'].findall(query)]

        # Clean query for engines and extract remaining terms
        cleaned_query = self._clean_query_for_engines(query)

        # Get remaining terms from the cleaned query
        remaining_terms = [
            term for term in cleaned_query.split() if term and not term.startswith(('!', ':')) and term != '!!'
        ]
        syntax['remaining_terms'] = remaining_terms

        return cleaned_query, syntax

    def _update_form_query(self, form, new_query: str) -> None:
        """Safely update form query value."""
        try:
            # Method 1: Try direct assignment (for mutable forms)
            if hasattr(form, 'q'):
                form.q = new_query
                return
        except (AttributeError, TypeError):
            pass

        try:
            # Method 2: Try dict-like access (for forms with __setitem__)
            form['q'] = new_query
            return
        except (TypeError, KeyError):
            pass

        # Method 3: If form is immutable, we can't modify it
        # This should be handled at the calling level
        raise TypeError("Form object is immutable and cannot be modified")

    def _create_new_form_with_query(self, original_form, new_query: str):
        """Create a new form object with updated query."""
        # Convert form to dict and update query
        form_dict = dict(original_form.items()) if hasattr(original_form, 'items') else dict(original_form)
        form_dict['q'] = new_query
        return ImmutableMultiDict(form_dict)

    def pre_search(self, request: SXNG_Request, search: "SearchWithPlugins") -> bool:
        """Parse the search query for advanced syntax patterns and modify the query sent to engines."""
        original_query = request.form.get('q', '')

        # Only process queries that contain advanced syntax
        if not self._has_advanced_syntax(original_query):
            request.search_syntax = {'has_advanced_syntax': False, 'original_query': original_query}
            return True

        # Parse advanced syntax (包括 AI 重写)
        cleaned_query, syntax = self._parse_advanced_syntax(original_query)
        syntax.update({'has_advanced_syntax': True, 'original_query': original_query, 'cleaned_query': cleaned_query})

        # Store original query for restoration later
        request.original_query = original_query
        request.search_syntax = syntax

        # Update the query in form
        if hasattr(request, 'form') and 'q' in request.form:
            try:
                self._update_form_query(request.form, cleaned_query or original_query)
            except TypeError:
                # Form is immutable, create new one
                request.form = self._create_new_form_with_query(request.form, cleaned_query or original_query)

        # Also update any other query attributes that might exist
        if hasattr(request, 'args') and 'q' in request.args:
            try:
                self._update_form_query(request.args, cleaned_query or original_query)
            except TypeError:
                # Args might be immutable too, but we can't easily replace it
                pass

        # Force update the query parameter for search engines
        if hasattr(search, 'search_query'):
            search.search_query.query = cleaned_query or original_query

        return True

    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain by removing protocol and path parts."""
        if not domain:
            return ""
        domain = re.sub(r'^https?://', '', domain).split('/')[0].split(':')[0]
        return domain.lower().strip()

    def _domain_matches(self, result_domain: str, target_domain: str) -> bool:
        """Check if result domain matches target domain (exact or subdomain)."""
        result_domain = self._normalize_domain(result_domain)
        target_domain = self._normalize_domain(target_domain)
        if not result_domain or not target_domain:
            return False
        return result_domain == target_domain or result_domain.endswith('.' + target_domain)

    def _check_site_filters(self, result_url: str, syntax: dict) -> bool:
        """Check if result passes site include/exclude filters."""
        if not result_url:
            return not syntax['site_include']

        try:
            result_domain = urlparse(result_url).hostname
            if not result_domain:
                return not syntax['site_include']

            # Check exclude filters first
            if syntax['site_exclude'] and any(self._domain_matches(result_domain, d) for d in syntax['site_exclude']):
                return False

            # Check include filters
            if not syntax['site_include']:
                return True
            return any(self._domain_matches(result_domain, d) for d in syntax['site_include'])

        except (ValueError, AttributeError):
            return not syntax['site_include']

    def _text_matches(self, text: str, word: str, exact: bool = False) -> bool:
        """Check if text contains word/phrase."""
        if not text or not word:
            return False

        text_lower = text.lower()
        word_lower = word.lower()

        if exact:  # Exact phrase match
            return word_lower in text_lower

        # Whole word match
        return bool(re.search(r'\b' + re.escape(word_lower) + r'\b', text_lower))

    def _text_matches_wildcard(self, text: str, wildcard_term: str) -> bool:
        """Check if text matches wildcard pattern."""
        if not text or not wildcard_term:
            return False

        prefix = wildcard_term.rstrip('*').lower()
        if not prefix:
            return False

        pattern = r'\b' + re.escape(prefix) + r'\w*'
        return bool(re.search(pattern, text.lower()))

    def _check_positional_filters(self, title: str, content: str, url: str, syntax: dict) -> bool:
        """Check positional filters (intitle, intext, inurl)."""
        # Check intitle filters
        for word in syntax['intitle_words']:
            if not self._text_matches(title, word):
                return False
        for phrase in syntax['intitle_phrases']:
            if not self._text_matches(title, phrase, exact=True):
                return False

        # Check inurl filters
        for word in syntax['inurl_words']:
            if not self._text_matches(url, word):
                return False

        # Check intext filters
        for word in syntax['intext_words']:
            if not self._text_matches(content, word):
                return False
        for phrase in syntax['intext_phrases']:
            if not self._text_matches(content, phrase, exact=True):
                return False

        return True

    def _check_exclusion_filters(self, search_text: str, syntax: dict) -> bool:
        """Check exclusion filters (excluded phrases and words)."""
        # Check excluded phrases
        for phrase in syntax['exclude_phrases']:
            if self._text_matches(search_text, phrase, exact=True):
                return False

        # Check excluded words
        for word in syntax['exclude_words']:
            if self._text_matches(search_text, word):
                return False

        return True

    def _check_inclusion_filters(self, search_text: str, syntax: dict) -> bool:
        """Check inclusion filters (exact phrases, OR/AND groups, wildcards, remaining terms)."""
        # Check exact phrases
        for phrase in syntax['exact_phrases']:
            if not self._text_matches(search_text, phrase, exact=True):
                return False

        # Check OR groups
        for or_group in syntax['or_groups']:
            if not any(self._text_matches(search_text, word) for word in or_group):
                return False

        # Check AND groups
        for and_group in syntax['and_groups']:
            if not all(self._text_matches(search_text, word) for word in and_group):
                return False

        # Check wildcard terms
        for wildcard in syntax['wildcard_terms']:
            if not self._text_matches_wildcard(search_text, wildcard):
                return False

        # Check remaining terms
        for term in syntax['remaining_terms']:
            if not self._text_matches(search_text, term):
                return False

        return True

    def _apply_filters(self, result: Result, syntax: dict) -> bool:
        """Apply all filters to a result."""
        # Get result text components
        title = getattr(result, 'title', '') or ''
        content = getattr(result, 'content', '') or ''
        url = getattr(result, 'url', '') or ''
        search_text = f"{title} {content}".strip()

        # Apply filters in order: site -> positional -> exclusion -> inclusion
        filter_checks = [
            lambda: (
                self._check_site_filters(url, syntax) if (syntax['site_include'] or syntax['site_exclude']) else True
            ),
            lambda: self._check_positional_filters(title, content, url, syntax),
            lambda: self._check_exclusion_filters(search_text, syntax),
            lambda: self._check_inclusion_filters(search_text, syntax),
        ]

        return all(check() for check in filter_checks)

    def on_result(self, request: SXNG_Request, search: "SearchWithPlugins", result: Result) -> bool:
        """Filter results based on advanced search syntax."""
        if not hasattr(request, 'search_syntax'):
            return True

        syntax = request.search_syntax
        if not syntax.get('has_advanced_syntax', False):
            return True

        return self._apply_filters(result, syntax)

    def post_search(self, request: SXNG_Request, search: "SearchWithPlugins") -> None:
        """Restore original query for UI display after search completion."""
        if hasattr(request, 'original_query') and hasattr(request, 'form'):
            # Restore original query for UI display
            try:
                self._update_form_query(request.form, request.original_query)
            except TypeError:
                # Form is immutable, create new one
                request.form = self._create_new_form_with_query(request.form, request.original_query)
