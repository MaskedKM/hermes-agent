"""Deduplicator — Jaccard similarity-based memory deduplication.

Uses jieba for Chinese tokenization and Jaccard coefficient for similarity.
Threshold 0.95: only merges near-identical entries (fixes M5).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Lazy jieba import
_jieba = None


def _get_jieba():
    global _jieba
    if _jieba is not None:
        return _jieba
    try:
        import jieba as _jb
        _jieba = _jb
        _jb.setLogLevel(logging.WARNING)
    except ImportError:
        pass
    return _jieba


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a set of tokens (Chinese + English)."""
    jb = _get_jieba()
    tokens = set()

    # Extract Chinese segments and English/code words
    for seg in re.split(r"([\u4e00-\u9fff]+)", text):
        if re.match(r"[\u4e00-\u9fff]+", seg):
            if jb:
                for w in jb.cut(seg):
                    if len(w.strip()) >= 2:
                        tokens.add(w.strip().lower())
        elif seg.strip():
            # English/code: split on non-alphanumeric
            for w in re.findall(r"[a-zA-Z0-9_]{2,}", seg):
                tokens.add(w.lower())

    return tokens


@dataclass
class DedupGroup:
    """A group of drawers identified as near-duplicates."""
    representative_id: str
    representative_content: str
    member_ids: List[str] = field(default_factory=list)
    max_similarity: float = 0.0


class Deduplicator:
    """Jaccard similarity-based memory deduplicator.

    Threshold 0.95: only merges near-identical entries. At this threshold,
    semantic differences are negligible — we keep the longest/newest version.
    """

    DEFAULT_SIMILARITY_THRESHOLD = 0.95

    def dedupe(self, items: List[Tuple[str, str]],
               threshold: float = 0.95) -> List[DedupGroup]:
        """Deduplicate a list of (drawer_id, content) tuples.

        O(n²) Jaccard comparison (n typically < 200, acceptable).

        Returns list of DedupGroup, one per group of duplicates found.
        """
        if not items:
            return []

        # Pre-tokenize all items
        tokenized = [(did, content, _tokenize(content)) for did, content in items]
        n = len(tokenized)
        visited = set()
        groups = []

        for i in range(n):
            if i in visited:
                continue
            did_i, content_i, tokens_i = tokenized[i]
            if not tokens_i:
                continue

            # Find all items similar to item i
            group_members = [(did_i, content_i)]
            max_sim = 0.0

            for j in range(i + 1, n):
                if j in visited:
                    continue
                did_j, content_j, tokens_j = tokenized[j]
                if not tokens_j:
                    continue

                sim = self._jaccard(tokens_i, tokens_j)
                if sim >= threshold:
                    group_members.append((did_j, content_j))
                    visited.add(j)
                    max_sim = max(max_sim, sim)

            if len(group_members) > 1:
                # Select representative (longest content, then newest by position)
                rep_id, rep_content = self._select_representative(group_members)
                group = DedupGroup(
                    representative_id=rep_id,
                    representative_content=rep_content,
                    member_ids=[m[0] for m in group_members],
                    max_similarity=max_sim,
                )
                groups.append(group)

        logger.info("Deduplicator: found %d groups (threshold=%.2f, items=%d)",
                     len(groups), threshold, n)
        return groups

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        """Compute Jaccard similarity coefficient."""
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _select_representative(members: List[Tuple[str, str]]) -> Tuple[str, str]:
        """Select representative: longest content wins, ties broken by position (newest)."""
        return max(members, key=lambda m: (len(m[1]), members.index(m)))
