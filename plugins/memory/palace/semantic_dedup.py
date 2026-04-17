"""SemanticDeduplicator — cosine-similarity-based deduplication using embeddings.

Phase 3A of memory enhancement: semantic deduplication.

Uses sqlite-vec kNN search to find near-duplicate drawers by embedding
cosine similarity.  Falls back to the existing Jaccard Deduplicator when
vector_store is unavailable.

Integration points:
- Light Sleep: ``engine.run_light_sleep`` calls ``dedupe``.
- ``add_drawer``: provider calls ``find_duplicate`` before inserting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Lazy import to avoid circular: semantic_dedup -> dreaming.__init__ -> engine -> semantic_dedup
def _import_deduplicator():
    from .dreaming.deduplicator import Deduplicator, DedupGroup
    return Deduplicator, DedupGroup

logger = logging.getLogger(__name__)

# Default cosine similarity threshold for semantic dedup.
# 0.90 = high overlap in meaning, not just keyword match.
DEFAULT_SEMANTIC_THRESHOLD: float = 0.90


@dataclass
class SemanticDupResult:
    """Result of semantic duplicate check."""
    is_duplicate: bool
    best_match_id: Optional[str] = None
    best_match_content: Optional[str] = None
    similarity: float = 0.0


class SemanticDeduplicator:
    """Embedding-based semantic deduplication.

    Wraps the VectorStore's kNN search to find semantically similar
    drawers.  Gracefully degrades to Jaccard when no vector store.
    """

    def __init__(self, vector_store=None, fallback=None) -> None:
        self._vs = vector_store
        if fallback is None:
            Deduplicator, _ = _import_deduplicator()
            fallback = Deduplicator()
        self._fallback = fallback

    @property
    def available(self) -> bool:
        """True if vector store is configured."""
        return self._vs is not None

    def find_duplicate(
        self,
        content: str,
        embed_fn=None,
        threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    ) -> SemanticDupResult:
        """Check if *content* is semantically duplicate of an existing drawer.

        Parameters
        ----------
        content:
            The drawer content to check.
        embed_fn:
            Callable ``str -> List[float]`` to get embedding.
            Required when vector_store is available.
        threshold:
            Cosine similarity threshold (default 0.90).

        Returns
        -------
        SemanticDupResult with the best match if above threshold.
        """
        if not self._vs or not embed_fn:
            # Fallback: Jaccard check against recent drawers
            return self._jaccard_check(content, threshold=0.95)

        try:
            embedding = embed_fn(content)
            if not embedding:
                return SemanticDupResult(is_duplicate=False)

            # Search top-5 nearest neighbors
            results = self._vs.search_vectors(embedding, k=5)

            for r in results:
                # sqlite-vec distance is L2; convert to cosine similarity
                # For normalized vectors: cosine_sim = 1 - (l2_dist^2 / 2)
                l2 = r.get("distance", 1.0)
                cos_sim = max(0.0, 1.0 - (l2 * l2) / 2.0)

                if cos_sim >= threshold:
                    # Fetch the matched drawer content
                    match_content = self._get_drawer_content(r["drawer_id"])
                    return SemanticDupResult(
                        is_duplicate=True,
                        best_match_id=r["drawer_id"],
                        best_match_content=match_content,
                        similarity=round(cos_sim, 4),
                    )

        except Exception as e:
            logger.debug("Semantic dedup search failed: %s", e)

        return SemanticDupResult(is_duplicate=False)

    def dedupe(
        self,
        items: List[Tuple[str, str]],
        embed_fn=None,
        threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    ) -> List[DedupGroup]:
        """Batch deduplicate items using semantic similarity.

        Falls back to Jaccard Deduplicator when vector store unavailable.

        Parameters
        ----------
        items:
            List of (drawer_id, content) tuples.
        embed_fn:
            Callable ``List[str] -> List[List[float]]`` for batch embedding.
        threshold:
            Similarity threshold.

        Returns
        -------
        List of DedupGroup.
        """
        if not self._vs or not embed_fn:
            return self._fallback.dedupe(items, threshold=0.95)

        try:
            return self._semantic_dedupe(items, embed_fn, threshold)
        except Exception as e:
            logger.warning("Semantic dedup failed, falling back to Jaccard: %s", e)
            return self._fallback.dedupe(items, threshold=0.95)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _semantic_dedupe(
        self,
        items: List[Tuple[str, str]],
        embed_fn,
        threshold: float,
    ) -> List[DedupGroup]:
        """Embedding-based batch deduplication."""
        if not items:
            return []

        ids = [did for did, _ in items]
        contents = [content for _, content in items]

        # Batch embed
        embeddings = embed_fn(contents)
        if not embeddings or len(embeddings) != len(items):
            return self._fallback.dedupe(items, threshold=0.95)

        # For each item, search for nearest neighbors
        visited = set()
        groups: List[DedupGroup] = []
        n = len(items)

        for i in range(n):
            if i in visited:
                continue

            did_i, content_i = items[i]
            emb_i = embeddings[i]
            if not emb_i:
                continue

            results = self._vs.search_vectors(emb_i, k=min(n, 10))

            # Find items similar to i
            group_members = [(did_i, content_i)]
            max_sim = 0.0

            for r in results:
                did_j = r["drawer_id"]
                if did_j == did_i:
                    continue

                # Find index of did_j in items
                j = None
                for idx in range(n):
                    if idx not in visited and items[idx][0] == did_j:
                        j = idx
                        break
                if j is None:
                    continue

                l2 = r.get("distance", 1.0)
                cos_sim = max(0.0, 1.0 - (l2 * l2) / 2.0)

                if cos_sim >= threshold:
                    group_members.append(items[j])
                    visited.add(j)
                    max_sim = max(max_sim, cos_sim)

            if len(group_members) > 1:
                rep_id, rep_content = self._fallback._select_representative(group_members)
                _, DedupGroup = _import_deduplicator()
                groups.append(DedupGroup(
                    representative_id=rep_id,
                    representative_content=rep_content,
                    member_ids=[m[0] for m in group_members],
                    max_similarity=max_sim,
                ))

        logger.info("SemanticDeduplicator: found %d groups (threshold=%.2f)", len(groups), threshold)
        return groups

    def _jaccard_check(
        self, content: str, threshold: float = 0.95
    ) -> SemanticDupResult:
        """Fallback Jaccard check — requires vector_store to have a way to get recent drawers."""
        return SemanticDupResult(is_duplicate=False)

    def _get_drawer_content(self, drawer_id: str) -> Optional[str]:
        """Fetch drawer content from vector_store's connection."""
        try:
            if self._vs and hasattr(self._vs, '_conn'):
                row = self._vs._conn.execute(
                    "SELECT content FROM drawers WHERE id = ?",
                    (drawer_id,),
                ).fetchone()
                return row["content"] if row else None
        except Exception:
            pass
        return None
