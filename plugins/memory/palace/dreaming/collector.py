"""RecallCollector — Signal collection for Dreaming phases.

Collects recall statistics, drawer candidates, and co-occurrence patterns
from the Palace database for Dreaming scoring and promotion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RecallSignal:
    """A single signal collected for Dreaming."""
    source: str  # "session" | "palace" | "recall_event"
    content: str
    timestamp: str
    session_id: Optional[str] = None
    drawer_id: Optional[str] = None
    bm25_rank: float = 0.0


@dataclass
class RecallCandidate:
    """A drawer candidate for Deep Sleep scoring."""
    drawer_id: str
    content: str
    wing: str
    room: str
    importance: float
    confidence: float
    filed_at: str
    memory_type: str
    # Recall statistics
    recall_count: int = 0
    recall_sessions: int = 0
    recall_days: int = 0
    last_recalled_at: Optional[str] = None
    avg_bm25_rank: float = 0.0
    # Link statistics
    link_count: int = 0
    # Source tracking
    source_session_id: Optional[str] = None
    # Phase signals
    light_signals: int = 0
    rem_signals: int = 0


@dataclass
class PatternCandidate:
    """A REM Sleep pattern candidate from co-occurrence analysis."""
    entities: List[str]
    frequency: int
    context_examples: List[str] = field(default_factory=list)
    concept_tags: List[str] = field(default_factory=list)


class RecallCollector:
    """Collects signals from Palace DB for Dreaming phases."""

    def __init__(self, store) -> None:
        self.store = store

    def collect_light_candidates(self, lookback_days: int = 2,
                                  limit: int = 200) -> List[tuple]:
        """Collect recent drawers for Light Sleep deduplication.

        Returns [(drawer_id, content), ...]
        """
        drawers = self.store.get_recent_drawers(
            lookback_days=lookback_days, limit=limit
        )
        return [(d["id"], d["content"]) for d in drawers]

    def collect_deep_candidates(self) -> List[RecallCandidate]:
        """Collect all active drawers with full recall statistics for Deep Sleep."""
        drawers = self.store.get_active_drawers(include_superseded=False, limit=1000)
        candidates = []

        for d in drawers:
            # Get recall stats
            recall_stats = self.store.get_recall_stats(d["id"])

            # Get link count
            link_count = self.store.get_link_count(d["id"])

            # Get dreaming signal counts
            signal_counts = self.store.get_dreaming_signal_counts(d["id"])

            candidate = RecallCandidate(
                drawer_id=d["id"],
                content=d["content"],
                wing=d.get("wing", ""),
                room=d.get("room", ""),
                importance=d.get("importance", 3.0),
                confidence=d.get("confidence", 0.5),
                filed_at=d.get("filed_at", ""),
                memory_type=d.get("memory_type", "general"),
                recall_count=recall_stats["recall_count"],
                recall_sessions=recall_stats["recall_sessions"],
                recall_days=recall_stats["recall_days"],
                last_recalled_at=recall_stats["last_recalled_at"],
                avg_bm25_rank=recall_stats["avg_bm25_rank"],
                link_count=link_count,
                source_session_id=d.get("source_session_id"),
                light_signals=signal_counts.get("light", 0),
                rem_signals=signal_counts.get("rem", 0),
            )
            candidates.append(candidate)

        logger.info("RecallCollector: %d deep candidates collected", len(candidates))
        return candidates

    def collect_rem_patterns(self, min_groups: int = 10,
                              min_frequency: int = 3) -> List[PatternCandidate]:
        """Collect co-occurrence patterns for REM Sleep analysis.

        Returns empty list if data is insufficient (fixes m2).
        """
        # Check total cooccurrence groups
        total = self.store.conn.execute(
            "SELECT COUNT(*) FROM cooccurrence_groups"
        ).fetchone()[0]

        if total < min_groups:
            logger.info("REM: insufficient data (%d groups, need %d), skipping",
                        total, min_groups)
            return []

        # Get high-frequency patterns
        rows = self.store.conn.execute(
            """
            SELECT entities, context, COUNT(*) as frequency
            FROM cooccurrence_groups
            GROUP BY entities
            HAVING frequency >= ?
            ORDER BY frequency DESC
            LIMIT ?
            """,
            (min_frequency, 20),
        ).fetchall()

        patterns = []
        for row in rows:
            import json
            try:
                entities = json.loads(row["entities"])
            except (json.JSONDecodeError, TypeError):
                continue

            patterns.append(PatternCandidate(
                entities=entities,
                frequency=row["frequency"],
                context_examples=[row["context"]] if row.get("context") else [],
            ))

        logger.info("REM: %d patterns collected (from %d groups)",
                     len(patterns), total)
        return patterns
