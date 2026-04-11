"""DreamingScorer — Dual-mode memory scoring (cold start + steady state).

Implements the scoring formula from the Dreaming plan:
- Cold start mode: relies on intrinsic quality + recency (when recall data is sparse)
- Steady state mode: full multi-dimensional scoring with recall signals
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from plugins.memory.palace.dreaming.defaults import (
    COLD_START_MIN_RECALL_EVENTS,
    COLD_START_MIN_DAYS,
    COLD_START_WEIGHTS,
    STEADY_STATE_WEIGHTS,
    RECENCY_HALF_LIFE_DAYS,
    CONNECTIVITY_MAX_LINKS,
    TIME_CONSOLIDATION_MAX_DAYS,
    MAX_EXPECTED_DRAWERS,
)

logger = logging.getLogger(__name__)


@dataclass
class ScoredCandidate:
    """A drawer with its computed Dreaming score."""
    drawer_id: str
    content: str
    wing: str
    room: str
    importance: float
    confidence: float
    total_score: float
    mode: str  # "cold_start" | "steady_state"
    component_scores: Dict[str, float] = field(default_factory=dict)
    phase_boost: float = 0.0


class DreamingScorer:
    """Dual-mode memory scorer for Dreaming."""

    def __init__(self, store) -> None:
        self.store = store
        self._cached_state: Optional[Dict[str, Any]] = None

    @property
    def state(self) -> Dict[str, Any]:
        """Lazy-load dreaming state."""
        if self._cached_state is None:
            self._cached_state = self.store.get_dreaming_state()
        return self._cached_state

    def is_cold_start(self) -> bool:
        """Check if the system is in cold start mode."""
        total_recalls = self.state["total_recall_count"]
        days = self.state["days_since_first_dreaming_run"]
        return total_recalls < COLD_START_MIN_RECALL_EVENTS or days < COLD_START_MIN_DAYS

    def score(self, candidate) -> ScoredCandidate:
        """Score a RecallCandidate using the appropriate mode."""
        if self.is_cold_start():
            return self._score_cold_start(candidate)
        else:
            return self._score_steady_state(candidate)

    def _score_cold_start(self, c) -> ScoredCandidate:
        """Cold start scoring: intrinsic_quality + recency + connectivity + time."""
        w = COLD_START_WEIGHTS

        # Intrinsic quality = (importance / 5.0) × confidence
        intrinsic = (c.importance / 5.0) * c.confidence

        # Recency: exponential decay
        recency = self._recency_score(c.filed_at)

        # Connectivity: drawer_links density
        connectivity = min(1.0, c.link_count / CONNECTIVITY_MAX_LINKS)

        # Time consolidation: how long the memory has existed
        time_consolidation = self._time_consolidation_score(c.filed_at)

        # Phase boost (from prior dreaming signals)
        phase_boost = self._phase_boost(c.light_signals, c.rem_signals)

        total = (
            w["intrinsic_quality"] * intrinsic
            + w["recency"] * recency
            + w["connectivity"] * connectivity
            + w["time_consolidation"] * time_consolidation
            + phase_boost
        )

        return ScoredCandidate(
            drawer_id=c.drawer_id, content=c.content,
            wing=c.wing, room=c.room,
            importance=c.importance, confidence=c.confidence,
            total_score=round(min(total, 1.0), 4),
            mode="cold_start",
            component_scores={
                "intrinsic_quality": round(intrinsic, 4),
                "recency": round(recency, 4),
                "connectivity": round(connectivity, 4),
                "time_consolidation": round(time_consolidation, 4),
            },
            phase_boost=round(phase_boost, 4),
        )

    def _score_steady_state(self, c) -> ScoredCandidate:
        """Steady state scoring: full multi-dimensional with recall signals."""
        w = STEADY_STATE_WEIGHTS

        # Recall relevance: BM25 normalized (fixes M2)
        if c.recall_count > 0:
            recall_relevance = 1.0 / (1.0 + abs(c.avg_bm25_rank))
        else:
            recall_relevance = 0.0

        # Recall frequency: log scale
        recall_frequency = math.log1p(c.recall_count) / math.log1p(10)

        # Recall diversity: across sessions/days
        recall_diversity = min(1.0, max(c.recall_sessions, c.recall_days) / 5.0)

        # Intrinsic quality
        intrinsic = (c.importance / 5.0) * c.confidence

        # Recency
        recency = self._recency_score(c.filed_at)

        # Connectivity
        connectivity = min(1.0, c.link_count / CONNECTIVITY_MAX_LINKS)

        # Time consolidation (pure time, fixes m4)
        time_consolidation = self._time_consolidation_score(c.filed_at)

        # Phase boost
        phase_boost = self._phase_boost(c.light_signals, c.rem_signals)

        total = (
            w["recall_relevance"] * recall_relevance
            + w["recall_frequency"] * recall_frequency
            + w["recall_diversity"] * recall_diversity
            + w["intrinsic_quality"] * intrinsic
            + w["recency"] * recency
            + w["connectivity"] * connectivity
            + w["time_consolidation"] * time_consolidation
            + phase_boost
        )

        return ScoredCandidate(
            drawer_id=c.drawer_id, content=c.content,
            wing=c.wing, room=c.room,
            importance=c.importance, confidence=c.confidence,
            total_score=round(min(total, 1.0), 4),
            mode="steady_state",
            component_scores={
                "recall_relevance": round(recall_relevance, 4),
                "recall_frequency": round(recall_frequency, 4),
                "recall_diversity": round(recall_diversity, 4),
                "intrinsic_quality": round(intrinsic, 4),
                "recency": round(recency, 4),
                "connectivity": round(connectivity, 4),
                "time_consolidation": round(time_consolidation, 4),
            },
            phase_boost=round(phase_boost, 4),
        )

    def _recency_score(self, filed_at: str) -> float:
        """Exponential recency decay: half-life = RECENCY_HALF_LIFE_DAYS."""
        if not filed_at:
            return 0.0
        try:
            age_days = self.store.conn.execute(
                "SELECT CAST(julianday('now') - julianday(?) AS REAL)",
                (filed_at,),
            ).fetchone()[0]
            if age_days is None or age_days < 0:
                return 1.0
            half_life = RECENCY_HALF_LIFE_DAYS
            return math.exp(-(math.log(2) / half_life) * age_days)
        except Exception:
            return 0.5

    def _time_consolidation_score(self, filed_at: str) -> float:
        """How long the memory has existed (pure time, fixes m4)."""
        if not filed_at:
            return 0.0
        try:
            days = self.store.conn.execute(
                "SELECT CAST(julianday('now') - julianday(?) AS REAL)",
                (filed_at,),
            ).fetchone()[0]
            if days is None or days < 0:
                return 0.0
            return min(1.0, math.log1p(days) / math.log1p(TIME_CONSOLIDATION_MAX_DAYS))
        except Exception:
            return 0.0

    @staticmethod
    def _phase_boost(light_signals: int, rem_signals: int) -> float:
        """Small boost for drawers that have received dreaming signals."""
        boost = 0.0
        if light_signals > 0:
            boost += 0.05
        if rem_signals > 0:
            boost += 0.08
        return min(boost, 0.10)

    def calculate_memory_health(self) -> float:
        """Memory health score (0.0 ~ 1.0).

        health = (active_ratio × 0.40)
               + (high_importance_ratio × 0.30)
               + (avg_importance / 5.0 × 0.15)
               + (avg_confidence × 0.15)
        """
        s = self.state
        active = s["active_count"]
        if active == 0:
            return 0.0

        active_ratio = min(1.0, active / MAX_EXPECTED_DRAWERS)
        high_imp_ratio = s["high_importance_count"] / active
        avg_imp_ratio = s["avg_importance"] / 5.0
        avg_conf = s["avg_confidence"]

        health = (
            active_ratio * 0.40
            + high_imp_ratio * 0.30
            + avg_imp_ratio * 0.15
            + avg_conf * 0.15
        )
        return round(min(health, 1.0), 4)
