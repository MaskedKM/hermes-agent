"""DreamingRecovery — Memory health recovery when quality degrades.

Triggered automatically when memory health drops below threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from plugins.memory.palace.dreaming.defaults import (
    RECOVERY_TRIGGER_BELOW_HEALTH,
    RECOVERY_LOOKBACK_DAYS,
    RECOVERY_MAX_CANDIDATES,
    RECOVERY_AUTO_WRITE_MIN_CONFIDENCE,
)

logger = logging.getLogger(__name__)


@dataclass
class RecoveryReport:
    """Report from memory recovery process."""
    triggered: bool = False
    health_before: float = 0.0
    health_after: float = 0.0
    recovered_count: int = 0
    candidates_found: int = 0
    errors: List[str] = field(default_factory=list)


class DreamingRecovery:
    """Memory health recovery — triggered when health drops below threshold."""

    def __init__(self, store, session_db=None) -> None:
        self.store = store
        self.session_db = session_db

    def should_trigger(self, health: float) -> bool:
        """Check if recovery should be triggered."""
        return health < RECOVERY_TRIGGER_BELOW_HEALTH

    def run_recovery(self, health: float) -> RecoveryReport:
        """Attempt to recover memory quality.

        1. Search session history for important lost memories
        2. Auto-recover high-confidence candidates
        """
        report = RecoveryReport(
            triggered=True,
            health_before=health,
        )

        if not self.session_db:
            report.errors.append("No SessionDB available for recovery")
            return report

        try:
            # Search for potentially important memories in recent sessions
            candidates = self._find_recovery_candidates()
            report.candidates_found = len(candidates)

            # Auto-recover high-confidence ones
            for c in candidates:
                if report.recovered_count >= RECOVERY_MAX_CANDIDATES:
                    break
                conf = c.get("confidence", 0)
                if conf >= RECOVERY_AUTO_WRITE_MIN_CONFIDENCE:
                    self.store.add_drawer(
                        content=c["content"],
                        wing=c.get("wing", "general"),
                        room=c.get("room", "recovered"),
                        importance=c.get("importance", 3.0),
                        confidence=conf,
                        memory_type="recovered",
                    )
                    report.recovered_count += 1

        except Exception as e:
            report.errors.append(f"Recovery error: {e}")
            logger.warning("Dreaming recovery error: %s", e)

        # Recalculate health
        from plugins.memory.palace.dreaming.scorer import DreamingScorer
        scorer = DreamingScorer(self.store)
        report.health_after = scorer.calculate_memory_health()

        logger.info(
            "Recovery: before=%.3f, after=%.3f, candidates=%d, recovered=%d",
            report.health_before, report.health_after,
            report.candidates_found, report.recovered_count,
        )
        return report

    def _find_recovery_candidates(self) -> List[Dict[str, Any]]:
        """Search session history for important memories that may have been lost."""
        candidates = []

        try:
            # Search for key terms in recent session messages
            search_terms = [
                "important", "decision", "preference", "rule",
                "重要", "决策", "偏好", "规则", "记住", "配置",
            ]
            for term in search_terms:
                try:
                    rows = self.session_db.search_messages(term, limit=5)
                    for row in rows:
                        content = row.get("content", "")
                        if content and len(content) > 20:
                            candidates.append({
                                "content": content[:500],
                                "confidence": 0.7,
                                "importance": 3.0,
                                "wing": "general",
                                "room": "recovered",
                            })
                except Exception:
                    continue

        except Exception as e:
            logger.debug("Recovery candidate search failed: %s", e)

        # Deduplicate by content prefix
        seen = set()
        unique = []
        for c in candidates:
            key = c["content"][:50]
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique[:RECOVERY_MAX_CANDIDATES]
