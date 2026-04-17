"""DreamingEngine — Main entry point for the Dreaming memory consolidation system.

Orchestrates Light Sleep, Deep Sleep, and REM Sleep phases.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from plugins.memory.palace.dreaming.collector import RecallCollector
from plugins.memory.palace.dreaming.deduplicator import Deduplicator
from plugins.memory.palace.dreaming.promotion import PromotionEngine
from plugins.memory.palace.dreaming.recovery import DreamingRecovery
from plugins.memory.palace.dreaming.scorer import DreamingScorer
from plugins.memory.palace.dreaming.defaults import (
    LIGHT_LOOKBACK_DAYS,
    LIGHT_LIMIT,
    LIGHT_DEDUP_THRESHOLD,
    LIGHT_MIN_HOURS_BETWEEN_RUNS,
    RECALL_EVENTS_MAX_AGE_DAYS,
)
from plugins.memory.palace.semantic_dedup import SemanticDeduplicator

logger = logging.getLogger(__name__)


class DreamingEngine:
    """Orchestrates Dreaming memory consolidation phases."""

    def __init__(self, store, session_db=None, vector_store=None) -> None:
        self.store = store
        self.session_db = session_db
        self.collector = RecallCollector(store)
        self.deduplicator = Deduplicator()
        self.semantic_deduplicator = SemanticDeduplicator(
            vector_store=vector_store, fallback=self.deduplicator,
        )
        self.scorer = DreamingScorer(store)
        self.promoter = PromotionEngine(store)
        self.recovery = DreamingRecovery(store, session_db)

    # ─── Light Sleep ───────────────────────────────────────────────

    def run_light_sleep(self) -> str:
        """Run Light Sleep phase: deduplication + signal collection.

        Returns a report string suitable for stdout (captured by cron).
        """
        start = time.time()

        # 0. Pre-check: skip if too soon since last run
        last_run = self.store.get_last_dreaming_run("light")
        if last_run and last_run.get("started_at"):
            hours_since = self.store.conn.execute(
                "SELECT CAST((julianday('now') - julianday(?)) * 24 AS REAL)",
                (last_run["started_at"],),
            ).fetchone()[0]
            if hours_since is not None and hours_since < LIGHT_MIN_HOURS_BETWEEN_RUNS:
                msg = f"Light Sleep skipped: last run {hours_since:.1f}h ago (min {LIGHT_MIN_HOURS_BETWEEN_RUNS}h)"
                logger.info(msg)
                return msg

        report: Dict[str, Any] = {
            "phase": "light",
            "drawers_active": 0,
            "drawers_processed": 0,
            "drawers_merged": 0,
            "duration_ms": 0,
            "errors": [],
        }

        try:
            # Get state
            state = self.store.get_dreaming_state()
            report["drawers_active"] = state["active_count"]

            # 1. Collect candidates
            items = self.collector.collect_light_candidates(
                lookback_days=LIGHT_LOOKBACK_DAYS,
                limit=LIGHT_LIMIT,
            )
            report["drawers_processed"] = len(items)

            # 2. Deduplicate (semantic when available, else Jaccard)
            embed_fn = None
            if self.semantic_deduplicator.available and hasattr(self.store, '_embedding_client'):
                embed_fn = self.store._embedding_client.embed_texts
            groups = self.semantic_deduplicator.dedupe(
                items, threshold=LIGHT_DEDUP_THRESHOLD, embed_fn=embed_fn,
            )

            # 3. Consolidate
            consolidation = self.promoter.run_light_consolidation(groups)
            report["drawers_merged"] = consolidation.merged_count
            report["errors"] = consolidation.errors

            # 4. Cleanup old recall events
            cleaned = self.store.cleanup_old_recall_events(RECALL_EVENTS_MAX_AGE_DAYS)
            if cleaned:
                report["recall_events_cleaned"] = cleaned

        except Exception as e:
            report["errors"].append(str(e))
            logger.error("Light Sleep failed: %s", e, exc_info=True)

        # Finalize
        elapsed_ms = int((time.time() - start) * 1000)
        report["duration_ms"] = elapsed_ms
        status = "success" if not report["errors"] else "partial"
        self.store.save_dreaming_run("light", report, status=status)

        summary = (
            f"Light Sleep [{status}]: "
            f"processed={report['drawers_processed']}, "
            f"merged={report['drawers_merged']}, "
            f"active={report['drawers_active']}, "
            f"duration={elapsed_ms}ms"
        )
        if report["errors"]:
            summary += f", errors={len(report['errors'])}"
        logger.info(summary)
        return summary

    # ─── Deep Sleep ────────────────────────────────────────────────

    def run_deep_sleep(self) -> str:
        """Run Deep Sleep phase: scoring + promotion + archival.

        Returns a report string suitable for stdout.
        """
        start = time.time()
        is_cold = self.scorer.is_cold_start()
        health = self.scorer.calculate_memory_health()

        report: Dict[str, Any] = {
            "phase": "deep",
            "mode": "cold_start" if is_cold else "steady_state",
            "memory_health": health,
            "drawers_active": 0,
            "drawers_processed": 0,
            "drawers_promoted": 0,
            "drawers_archived": 0,
            "stale_confirmed": 0,
            "duration_ms": 0,
            "errors": [],
        }

        try:
            # 0. Recovery check
            if self.recovery.should_trigger(health):
                logger.warning("Memory health %.3f < %.3f, triggering recovery",
                               health, 0.35)
                recovery_report = self.recovery.run_recovery(health)
                report["recovery"] = {
                    "recovered": recovery_report.recovered_count,
                    "health_after": recovery_report.health_after,
                }

            # Get state
            state = self.store.get_dreaming_state()
            report["drawers_active"] = state["active_count"]

            # 1. Collect candidates with full stats
            candidates = self.collector.collect_deep_candidates()
            report["drawers_processed"] = len(candidates)

            # 2. Score all candidates
            scored = [self.scorer.score(c) for c in candidates]

            # 3. Promote & Archive
            promotion = self.promoter.run_deep_promotion(scored, is_cold)
            report["drawers_promoted"] = promotion.promoted_count
            report["drawers_archived"] = promotion.archived_count + promotion.superseded_archived_count
            report["stale_confirmed"] = promotion.stale_confirmed_count
            report["errors"] = promotion.errors

            # 3b. Cold start exit estimation
            if is_cold:
                report["cold_start_exit"] = self._estimate_cold_start_exit(state)

            # 4. Optional: LLM consolidation for top promoted (future enhancement)
            # TODO: integrate with auxiliary_client for LLM polish

        except Exception as e:
            report["errors"].append(str(e))
            logger.error("Deep Sleep failed: %s", e, exc_info=True)

        # Finalize
        elapsed_ms = int((time.time() - start) * 1000)
        report["duration_ms"] = elapsed_ms
        status = "success" if not report["errors"] else "partial"
        self.store.save_dreaming_run("deep", report, status=status)

        summary = (
            f"Deep Sleep [{report['mode']}, {status}]: "
            f"health={health:.3f}, "
            f"processed={report['drawers_processed']}, "
            f"promoted={report['drawers_promoted']}, "
            f"archived={report['drawers_archived']}, "
            f"stale_confirmed={report.get('stale_confirmed', 0)}, "
            f"duration={elapsed_ms}ms"
        )
        if report.get("cold_start_exit"):
            exit_est = report["cold_start_exit"]
            summary += (
                f"\n  Cold start exit: "
                f"recall={exit_est['recall_count']}/{exit_est['recall_needed']}, "
                f"days={exit_est['days_active']}/{exit_est['days_needed']}"
            )
        if report["errors"]:
            summary += f"\n  errors={len(report['errors'])}"
        logger.info(summary)
        return summary

    def _estimate_cold_start_exit(self, state: dict) -> dict:
        """Estimate how many deep sleep runs until cold start exits.

        Uses current recall event count and days since first dreaming run
        to project against COLD_START_MIN_RECALL_EVENTS and
        COLD_START_MIN_DAYS thresholds.
        """
        from plugins.memory.palace.dreaming.defaults import (
            COLD_START_MIN_RECALL_EVENTS,
            COLD_START_MIN_DAYS,
        )

        recall_count = state.get("total_recall_count", 0)
        recall_needed = max(0, COLD_START_MIN_RECALL_EVENTS - recall_count)

        days_active = state.get("days_since_first_dreaming_run", 0)
        days_needed = max(0, COLD_START_MIN_DAYS - days_active)

        return {
            "recall_count": recall_count,
            "recall_needed": recall_needed,
            "days_active": days_active,
            "days_needed": days_needed,
        }

    # ─── REM Sleep ─────────────────────────────────────────────────

    def run_rem_sleep(self) -> str:
        """Run REM Sleep phase: pattern recognition + deep reflection.

        Returns a report string suitable for stdout.
        """
        start = time.time()

        report: Dict[str, Any] = {
            "phase": "rem",
            "patterns_found": 0,
            "truths_extracted": 0,
            "duration_ms": 0,
            "errors": [],
        }

        try:
            # 1. Collect patterns
            patterns = self.collector.collect_rem_patterns()

            if not patterns:
                report["status"] = "skipped_insufficient_data"
                elapsed_ms = int((time.time() - start) * 1000)
                report["duration_ms"] = elapsed_ms
                self.store.save_dreaming_run("rem", report, status="skipped")
                msg = "REM Sleep skipped: insufficient co-occurrence data"
                logger.info(msg)
                return msg

            # 2. LLM pattern analysis (future enhancement)
            # TODO: integrate with auxiliary_client for pattern analysis
            truths = []  # Will be populated by LLM analysis

            # 3. Store reflections
            reflection = self.promoter.run_rem_reflection(patterns, truths)
            report["patterns_found"] = reflection.patterns_found
            report["truths_extracted"] = reflection.truths_extracted
            report["errors"] = reflection.errors

        except Exception as e:
            report["errors"].append(str(e))
            logger.error("REM Sleep failed: %s", e, exc_info=True)

        # Finalize
        elapsed_ms = int((time.time() - start) * 1000)
        report["duration_ms"] = elapsed_ms
        status = "success" if not report["errors"] else "partial"
        self.store.save_dreaming_run("rem", report, status=status)

        summary = (
            f"REM Sleep [{status}]: "
            f"patterns={report['patterns_found']}, "
            f"truths={report['truths_extracted']}, "
            f"duration={elapsed_ms}ms"
        )
        if report["errors"]:
            summary += f", errors={len(report['errors'])}"
        logger.info(summary)
        return summary
