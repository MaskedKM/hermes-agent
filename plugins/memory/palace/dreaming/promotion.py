"""PromotionEngine — Memory promotion, archival, and consolidation decisions.

Handles Light Sleep consolidation, Deep Sleep promotion/archival,
and REM Sleep reflection storage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from plugins.memory.palace.dreaming.defaults import (
    COLD_START_ARCHIVE_RULES,
    STEADY_STATE_ARCHIVE_RULES,
    DEEP_MIN_SCORE,
    DEEP_MAX_PROMOTIONS,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationReport:
    """Report from Light Sleep consolidation."""
    merged_count: int = 0
    updated_count: int = 0
    skipped_count: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class PromotionReport:
    """Report from Deep Sleep promotion."""
    mode: str = ""  # "cold_start" | "steady_state"
    memory_health: float = 0.0
    promoted_count: int = 0
    archived_count: int = 0
    superseded_archived_count: int = 0
    stale_confirmed_count: int = 0  # high-importance drawers confirmed despite no recalls
    errors: List[str] = field(default_factory=list)


@dataclass
class ReflectionReport:
    """Report from REM Sleep reflection."""
    patterns_found: int = 0
    truths_extracted: int = 0
    errors: List[str] = field(default_factory=list)


class PromotionEngine:
    """Memory promotion and archival decision engine."""

    def __init__(self, store) -> None:
        self.store = store

    def run_light_consolidation(self, dedup_groups) -> ConsolidationReport:
        """Light Sleep: merge duplicate drawers and boost representatives.

        For each DedupGroup:
        1. Select representative (longest/newest — already done by Deduplicator)
        2. Merge all non-representative members into representative
        3. Boost representative: importance += 0.3, confidence += 0.1
        4. Record phase signal
        """
        report = ConsolidationReport()

        for group in dedup_groups:
            rep_id = group.representative_id
            member_ids = [m for m in group.member_ids if m != rep_id]

            if not member_ids:
                report.skipped_count += 1
                continue

            try:
                # Merge all members into representative
                merged_ok = 0
                for member_id in member_ids:
                    if self.store.merge_drawers(member_id, rep_id):
                        merged_ok += 1

                if merged_ok > 0:
                    # Boost representative
                    rep = self.store.get_drawer(rep_id)
                    if rep:
                        new_imp = min(5.0, rep["importance"] + 0.3)
                        new_conf = min(1.0, rep["confidence"] + 0.1)
                        self.store.conn.execute(
                            "UPDATE drawers SET importance = ?, confidence = ? WHERE id = ?",
                            (new_imp, new_conf, rep_id),
                        )
                        self.store.conn.commit()

                    # Record phase signal
                    self.store.add_dreaming_signal(rep_id, "light", strength=1.0)
                    report.merged_count += merged_ok
                    report.updated_count += 1
                else:
                    report.skipped_count += 1

            except Exception as e:
                report.errors.append(f"Error merging group {rep_id}: {e}")
                logger.warning("Light consolidation error for %s: %s", rep_id, e)

        logger.info(
            "Light consolidation: merged=%d, updated=%d, skipped=%d, errors=%d",
            report.merged_count, report.updated_count, report.skipped_count,
            len(report.errors),
        )
        return report

    def run_deep_promotion(self, scored: list, is_cold_start: bool) -> PromotionReport:
        """Deep Sleep: promote high-scoring memories and archive low-scoring ones.

        Dual-mode: cold start (conservative) vs steady state (normal).
        """
        report = PromotionReport(mode="cold_start" if is_cold_start else "steady_state")

        # Sort by score descending
        sorted_scored = sorted(scored, key=lambda s: s.total_score, reverse=True)

        if is_cold_start:
            self._promote_cold_start(sorted_scored, report)
            self._confirm_stale_high_importance(report)
            self._archive_cold_start(sorted_scored, report)
        else:
            self._promote_steady_state(sorted_scored, report)
            self._archive_steady_state(sorted_scored, report)

        # Archive old superseded drawers
        self._archive_old_superseded(report)

        logger.info(
            "Deep promotion [%s]: promoted=%d, archived=%d, superseded_archived=%d",
            report.mode, report.promoted_count, report.archived_count,
            report.superseded_archived_count,
        )
        return report

    def _promote_cold_start(self, scored: list, report: PromotionReport) -> None:
        """Cold start promotion: promote high-importance drawers conservatively.

        Uses importance directly (not intrinsic) since confidence is
        unreliable during cold start (no recall data to validate it).
        - importance >= 4.5 AND < 5.0: boost +0.3 importance, +0.1 confidence
        - Skips drawers already at 5.0 to prevent re-promotion saturation
        """
        updates = []
        for s in scored:
            if s.importance >= 4.5 and s.importance < 5.0:
                new_imp = min(5.0, s.importance + 0.3)
                new_conf = min(1.0, s.confidence + 0.1)
                updates.append({
                    "id": s.drawer_id,
                    "importance": new_imp,
                    "confidence": new_conf,
                    "memory_type": "consolidated",
                })
                self.store.add_dreaming_signal(s.drawer_id, "deep", strength=1.0)
                report.promoted_count += 1

        if updates:
            self.store.bulk_update_scores(updates)

    def _confirm_stale_high_importance(self, report: PromotionReport) -> None:
        """Confirm high-importance drawers that haven't been recalled recently.

        During cold start, some high-value drawers may never be recalled
        (cold topics, infrequent queries). Without this safety net, they
        risk being archived by cold start rules despite being important.

        Effect: mild confidence boost (+0.05), capped at 0.7.
        Runs at most once per drawer (tracked via dreaming_signals).
        """
        STALE_MIN_AGE_DAYS = 14
        STALE_MIN_IMPORTANCE = 4.0
        STALE_CONFIDENCE_BOOST = 0.05
        STALE_MAX_CONFIDENCE = 0.70

        rows = self.store.conn.execute(
            """
            SELECT d.id, d.importance, d.confidence, d.filed_at
            FROM drawers d
            LEFT JOIN dreaming_signals ds
                ON ds.drawer_id = d.id AND ds.phase = 'stale_confirm'
            WHERE d.archived = 0
              AND d.importance >= ?
              AND d.confidence < ?
              AND julianday('now') - julianday(d.filed_at) > ?
              AND ds.id IS NULL
            LIMIT 50
            """,
            (STALE_MIN_IMPORTANCE, STALE_MAX_CONFIDENCE, STALE_MIN_AGE_DAYS),
        ).fetchall()

        updates = []
        for row in rows:
            new_conf = min(STALE_MAX_CONFIDENCE, row["confidence"] + STALE_CONFIDENCE_BOOST)
            updates.append({"id": row["id"], "confidence": new_conf})
            # Mark as confirmed to prevent re-processing
            self.store.add_dreaming_signal(row["id"], "stale_confirm", strength=0.5)

        if updates:
            self.store.bulk_update_scores(updates)

        report.stale_confirmed_count = len(updates)
        if updates:
            logger.info(
                "Stale high-importance confirmed: %d drawers (age>%dd, imp>=%.1f)",
                len(updates), STALE_MIN_AGE_DAYS, STALE_MIN_IMPORTANCE,
            )

    def _promote_steady_state(self, scored: list, report: PromotionReport) -> None:
        """Steady state promotion: top N above threshold."""
        updates = []
        promoted = 0
        for s in scored:
            if promoted >= DEEP_MAX_PROMOTIONS:
                break
            if s.total_score >= DEEP_MIN_SCORE:
                new_imp = min(5.0, s.importance + 1.0)
                new_conf = min(1.0, s.confidence + 0.2)
                updates.append({
                    "id": s.drawer_id,
                    "importance": new_imp,
                    "confidence": new_conf,
                    "memory_type": "consolidated",
                })
                self.store.add_dreaming_signal(s.drawer_id, "deep", strength=1.0)
                promoted += 1
                report.promoted_count += 1

        if updates:
            self.store.bulk_update_scores(updates)

    def _archive_cold_start(self, scored: list, report: PromotionReport) -> None:
        """Cold start archival: very conservative rules."""
        rules = COLD_START_ARCHIVE_RULES
        archived = 0

        for s in scored:
            if archived >= rules["max_archived_per_run"]:
                break
            if s.total_score >= rules["max_score_threshold"]:
                continue
            # Check age
            age_days = self._get_age_days(s)
            if age_days < rules["min_age_days"]:
                continue
            if rules["require_zero_links"] and s.component_scores.get("connectivity", 0) > 0:
                continue
            if rules["require_zero_recalls"] and s.component_scores.get("recall_frequency", 0) > 0:
                continue

            if self.store.archive_drawer(s.drawer_id):
                archived += 1
                report.archived_count += 1

    def _archive_steady_state(self, scored: list, report: PromotionReport) -> None:
        """Steady state archival: normal rules."""
        rules = STEADY_STATE_ARCHIVE_RULES
        archived = 0

        for s in scored:
            if archived >= rules["max_archived_per_run"]:
                break
            if s.total_score >= rules["max_score_threshold"]:
                continue
            age_days = self._get_age_days(s)
            if age_days < rules["min_age_days"]:
                continue

            if self.store.archive_drawer(s.drawer_id):
                archived += 1
                report.archived_count += 1

    def _archive_old_superseded(self, report: PromotionReport) -> None:
        """Archive superseded drawers older than 7 days."""
        rows = self.store.conn.execute(
            """
            SELECT id FROM drawers
            WHERE superseded_by IS NOT NULL AND archived = 0
              AND julianday('now') - julianday(filed_at) > 7
            LIMIT 50
            """
        ).fetchall()
        for row in rows:
            if self.store.archive_drawer(row["id"]):
                report.superseded_archived_count += 1

    def _get_age_days(self, scored_item) -> float:
        """Get age in days for a scored candidate."""
        try:
            return self.store.conn.execute(
                "SELECT CAST(julianday('now') - julianday(?) AS REAL)",
                (scored_item.component_scores.get("filed_at", "") or scored_item.drawer_id,),
            ).fetchone()[0] or 0
        except Exception:
            return 0

    def run_rem_reflection(self, patterns: list, truths: list) -> ReflectionReport:
        """REM Sleep: store patterns and lasting truths as Palace drawers."""
        report = ReflectionReport()

        # Store patterns
        for p in patterns:
            if report.patterns_found >= 20:
                break
            try:
                import json
                content = json.dumps({
                    "entities": p.entities,
                    "frequency": p.frequency,
                    "examples": p.context_examples[:3],
                }, ensure_ascii=False, default=str)
                self.store.add_drawer(
                    content=content,
                    wing="project",
                    room="patterns",
                    importance=3.0,
                    confidence=0.7,
                    memory_type="pattern",
                )
                report.patterns_found += 1
            except Exception as e:
                report.errors.append(f"Pattern storage error: {e}")

        # Store lasting truths
        for t in truths:
            if report.truths_extracted >= 10:
                break
            try:
                content = t.get("content", "")
                if not content:
                    continue
                tags = t.get("tags", [])
                self.store.add_drawer(
                    content=content,
                    wing="general",
                    room="truths",
                    importance=min(5.0, t.get("importance", 4.5)),
                    confidence=min(1.0, t.get("confidence", 0.9)),
                    memory_type="truth",
                )
                self.store.add_dreaming_signal(
                    self.store._make_drawer_id(content, "general", "truths"),
                    "rem", strength=1.0,
                )
                report.truths_extracted += 1
            except Exception as e:
                report.errors.append(f"Truth storage error: {e}")

        logger.info("REM reflection: patterns=%d, truths=%d",
                     report.patterns_found, report.truths_extracted)
        return report
