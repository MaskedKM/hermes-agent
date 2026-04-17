"""Feedback-driven importance adjustment for Palace drawers.

Phase 2B of memory enhancement: EMA-based weight updates.

When a user message contains a feedback signal (detected by
``detect_feedback_signal`` in ``queue.py``), we identify which recently-
stored drawers are *relevant* and adjust their importance using an
exponential moving average (EMA).

EMA formula
-----------
    new_importance = α × signal_impact + (1 − α) × current_importance

Where ``signal_impact`` maps the feedback signal strength to a target
importance value, and α is the learning rate (default 0.3).

Usage
-----
Called from the Palace provider's ``_on_user_message`` hook after
auto-extraction completes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import PalaceStorage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# EMA learning rate — how fast importance adapts to feedback.
EMA_ALPHA: float = 0.3

# How many recent drawers (by filed_at) to consider for feedback.
FEEDBACK_WINDOW: int = 20

# Minimum |signal_strength| to trigger an update.
MIN_SIGNAL_THRESHOLD: float = 0.2

# ---------------------------------------------------------------------------
# Signal-to-impact mapping
# ---------------------------------------------------------------------------


def _signal_to_impact(signal_strength: float) -> float:
    """Map a feedback signal in [-1, 1] to a target importance in [1, 5].

    Negative signals push importance down (min 1.0).
    Positive signals push importance up (max 5.0).
    Neutral center (0) maps to 3.0.
    """
    # Linear mapping: -1→1.0, 0→3.0, +1→5.0
    return 3.0 + signal_strength * 2.0


# ---------------------------------------------------------------------------
# FeedbackReactor
# ---------------------------------------------------------------------------


class FeedbackReactor:
    """Apply feedback signals to recently-added Palace drawers."""

    def __init__(self, store: PalaceStorage) -> None:
        self._store = store

    def apply_feedback(
        self,
        signal_strength: float,
        relevant_drawer_ids: list[str] | None = None,
    ) -> int:
        """Adjust importance of drawers based on a feedback signal.

        Parameters
        ----------
        signal_strength:
            Value in [-1, 1] from ``detect_feedback_signal``.
        relevant_drawer_ids:
            Optional explicit drawer IDs to adjust.  If *None*, the
            reactor picks the most recently filed drawers (window).

        Returns
        -------
        Number of drawers updated.
        """
        if abs(signal_strength) < MIN_SIGNAL_THRESHOLD:
            logger.debug("Signal %s below threshold, skipping", signal_strength)
            return 0

        target_ids = self._resolve_targets(relevant_drawer_ids)
        if not target_ids:
            return 0

        impact = _signal_to_impact(signal_strength)
        updates = []

        for drawer_id in target_ids:
            drawer = self._store.get_drawer(drawer_id)
            if drawer is None:
                continue
            current = drawer.get("importance", 3.0)
            new_imp = EMA_ALPHA * impact + (1 - EMA_ALPHA) * current
            # Clamp to [1.0, 5.0]
            new_imp = max(1.0, min(5.0, round(new_imp, 2)))
            updates.append({"id": drawer_id, "importance": new_imp})

        if updates:
            self._store.batch_update_drawers(updates)
            # Also update feedback_weight on each drawer
            for u in updates:
                try:
                    self._store.set_feedback_weight(u["id"], u["importance"] / 5.0)
                except Exception:
                    pass
            logger.info(
                "Feedback reactor: %d drawers updated (signal=%.2f, impact=%.2f)",
                len(updates),
                signal_strength,
                impact,
            )

        return len(updates)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_targets(
        self, explicit_ids: list[str] | None
    ) -> list[str]:
        """Return drawer IDs to adjust."""
        if explicit_ids:
            return explicit_ids

        # Fall back to most recent drawers
        rows = self._store.list_drawers(limit=FEEDBACK_WINDOW)
        return [r["id"] for r in rows if r.get("id")]
