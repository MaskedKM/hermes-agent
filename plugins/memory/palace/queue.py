"""Debounce queue for LLM memory extraction.

Batches rapid successive turns into a single LLM call to reduce
API usage and improve extraction quality.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# detect_correction  – lightweight heuristic, no LLM needed
# ---------------------------------------------------------------------------

_CORRECTION_PATTERNS = [
    "不对",
    "不是",
    "错了",
    "别这样",
    "不要这样",
    "我说的不是",
    "我纠正",
    "纠正一下",
    "我说的是",
    "不，",
    "actually",
    "no, that",
    "that's wrong",
    "i meant",
    "correction",
    "i didn't say",
    "not what i",
    "let me correct",
]


def detect_correction(text: str) -> bool:
    """Return True if *text* looks like a user correction signal."""
    if not text:
        return False
    lower = text.lower()
    return any(pat in lower for pat in _CORRECTION_PATTERNS)


# ---------------------------------------------------------------------------
# detect_feedback_signal  – graded feedback detection (Phase 2)
# ---------------------------------------------------------------------------

_FEEDBACK_SIGNALS: dict[str, float] = {
    # Negative (correction)
    "不对": -0.6,
    "错了": -0.7,
    "的错": -0.5,
    "不是这样": -0.6,
    "错了不是": -0.7,
    "别这样": -0.5,
    "不要这样": -0.5,
    "我说的不是": -0.6,
    "我纠正": -0.6,
    "纠正一下": -0.6,
    "我说的是": -0.4,
    "不，": -0.3,
    "actually": -0.5,
    "that's wrong": -0.6,
    "incorrect": -0.5,
    "不是": -0.3,
    "错了啦": -0.6,
    "no, that": -0.5,
    "i meant": -0.4,
    "correction": -0.5,
    "i didn't say": -0.6,
    "not what i": -0.5,
    "let me correct": -0.5,
    # Positive (affirmation)
    "没错": 0.4,
    "对的": 0.4,
    "就是这样": 0.5,
    "正确": 0.4,
    "exactly": 0.5,
    "that's right": 0.4,
    "correct": 0.4,
    "有用": 0.4,
    "有帮助": 0.4,
    "谢谢": 0.3,
    "感谢": 0.3,
    "好的": 0.2,
    "太棒了": 0.5,
    "很好": 0.4,
    "解决了": 0.5,
    "搞定了": 0.5,
    "完美": 0.5,
    "做得好": 0.5,
    "干得好": 0.5,
    "厉害": 0.4,
    "不错": 0.3,
    "great": 0.4,
    "good job": 0.5,
    "thanks": 0.3,
    "helpful": 0.4,
    "perfect": 0.5,
    "well done": 0.5,
    "excellent": 0.5,
}


def detect_feedback_signal(text: str) -> float | None:
    """Detect feedback signal in *text*.

    Returns signal_strength in [-1, 1], or None when no keyword matches.
    Positive = affirmation, negative = correction.  When multiple keywords
    match the one with the largest absolute value wins.
    """
    if not text:
        return None
    lower = text.lower()
    best: float | None = None
    for pattern, strength in _FEEDBACK_SIGNALS.items():
        if pattern in lower:
            if best is None or abs(strength) > abs(best):
                best = strength
    return best


# ---------------------------------------------------------------------------
# ExtractionQueue  – debounce rapid turns, then call LLM extractor
# ---------------------------------------------------------------------------

class ExtractionQueue:
    """Collects turn texts and fires a single LLM extraction after a quiet
    period of *debounce_seconds* elapses without new input.

    Parameters
    ----------
    debounce_seconds : float
        Seconds of silence before flushing the queue.
    extract_fn : callable(text, min_confidence) -> dict
        The LLM extraction function (e.g. ``extract_memories_with_fallback``).
    store_fn : callable(memories, entities, has_correction) -> None
        Callback to store extracted results.
    """

    def __init__(
        self,
        debounce_seconds: float = 30.0,
        extract_fn: Optional[Callable] = None,
        store_fn: Optional[Callable] = None,
    ) -> None:
        self._debounce = debounce_seconds
        self._extract_fn = extract_fn
        self._store_fn = store_fn

        self._queue: List[str] = []
        self._min_confidence: float = 0.3
        self._has_correction: bool = False
        self._keyword_feedback: Optional[float] = None  # fast keyword pre-check
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._stopped = False

    def enqueue(
        self,
        text: str,
        *,
        min_confidence: float = 0.3,
        has_correction: bool = False,
        keyword_feedback: Optional[float] = None,
    ) -> None:
        """Add *text* to the pending queue and (re)set the debounce timer."""
        if self._stopped or not text:
            return

        with self._lock:
            self._queue.append(text)
            if min_confidence > self._min_confidence:
                self._min_confidence = min_confidence
            if has_correction:
                self._has_correction = True
            if keyword_feedback is not None:
                # Keep strongest keyword signal
                if self._keyword_feedback is None or abs(keyword_feedback) > abs(self._keyword_feedback):
                    self._keyword_feedback = keyword_feedback

        # Cancel any pending flush and schedule a new one
        if self._timer is not None:
            self._timer.cancel()

        self._timer = threading.Timer(self._debounce, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self) -> None:
        """Combine queued texts and run the LLM extractor."""
        with self._lock:
            if not self._queue:
                return
            combined = "\n".join(self._queue)
            min_conf = self._min_confidence
            has_corr = self._has_correction
            kw_feedback = self._keyword_feedback
            self._queue.clear()
            self._min_confidence = 0.3
            self._has_correction = False
            self._keyword_feedback = None

        if not combined.strip() or not self._extract_fn or not self._store_fn:
            return

        try:
            result = self._extract_fn(combined, min_confidence=min_conf)
            if not result:
                return
            # Accept (memories, entities, feedback) from LLM, or legacy 2-tuple
            if isinstance(result, tuple) and len(result) == 3:
                memories, entities, feedback = result
            elif isinstance(result, dict):
                memories = result.get("memories", [])
                entities = result.get("entities", [])
                feedback = None
            elif isinstance(result, list):
                memories = result
                entities = []
                feedback = None
            elif isinstance(result, tuple) and len(result) == 2:
                memories, entities = result
                feedback = None
            else:
                return

            # If LLM didn't detect feedback but keyword pre-check did, use keyword signal
            if feedback is None and kw_feedback is not None:
                signal_type = "positive" if kw_feedback > 0 else "negative"
                feedback = {
                    "signal": signal_type,
                    "strength": abs(kw_feedback),
                    "targets": [],  # keyword can't identify targets
                    "source": "keyword_fallback",
                }

            self._store_fn(
                memories=memories,
                entities=entities,
                has_correction=has_corr,
                feedback=feedback,
            )
        except Exception as e:
            logger.debug("ExtractionQueue flush failed: %s", e)

    def stop(self) -> None:
        """Stop the queue and flush any remaining items synchronously."""
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        # Final flush
        self._flush()
