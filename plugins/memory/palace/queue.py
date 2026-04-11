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
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._stopped = False

    def enqueue(
        self,
        text: str,
        *,
        min_confidence: float = 0.3,
        has_correction: bool = False,
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
            self._queue.clear()
            self._min_confidence = 0.3
            self._has_correction = False

        if not combined.strip() or not self._extract_fn or not self._store_fn:
            return

        try:
            result = self._extract_fn(combined, min_confidence=min_conf)
            if not result:
                return
            # Accept both {memories: [...], entities: [...]} and plain list
            if isinstance(result, dict):
                memories = result.get("memories", [])
                entities = result.get("entities", [])
            elif isinstance(result, list):
                memories = result
                entities = []
            else:
                return

            self._store_fn(
                memories=memories,
                entities=entities,
                has_correction=has_corr,
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
