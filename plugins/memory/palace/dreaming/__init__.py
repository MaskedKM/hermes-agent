"""Dreaming — Memory consolidation system for the Memory Palace.

Three-phase automatic memory management inspired by OpenClaw:
- Light Sleep (every 6h): deduplication + signal collection
- Deep Sleep (daily 3am): scoring + promotion + archival
- REM Sleep (weekly Sunday 5am): pattern recognition + deep reflection
"""

from plugins.memory.palace.dreaming.engine import DreamingEngine

__all__ = ["DreamingEngine"]
