"""Memory Palace plugin — hierarchical knowledge storage with FTS5 search."""

from .provider import PalaceMemoryProvider, register

__all__ = ["PalaceMemoryProvider", "register"]
