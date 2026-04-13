"""Tests for Palace store — recall-weighted top_importance and archive."""

import os
import tempfile

import pytest

from plugins.memory.palace.store import PalaceStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh PalaceStore with an in-memory DB."""
    db_path = str(tmp_path / "test.db")
    s = PalaceStore(db_path)
    yield s
    s.close()


# ------------------------------------------------------------------
# Phase 4: recall-weighted get_top_importance
# ------------------------------------------------------------------

class TestRecallWeightedTopImportance:
    """get_top_importance should rank drawers higher when they have recall events."""

    def test_recall_boosts_ranking(self, store: PalaceStore):
        """A drawer with recall events should outrank a same-importance drawer without."""
        # Drawer A: importance=5.0, confidence=1.0, no recall
        id_a = store.add_drawer(
            "Drawer A content", wing="w", room="room_a",
            importance=5.0, confidence=1.0,
        )
        # Drawer B: importance=5.0, confidence=1.0, with 5 recall events
        id_b = store.add_drawer(
            "Drawer B content", wing="w", room="room_b",
            importance=5.0, confidence=1.0,
        )
        for i in range(5):
            store.record_recall(id_b, session_id=f"s{i}", query="test")

        result = store.get_top_importance(n=2)
        lines = result.strip().split("\n")
        # room_b should appear before room_a
        room_order = []
        for line in lines:
            if "room_a" in line:
                room_order.append("room_a")
            elif "room_b" in line:
                room_order.append("room_b")
        assert room_order == ["room_b", "room_a"], (
            f"Expected room_b before room_a, got {room_order}"
        )

    def test_no_recall_still_works(self, store: PalaceStore):
        """Without any recall events, ranking falls back to importance*confidence."""
        id_low = store.add_drawer(
            "Low importance", wing="w", room="low",
            importance=3.0, confidence=0.5,
        )
        id_high = store.add_drawer(
            "High importance", wing="w", room="high",
            importance=5.0, confidence=1.0,
        )
        result = store.get_top_importance(n=2)
        lines = result.strip().split("\n")
        assert "high" in lines[0]
        assert "low" in lines[1]

    def test_empty_db_returns_empty(self, store: PalaceStore):
        assert store.get_top_importance() == ""

    def test_recall_weight_formula_saturates(self, store: PalaceStore):
        """Even a very high recall count shouldn't cause a low-importance drawer
        to beat a very high importance drawer."""
        # Drawer low: importance=2.0, confidence=0.5, 100 recalls
        id_low = store.add_drawer(
            "Low drawer", wing="w", room="low",
            importance=2.0, confidence=0.5,
        )
        for i in range(100):
            store.record_recall(id_low, session_id=f"s{i}", query="test")

        # Drawer high: importance=5.0, confidence=1.0, 0 recalls
        id_high = store.add_drawer(
            "High drawer", wing="w", room="high",
            importance=5.0, confidence=1.0,
        )

        result = store.get_top_importance(n=2)
        lines = result.strip().split("\n")
        assert "high" in lines[0], "Very high importance should still beat low+high recall"


# ------------------------------------------------------------------
# Phase 4: archive stale debugging drawer
# ------------------------------------------------------------------

class TestArchiveDrawer:
    def test_archive_removes_from_top(self, store: PalaceStore):
        """Archived drawers should not appear in get_top_importance."""
        id_a = store.add_drawer(
            "Stale debug info", wing="problems", room="debugging",
            importance=5.0, confidence=1.0,
        )
        result_before = store.get_top_importance(n=1)
        assert "Stale debug info" in result_before

        archived = store.archive_drawer(id_a)
        assert archived is True

        result_after = store.get_top_importance(n=1)
        assert "Stale debug info" not in result_after

    def test_archive_nonexistent_returns_false(self, store: PalaceStore):
        assert store.archive_drawer("nonexistent_id") is False
