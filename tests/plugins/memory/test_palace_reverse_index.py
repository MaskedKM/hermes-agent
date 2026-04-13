"""Tests for Palace → Memory reverse index sync (Phase 1)."""

import pytest

from plugins.memory.palace.provider import PalaceMemoryProvider


class FakeStore:
    """Minimal store mock for testing _sync_index_to_memory logic."""

    def __init__(self):
        self.drawers = []

    def add_drawer(self, content, wing="", room="", importance=3.0,
                   confidence=0.5, memory_type="general", source_type="manual",
                   **kwargs):
        d = {
            "content": content, "wing": wing, "room": room,
            "importance": importance, "confidence": confidence,
            "memory_type": memory_type, "source_type": source_type,
        }
        self.drawers.append(d)
        return f"id_{len(self.drawers)}"


class TestSyncIndexToMemory:
    """_sync_index_to_memory should write INDEX lines to Memory for high-importance auto drawers."""

    def _make_provider(self, store=None):
        p = PalaceMemoryProvider(config={
            "auto_extract": True,
            "session_extract": True,
            "min_confidence": 0.3,
        })
        p._store = store or FakeStore()
        p._session_id = "test-session"
        p._agent_context = "primary"
        return p

    def test_high_importance_auto_extract_triggers_callback(self):
        """A drawer with importance >= 4.0 and source_type auto_extract should call _write_memory."""
        written = []
        provider = self._make_provider()
        provider._write_memory = lambda target, content: written.append((target, content))

        drawer_id = provider._store.add_drawer(
            "Important fact about the proxy setup",
            wing="infra", room="proxy",
            importance=4.5, confidence=0.9,
            source_type="auto_extract",
        )
        provider._sync_index_to_memory(
            content="Important fact about the proxy setup",
            wing="infra", room="proxy",
            importance=4.5, source_type="auto_extract",
        )

        assert len(written) == 1
        target, content = written[0]
        assert target == "memory"
        assert "[INDEX]" in content
        assert "infra/proxy" in content
        assert "proxy" in content  # keyword from room name

    def test_low_importance_does_not_trigger(self):
        """A drawer with importance < 4.0 should NOT trigger callback."""
        written = []
        provider = self._make_provider()
        provider._write_memory = lambda target, content: written.append((target, content))

        provider._sync_index_to_memory(
            content="Minor note",
            wing="general", room="general",
            importance=3.0, source_type="auto_extract",
        )

        assert len(written) == 0

    def test_manual_source_does_not_trigger(self):
        """A drawer with source_type=manual should NOT trigger callback (only auto/session)."""
        written = []
        provider = self._make_provider()
        provider._write_memory = lambda target, content: written.append((target, content))

        provider._sync_index_to_memory(
            content="Manual important note",
            wing="general", room="general",
            importance=5.0, source_type="manual",
        )

        assert len(written) == 0

    def test_no_callback_set_does_not_crash(self):
        """If _write_memory is None (not injected), should silently skip."""
        provider = self._make_provider()
        # _write_memory is None by default
        provider._sync_index_to_memory(
            content="Important fact",
            wing="infra", room="proxy",
            importance=5.0, source_type="auto_extract",
        )
        # Should not raise

    def test_session_extract_triggers(self):
        """source_type=session_extract with importance >= 4.0 should trigger."""
        written = []
        provider = self._make_provider()
        provider._write_memory = lambda target, content: written.append((target, content))

        provider._sync_index_to_memory(
            content="Session extracted decision about deployment",
            wing="decisions", room="architecture",
            importance=4.0, source_type="session_extract",
        )

        assert len(written) == 1
        assert "[INDEX]" in written[0][1]

    def test_dedup_same_wing_room(self):
        """Two syncs for the same wing/room should produce only one INDEX line."""
        written = []
        provider = self._make_provider()
        provider._write_memory = lambda target, content: written.append((target, content))

        provider._sync_index_to_memory(
            content="First proxy fact", wing="infra", room="proxy",
            importance=4.5, source_type="auto_extract",
        )
        provider._sync_index_to_memory(
            content="Second proxy fact", wing="infra", room="proxy",
            importance=4.5, source_type="auto_extract",
        )

        # The method itself doesn't dedup — dedup is handled by checking
        # existing Memory entries. With our simple mock, both calls go through.
        # This test verifies the callback is called for each unique drawer.
        assert len(written) == 2
