"""Tests for Session → Memory rule refinement (Phase 2)."""

import pytest

from plugins.memory.palace.provider import PalaceMemoryProvider


class TestSessionRuleRefinement:
    """_refine_session_to_memory_rules should distill preference/decision memories into Memory rules."""

    def _make_provider(self):
        p = PalaceMemoryProvider(config={
            "auto_extract": True,
            "session_extract": True,
            "min_confidence": 0.3,
        })
        p._session_id = "test-session"
        p._agent_context = "primary"
        return p

    def test_preference_high_importance_generates_rule(self):
        """preference + importance >= 4.0 → [RULE] written to Memory."""
        written = []
        provider = self._make_provider()
        provider._write_memory = lambda target, content: written.append((target, content))

        memories = [
            {"content": "User prefers concise responses without markdown", "memory_type": "preference", "confidence": 0.9},
        ]
        # importance = min(5.0, 0.9 * 5) = 4.5
        provider._refine_session_to_memory_rules(memories)

        assert len(written) == 1
        target, content = written[0]
        assert target == "memory"
        assert "[RULE]" in content

    def test_preference_low_importance_no_rule(self):
        """preference + importance < 4.0 → no [RULE]."""
        written = []
        provider = self._make_provider()
        provider._write_memory = lambda target, content: written.append((target, content))

        memories = [
            {"content": "User said they might like shorter answers", "memory_type": "preference", "confidence": 0.5},
        ]
        # importance = min(5.0, 0.5 * 5) = 2.5
        provider._refine_session_to_memory_rules(memories)

        assert len(written) == 0

    def test_decision_high_importance_generates_index(self):
        """decision + importance >= 4.5 → [INDEX] written to Memory."""
        written = []
        provider = self._make_provider()
        provider._write_memory = lambda target, content: written.append((target, content))

        memories = [
            {"content": "Project decided to use PostgreSQL for all new services", "memory_type": "decision", "confidence": 0.95},
        ]
        # importance = min(5.0, 0.95 * 5) = 4.75
        provider._refine_session_to_memory_rules(memories)

        assert len(written) == 1
        target, content = written[0]
        assert target == "memory"
        assert "[INDEX]" in content

    def test_decision_medium_importance_no_index(self):
        """decision + importance < 4.5 → no [INDEX]."""
        written = []
        provider = self._make_provider()
        provider._write_memory = lambda target, content: written.append((target, content))

        memories = [
            {"content": "Minor decision about color scheme", "memory_type": "decision", "confidence": 0.7},
        ]
        # importance = min(5.0, 0.7 * 5) = 3.5
        provider._refine_session_to_memory_rules(memories)

        assert len(written) == 0

    def test_general_memory_no_rule(self):
        """general memory_type → no [RULE] or [INDEX]."""
        written = []
        provider = self._make_provider()
        provider._write_memory = lambda target, content: written.append((target, content))

        memories = [
            {"content": "Some general fact about the project", "memory_type": "general", "confidence": 0.95},
        ]
        provider._refine_session_to_memory_rules(memories)

        assert len(written) == 0

    def test_no_callback_does_not_crash(self):
        """If _write_memory is None, should silently skip."""
        provider = self._make_provider()
        # _write_memory is None by default
        memories = [
            {"content": "User prefers concise responses", "memory_type": "preference", "confidence": 0.9},
        ]
        provider._refine_session_to_memory_rules(memories)
        # Should not raise

    def test_mixed_memories_correct_filtering(self):
        """Mix of preference, decision, and general → only qualifying ones trigger."""
        written = []
        provider = self._make_provider()
        provider._write_memory = lambda target, content: written.append((target, content))

        memories = [
            {"content": "User prefers dark mode", "memory_type": "preference", "confidence": 0.85},  # imp=4.25 → RULE
            {"content": "Low conf preference", "memory_type": "preference", "confidence": 0.5},  # imp=2.5 → skip
            {"content": "Use Redis for caching", "memory_type": "decision", "confidence": 0.95},  # imp=4.75 → INDEX
            {"content": "Minor general note", "memory_type": "general", "confidence": 0.95},  # → skip
            {"content": "Medium decision", "memory_type": "decision", "confidence": 0.8},  # imp=4.0 → skip (<4.5)
        ]
        provider._refine_session_to_memory_rules(memories)

        assert len(written) == 2
        tags = [c[1].split("]")[0] + "]" for c in written]
        assert "[RULE]" in tags
        assert "[INDEX]" in tags
