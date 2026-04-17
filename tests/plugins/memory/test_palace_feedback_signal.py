"""Tests for Palace feedback signal detection (Phase 2A)."""

import pytest

from plugins.memory.palace.queue import detect_feedback_signal


class TestDetectFeedbackSignal:
    def test_returns_none_for_empty_text(self):
        assert detect_feedback_signal("") is None
        assert detect_feedback_signal("   ") is None

    def test_detects_negative_correction_signal(self):
        # Strong negative phrase
        signal = detect_feedback_signal("你这个说法不对，完全错了")
        assert signal is not None
        assert signal < 0

    def test_detects_positive_affirmation_signal(self):
        signal = detect_feedback_signal("没错，就是这样，正确")
        assert signal is not None
        assert signal > 0

    def test_selects_strongest_signal_when_multiple_patterns_match(self):
        # Contains both weak and strong negative cues; should return strongest by abs value
        signal = detect_feedback_signal("不是这样，你错了")
        # plan target: "错了"=-0.7 stronger than "不是"=-0.3
        assert signal == pytest.approx(-0.7)

    def test_returns_none_when_no_feedback_keywords(self):
        assert detect_feedback_signal("今天北京天气怎么样") is None

    def test_english_feedback_detection(self):
        neg = detect_feedback_signal("that's wrong, actually")
        pos = detect_feedback_signal("exactly, that's right")
        assert neg is not None and neg < 0
        assert pos is not None and pos > 0
