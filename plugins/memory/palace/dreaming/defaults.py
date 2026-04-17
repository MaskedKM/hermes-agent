"""Dreaming configuration defaults."""

# ─── Cold Start Thresholds ───
COLD_START_MIN_RECALL_EVENTS = 30
COLD_START_MIN_DAYS = 3

# ─── Cold Start Scoring Weights ───
COLD_START_WEIGHTS = {
    "intrinsic_quality": 0.50,
    "recency": 0.30,
    "connectivity": 0.10,
    "time_consolidation": 0.10,
}

# ─── Steady State Scoring Weights ───
STEADY_STATE_WEIGHTS = {
    "recall_relevance": 0.25,
    "recall_frequency": 0.20,
    "recall_diversity": 0.12,
    "intrinsic_quality": 0.13,
    "feedback_adjustment": 0.05,
    "recency": 0.13,
    "connectivity": 0.07,
    "time_consolidation": 0.05,
}

# ─── Light Sleep ───
LIGHT_LOOKBACK_DAYS = 2
LIGHT_LIMIT = 200
LIGHT_DEDUP_THRESHOLD = 0.95
LIGHT_MIN_HOURS_BETWEEN_RUNS = 4

# ─── Deep Sleep ───
DEEP_LOOKBACK_DAYS = 14
DEEP_MIN_SCORE = 0.60
DEEP_MAX_PROMOTIONS = 10
DEEP_LLM_CONSOLIDATE = False  # Optional: use LLM to polish top promoted memories
DEEP_LLM_CONSOLIDATE_TOP_N = 3

# ─── REM Sleep ───
REM_MIN_COOCCURRENCE_GROUPS = 10
REM_MIN_PATTERN_FREQUENCY = 3
REM_MAX_PATTERNS = 20

# ─── Cold Start Archive Rules (conservative) ───
COLD_START_ARCHIVE_RULES = {
    "min_age_days": 60,
    "max_score_threshold": 0.15,
    "require_zero_links": True,
    "require_zero_recalls": True,
    "max_archived_per_run": 3,
}

# ─── Steady State Archive Rules ───
STEADY_STATE_ARCHIVE_RULES = {
    "min_age_days": 30,
    "max_score_threshold": 0.30,
    "require_zero_links": False,
    "require_zero_recalls": False,
    "max_archived_per_run": 10,
}

# ─── Scoring Parameters ───
RECENCY_HALF_LIFE_DAYS = 14
CONNECTIVITY_MAX_LINKS = 6
TIME_CONSOLIDATION_MAX_DAYS = 30
MAX_EXPECTED_DRAWERS = 500

# ─── Recovery ───
RECOVERY_TRIGGER_BELOW_HEALTH = 0.35
RECOVERY_LOOKBACK_DAYS = 30
RECOVERY_MAX_CANDIDATES = 20
RECOVERY_AUTO_WRITE_MIN_CONFIDENCE = 0.90

# ─── Recall Events Cleanup ───
RECALL_EVENTS_MAX_AGE_DAYS = 90
