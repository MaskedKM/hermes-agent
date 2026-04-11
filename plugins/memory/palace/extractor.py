"""PalaceExtractor — regex-based memory extraction with 5 marker types.

Ported from MemPalace's general_extractor.py with 5 marker types totaling 111 regex patterns.
Pure stdlib (re, logging) — zero external dependencies.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. DECISION_MARKERS — 21 patterns
# ---------------------------------------------------------------------------
DECISION_MARKERS = [
    r"\blet'?s (use|go with|try|pick|choose|switch to)\b",
    r"\bwe (should|decided|chose|went with|picked|settled on)\b",
    r"\bi'?m going (to|with)\b",
    r"\bbetter (to|than|approach|option|choice)\b",
    r"\binstead of\b",
    r"\brather than\b",
    r"\bthe reason (is|was|being)\b",
    r"\bbecause\b",
    r"\btrade-?off\b",
    r"\bpros and cons\b",
    r"\bover\b.*\bbecause\b",
    r"\barchitecture\b",
    r"\bapproach\b",
    r"\bstrategy\b",
    r"\bpattern\b",
    r"\bstack\b",
    r"\bframework\b",
    r"\bset (it |this )?to\b",
    r"\bconfigure\b",
    r"\bdefault\b",
]

# ---------------------------------------------------------------------------
# 2. PREFERENCE_MARKERS — 17 patterns
# ---------------------------------------------------------------------------
PREFERENCE_MARKERS = [
    r"\bi prefer\b",
    r"\balways use\b",
    r"\bnever use\b",
    r"\bdon'?t (ever |like to )?(use|do|mock|stub|import)\b",
    r"\bi like (to|when|how)\b",
    r"\bi hate (when|how|it when)\b",
    r"\bplease (always|never|don'?t)\b",
    r"\bmy (rule|preference|style|convention) is\b",
    r"\bwe (always|never)\b",
    r"\bfunctional\b.*\bstyle\b",
    r"\bimperative\b",
    r"\bsnake_?case\b",
    r"\bcamel_?case\b",
    r"\btabs\b.*\bspaces\b",
    r"\bspaces\b.*\btabs\b",
    r"\buse\b.*\binstead of\b",
]

# ---------------------------------------------------------------------------
# 3. MILESTONE_MARKERS — 35 patterns
# ---------------------------------------------------------------------------
MILESTONE_MARKERS = [
    r"\bit works\b",
    r"\bit worked\b",
    r"\bgot it working\b",
    r"\bfixed\b",
    r"\bsolved\b",
    r"\bbreakthrough\b",
    r"\bfigured (it )?out\b",
    r"\bnailed it\b",
    r"\bcracked (it|the)\b",
    r"\bfinally\b",
    r"\bfirst time\b",
    r"\bfirst ever\b",
    r"\bnever (done|been|had) before\b",
    r"\bdiscovered\b",
    r"\brealized\b",
    r"\bfound (out|that)\b",
    r"\bturns out\b",
    r"\bthe key (is|was|insight)\b",
    r"\bthe trick (is|was)\b",
    r"\bnow i (understand|see|get it)\b",
    r"\bbuilt\b",
    r"\bcreated\b",
    r"\bimplemented\b",
    r"\bshipped\b",
    r"\blaunched\b",
    r"\bdeployed\b",
    r"\breleased\b",
    r"\bprototype\b",
    r"\bproof of concept\b",
    r"\bdemo\b",
    r"\bversion \d",
    r"\bv\d+\.\d+",
    r"\d+x (compression|faster|slower|better|improvement|reduction)",
    r"\d+% (reduction|improvement|faster|better|smaller)",
]

# ---------------------------------------------------------------------------
# 4. PROBLEM_MARKERS — 18 patterns
# ---------------------------------------------------------------------------
PROBLEM_MARKERS = [
    r"\b(bug|error|crash|fail|broke|broken|issue|problem)\b",
    r"\bdoesn'?t work\b",
    r"\bnot working\b",
    r"\bwon'?t\b.*\bwork\b",
    r"\bkeeps? (failing|crashing|breaking|erroring)\b",
    r"\broot cause\b",
    r"\bthe (problem|issue|bug) (is|was)\b",
    r"\bturns out\b.*\b(was|because|due to)\b",
    r"\bthe fix (is|was)\b",
    r"\bworkaround\b",
    r"\bthat'?s why\b",
    r"\bthe reason it\b",
    r"\bfixed (it |the |by )\b",
    r"\bsolution (is|was)\b",
    r"\bresolved\b",
    r"\bpatched\b",
    r"\bthe answer (is|was)\b",
    r"\b(had|need) to\b.*\binstead\b",
]

# ---------------------------------------------------------------------------
# 5. EMOTION_MARKERS — 26 patterns
# ---------------------------------------------------------------------------
EMOTION_MARKERS = [
    r"\blove\b",
    r"\bscared\b",
    r"\bafraid\b",
    r"\bproud\b",
    r"\bhurt\b",
    r"\bhappy\b",
    r"\bsad\b",
    r"\bcry\b",
    r"\bcrying\b",
    r"\bmiss\b",
    r"\bsorry\b",
    r"\bgrateful\b",
    r"\bangry\b",
    r"\bfrustrat",
    r"\bexcited\b",
    r"\brelieved\b",
    r"\banxious\b",
    r"\bworried\b",
    r"\bhopeful\b",
    r"\bdisappointed\b",
    r"\boverwhelm",
    r"\bsatisf",
    r"\benjoy",
    r"\bfurious\b",
    r"\bthrilled\b",
    r"\btired\b",
]

# ---------------------------------------------------------------------------
# 6. ZH_DECISION_MARKERS — Chinese decision patterns
# ---------------------------------------------------------------------------
ZH_DECISION_MARKERS = [
    r"决定(?:使用|采用|选择|用|换成|改为)",
    r"我们(?:用|选择|决定用|改成)",
    r"(?:最好|应该|最好)用",
    r"改为",
    r"换成",
    r"架构(?:是|用|采用|选择)",
    r"方案(?:是|确定|选择)",
    r"策略(?:是|确定)",
    r"配置(?:为|成|了)",
    r"设置(?:为|成|了)",
    r"默认(?:用|使用|是)",
    r"不用.*(?:而|改)用",
    r"不用.*(?:而|改)用",
]

# ---------------------------------------------------------------------------
# 7. ZH_PREFERENCE_MARKERS — Chinese preference patterns
# ---------------------------------------------------------------------------
ZH_PREFERENCE_MARKERS = [
    r"(?:我|你)(?:偏好|喜欢|习惯|偏爱)",
    r"(?:总是|一直|永远)(?:用|使用|选择)",
    r"(?:不要|别|千万别|绝对不)(?:用|使用|做)",
    r"(?:我喜欢|不喜欢|讨厌)(?:.*(?:的|地))?(?:风格|方式|做法)",
    r"请(?:总是|永远|一定)(?:不要|别|别忘)",
    r"(?:我的|你的)(?:规则|偏好|风格|习惯|约定)是",
    r"(?:我们|大家)(?:总是|从不|一律)",
    r"风格(?:是|偏好)",
    r"(?:习惯|偏好)(?:用|使用|是)",
]

# ---------------------------------------------------------------------------
# 8. ZH_MILESTONE_MARKERS — Chinese milestone patterns
# ---------------------------------------------------------------------------
ZH_MILESTONE_MARKERS = [
    r"(?:搞定了|成功了|搞好了|弄好了|搞通了|搞懂了)",
    r"(?:修复了|解决了|修好了|搞定|处理了)",
    r"(?:终于|总算|好歹|好不容易)",
    r"(?:第一次|首次)",
    r"(?:发现了|找到了|意识到|注意到|明白了)",
    r"(?:原来|原来是|其实是|实际上是)",
    r"(?:关键|窍门|秘诀|核心)(?:是|在于|就是)",
    r"(?:构建|搭建|创建|开发|实现)了",
    r"(?:部署|上线|发布|发布|完成)了",
    r"(?:突破|进展|里程碑)",
    r"(?:验证|测试)通过",
    r"(?:工作|运行)(?:正常了|成功了)",
    r"(?:搞定|完成|成功)(?:了|！)",
]

# ---------------------------------------------------------------------------
# 9. ZH_PROBLEM_MARKERS — Chinese problem patterns
# ---------------------------------------------------------------------------
ZH_PROBLEM_MARKERS = [
    r"(?:bug|错误|崩溃|失败|报错|异常|问题)",
    r"(?:不工作|不能用|无法|跑不通|出错了)",
    r"(?:一直|总是|反复|经常)(?:失败|报错|崩溃|出错)",
    r"(?:根因|根本原因|原因)(?:是|在于|就是)",
    r"(?:问题|bug|错误)(?:是|出在|在于)",
    r"(?:原因是|因为|由于)",
    r"(?:解决|修复)(?:方案|办法|方法是|了)",
    r"(?:workaround|临时方案|绕过)",
    r"(?:原因|所以)(?:才|就)",
    r"(?:需要|必须|不得不)(?:.*(?:才|就|改为))",
]

# ---------------------------------------------------------------------------
# 10. ZH_EMOTION_MARKERS — Chinese emotion patterns
# ---------------------------------------------------------------------------
ZH_EMOTION_MARKERS = [
    r"(?:喜欢|讨厌|热爱|厌恶|嫌弃)",
    r"(?:开心|高兴|快乐|兴奋|激动)",
    r"(?:难过|伤心|沮丧|焦虑|紧张|担心)",
    r"(?:生气|愤怒|崩溃|烦躁|无语)",
    r"(?:满意|满意|欣慰|松了口气|庆幸)",
    r"(?:累了|疲惫|困|无聊|无奈)",
    r"(?:期待|希望|怕|害怕|恐惧)",
    r"(?:感动|感恩|抱歉|愧疚)",
]

# ---------------------------------------------------------------------------
# Marker registry & compiled patterns
# ---------------------------------------------------------------------------
# English markers (compiled separately)
EN_MARKERS = {
    "decision": DECISION_MARKERS,
    "preference": PREFERENCE_MARKERS,
    "milestone": MILESTONE_MARKERS,
    "problem": PROBLEM_MARKERS,
    "emotional": EMOTION_MARKERS,
}

# Chinese markers (compiled separately)
ZH_MARKERS = {
    "decision": ZH_DECISION_MARKERS,
    "preference": ZH_PREFERENCE_MARKERS,
    "milestone": ZH_MILESTONE_MARKERS,
    "problem": ZH_PROBLEM_MARKERS,
    "emotional": ZH_EMOTION_MARKERS,
}

# Combined for backward compatibility
ALL_MARKERS = {
    mtype: EN_MARKERS[mtype] + ZH_MARKERS[mtype]
    for mtype in EN_MARKERS
}

# Pre-compile: {lang_type: [(compiled_re, raw_pattern), ...]}
COMPILED_MARKERS: dict[str, list[tuple[re.Pattern, str]]] = {}
for _mtype, _patterns in ALL_MARKERS.items():
    COMPILED_MARKERS[_mtype] = [(re.compile(p, re.IGNORECASE), p) for p in _patterns]

# Pre-compile per-language for independent scoring
COMPILED_EN: dict[str, list[tuple[re.Pattern, str]]] = {}
for _mtype, _patterns in EN_MARKERS.items():
    COMPILED_EN[_mtype] = [(re.compile(p, re.IGNORECASE), p) for p in _patterns]

COMPILED_ZH: dict[str, list[tuple[re.Pattern, str]]] = {}
for _mtype, _patterns in ZH_MARKERS.items():
    COMPILED_ZH[_mtype] = [(re.compile(p, re.IGNORECASE), p) for p in _patterns]

# Tiebreak order: when multiple types have the same match count,
# prefer the type that appears earlier in this list.
_TIEBREAK_ORDER = [
    "decision", "preference", "milestone", "problem", "emotional",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_prose(text: str) -> str:
    """Remove code blocks and heavily indented lines.

    Rules:
    - Remove ``` fenced code blocks entirely
    - Remove lines starting with 4+ spaces
    - Remove lines matching Python syntax: import/from/class/def/@/return/raise/except/try/if __
    - Keep everything else
    """
    # Remove fenced code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)

    lines = text.split("\n")
    filtered: list[str] = []

    # Lines that look like code syntax
    code_prefixes = re.compile(
        r"^\s*(import |from |class |def |@|return |raise |except |try:|if __)"
    )

    for line in lines:
        # Skip heavily indented lines (4+ leading spaces)
        if line.startswith("    "):
            continue
        # Skip lines starting with Python syntax
        if code_prefixes.match(line):
            continue
        filtered.append(line)

    return "\n".join(filtered)


def _split_sentences(text: str) -> list[str]:
    """Split filtered text into sentences.

    Split on: '. ' + newline + '! ' + '? ' + '; '
    Also split on Chinese punctuation: '。！？；' followed by optional space
    Also split on double newlines (paragraph boundaries).
    """
    # Normalize paragraph boundaries to sentence splits
    text = re.sub(r"\n{2,}", ". ", text)
    # Normalize single newlines to spaces (join wrapped lines)
    text = text.replace("\n", " ")

    # Split on sentence-ending punctuation followed by space or end
    # Supports both English (.!?;) and Chinese (。！？；) sentence enders
    parts = re.split(r"(?<=[.!?;。！？；])\s*", text)

    # Filter out empty / whitespace-only parts
    return [p.strip() for p in parts if p.strip()]


def _score_sentence(sentence: str) -> tuple[str, float, list[str]]:
    """Score a sentence against all marker types.

    Scores EN and ZH patterns independently to avoid dilution bias.
    Uses matched count (not ratio) for type ranking so a sentence that
    matches 2 decision markers beats one that matches 1 problem marker,
    regardless of pool sizes.

    Returns (memory_type, confidence, matched_patterns).
    Confidence = min(1.0, matched_count / 3.0).
    """
    best_type = "decision"
    best_count = 0
    best_matched: list[str] = []
    best_tiebreak = len(_TIEBREAK_ORDER)  # worst possible

    # Score each language group independently
    for compiled_group in (COMPILED_EN, COMPILED_ZH):
        for mtype, compiled_list in compiled_group.items():
            if not compiled_list:
                continue
            matched: list[str] = []
            for compiled_re, raw_pattern in compiled_list:
                if compiled_re.search(sentence):
                    matched.append(raw_pattern)
            if not matched:
                continue
            # Use count (not ratio) so the type with more actual matches wins
            # On tie, prefer types earlier in _TIEBREAK_ORDER
            count = len(matched)
            tiebreak = _TIEBREAK_ORDER.index(mtype) if mtype in _TIEBREAK_ORDER else len(_TIEBREAK_ORDER)
            if count > best_count or (count == best_count and tiebreak < best_tiebreak):
                best_count = count
                best_type = mtype
                best_matched = matched
                best_tiebreak = tiebreak

    # Confidence: 1 match → 0.33, 2 matches → 0.67, 3+ → 1.0
    confidence = min(1.0, len(best_matched) / 3.0)

    return best_type, confidence, best_matched


def _disambiguate(memory_type: str, sentence: str) -> str:
    """Disambiguate memory type.

    If problem markers match but sentence contains 'solved/resolved/fixed/workaround'
    → reclassify as milestone. Supports both English and Chinese.
    """
    if memory_type != "problem":
        return memory_type

    resolution_words = [
        "solved", "resolved", "fixed", "workaround",
        "修复了", "解决了", "搞定了", "修好了", "处理了", "搞定",
        "修复方法", "解决办法", "解决方案",
    ]
    lower = sentence.lower()
    for word in resolution_words:
        if word in lower:
            return "milestone"

    return memory_type


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_memories(text: str, min_confidence: float = 0.3) -> list[dict]:
    """Extract structured memories from text.

    Process:
    1. Filter out code lines (lines starting with >4 spaces, or inside ``` fences)
    2. Split into sentences (rough split on . ! ? newline)
    3. For each sentence, check against all marker types
    4. Score = matched_patterns / total_patterns_per_type
    5. Disambiguate: if problem markers match and sentence contains 'solved/resolved/fixed' → milestone
    6. Confidence = min(1.0, max_score / 5.0) -- from MemPalace source
    7. Filter by min_confidence

    Returns list of dicts:
    [{
        'content': str,            # the extracted sentence/paragraph
        'memory_type': str,        # decision/preference/milestone/problem/emotional
        'confidence': float,       # 0.0-1.0
        'matched_patterns': list,  # matched regex pattern strings
    }]
    """
    if not text or not text.strip():
        return []

    # Step 1: filter prose
    prose = _filter_prose(text)
    if not prose.strip():
        return []

    # Step 2: split into sentences
    sentences = _split_sentences(prose)
    if not sentences:
        return []

    # Steps 3-7: score, disambiguate, filter
    results: list[dict] = []
    for sentence in sentences:
        memory_type, confidence, matched = _score_sentence(sentence)
        if not matched:
            continue

        # Disambiguate
        memory_type = _disambiguate(memory_type, sentence)

        if confidence < min_confidence:
            continue

        results.append({
            "content": sentence,
            "memory_type": memory_type,
            "confidence": round(confidence, 4),
            "matched_patterns": matched,
        })

    return results


# ---------------------------------------------------------------------------
# Entity extraction for co-occurrence groups
# ---------------------------------------------------------------------------

# Stopwords — common words that don't carry entity semantics
_EN_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "but", "and", "or",
    "if", "while", "that", "this", "these", "those", "it", "its", "i",
    "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
    "what", "which", "who", "whom", "up", "about", "also", "get", "got",
    "much", "many", "like", "well", "even", "still", "already", "really",
    "now", "new", "old", "good", "bad", "first", "last", "next", "way",
    "use", "used", "using", "make", "made", "know", "think", "see", "say",
    "said", "go", "went", "come", "came", "take", "took", "want", "need",
    "try", "tried", "thing", "things", "something", "anything", "everything",
    "nothing", "part", "lot", "bit", "point", "case", "time", "work",
    "works", "working", "don", "didn", "doesn", "isn", "wasn", "won",
    "wouldn", "couldn", "shouldn", "haven", "hasn", "hadn", "aren",
})

_ZH_STOPWORDS = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
    "们", "那", "个", "来", "吗", "吧", "呢", "啊", "哦", "嗯",
    "把", "被", "让", "给", "从", "对", "但", "而", "又", "还",
    "只", "已", "与", "及", "或", "中", "下", "里", "为", "以",
    "之", "等", "能", "可以", "可", "应该", "该", "得", "地",
    "所", "于", "其", "这个", "那个", "什么", "怎么", "如何", "为什么",
    "因为", "所以", "但是", "而且", "不过", "如果", "虽然", "还是",
    "不是", "没有", "就是", "也是", "已经", "可能", "一些", "比较",
    "比较", "非常", "特别", "真的", "确实", "其实", "然后", "之后",
    "之前", "同时", "以及", "通过", "根据", "关于", "进行", "使用",
    "包括", "需要", "没有", "问题", "时候", "这样", "那样", "怎么",
})

_ALL_STOPWORDS = _EN_STOPWORDS | _ZH_STOPWORDS

# Regex patterns for entity extraction
# Matches quoted terms, camelCase, snake_case, and capitalized multi-word phrases
_ENTITY_PATTERNS = [
    # Quoted terms: "foo bar" or 'foo bar'
    re.compile(r'["\']([^"\']{2,60})["\']'),
    # CamelCase identifiers (2+ words)
    re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b'),
    # snake_case identifiers (2+ parts, min 2 chars each)
    re.compile(r'\b([a-z][a-z0-9]*_(?:[a-z][a-z0-9]*_?)+)\b'),
    # File/URL-like paths
    re.compile(r'\b([A-Za-z][\w./-]{3,80})\.(?:py|js|ts|yaml|yml|json|toml|md|txt|cfg|ini|sh|sql)\b'),
    # Config keys or env vars
    re.compile(r'\b([A-Z][A-Z0-9_]{2,})\b'),
]


def extract_entities(text: str, min_length: int = 2, max_entities: int = 30) -> list[str]:
    """Extract named entities from text for co-occurrence grouping.

    Uses regex patterns to find potential entities (quoted terms, identifiers,
    capitalized phrases, file paths, env vars) then filters against stopwords.

    Args:
        text: Input text to extract entities from.
        min_length: Minimum entity length in characters (default 2).
        max_entities: Maximum number of entities to return (default 30).

    Returns:
        Deduplicated list of entity names (lowercase), in order of first appearance.
    """
    if not text or not text.strip():
        return []

    seen: dict[str, None] = {}  # preserve insertion order
    for pattern in _ENTITY_PATTERNS:
        for match in pattern.finditer(text):
            entity = match.group(1).strip()
            entity_lower = entity.lower()
            # Filter: min length, not a stopword, not pure numbers
            if len(entity_lower) < min_length:
                continue
            if entity_lower in _ALL_STOPWORDS:
                continue
            if entity_lower.isdigit():
                continue
            if entity_lower not in seen:
                seen[entity_lower] = None
            if len(seen) >= max_entities:
                return list(seen.keys())

    return list(seen.keys())
