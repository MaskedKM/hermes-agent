"""LLM-based memory extraction for Palace.

Uses a small/fast LLM via auxiliary_client to extract structured memories
from conversation text. Falls back to regex-based extraction when LLM is
unavailable.

This replaces the pure-regex approach for better semantic understanding,
especially for Chinese text that regex patterns cannot reliably match.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
你是一个记忆提取助手。从对话中提取值得长期记住的信息，并识别其中提到的关键实体。

## 提取原则

1. **只提取事实性、持久性的信息** — 不提取临时状态、对话过程、工具输出
2. **保留原文精华** — 不要改写，保留关键细节（端口号、文件路径、版本号等）
3. **去重合并** — 同类信息合并为一条，不要重复
4. **忽略噪音** — 不提取代码片段、终端输出、系统日志

## 记忆类型

| 类型 | 说明 | 示例 |
|------|------|------|
| decision | 技术决策、架构选择 | "决定使用 PostgreSQL 作为主数据库" |
| preference | 用户偏好、习惯 | "用户偏好中文回复" |
| milestone | 完成的事项、突破 | "修复了 SQLite 线程安全问题" |
| problem | 遇到的问题及原因 | "on_pre_compress 只提取不存储，是忘记调用 add_drawer 导致的" |
| fact | 环境事实、配置信息 | "代理运行在 7890 端口，使用 mihomo" |

## 实体提取

同时提取对话中提到的关键实体词，用于建立知识关联。包括但不限于：
- 人名、项目名、工具名、库名
- 文件名、配置键、环境变量
- 技术概念、协议、服务名
- 中文专有名词

**排除**：常见停用词（的、了、是、the、is 等），过短的词（<2字符）

## 输出格式

严格返回 JSON 对象，不要包含任何其他文字：

```json
{
  "memories": [
    {
      "content": "提取的内容原文",
      "type": "decision|preference|milestone|problem|fact",
      "confidence": 0.8
    }
  ],
  "entities": ["实体A", "实体B", "实体C"]
}
```

- memories: 记忆数组，如果没有则返回空数组
- entities: 实体列表（小写），如果没有则返回空数组
- confidence: 0.0-1.0，表示这条信息的确定性和重要性
- 如果没有任何值得提取的信息，返回 `{"memories": [], "entities": []}`\
"""

EXTRACTION_USER_TEMPLATE = """\
从以下对话中提取值得长期记住的信息。

--- 对话开始 ---
{text}
--- 对话结束 ---

返回 JSON 数组。如果没有值得提取的信息，返回 `[]`。\
"""


# ---------------------------------------------------------------------------
# JSON extraction from LLM response
# ---------------------------------------------------------------------------

_JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _parse_json_response(text: str) -> Any:
    """Extract JSON value from LLM response text.

    Handles: raw JSON, ```json``` fenced blocks, markdown with embedded JSON.
    Returns parsed value (list, dict, etc.) or None.
    """
    text = text.strip()
    if not text:
        return None

    # Try fenced code block first
    m = _JSON_CODE_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Try raw JSON (array or object)
    m = _JSON_ARRAY_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    # Try full text as JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _parse_extraction_response(text: str) -> tuple[Optional[List[Dict[str, Any]]], List[str]]:
    """Parse LLM response into (memories, entities).

    Supports both new format ({"memories": [...], "entities": [...]})
    and legacy format ([...]) for backward compatibility.
    """
    parsed = _parse_json_response(text)
    if parsed is None:
        return None, []

    entities: List[str] = []

    # New format: {"memories": [...], "entities": [...]}
    if isinstance(parsed, dict):
        memories = parsed.get("memories", [])
        raw_entities = parsed.get("entities", [])
        if isinstance(raw_entities, list):
            entities = [
                str(e).strip().lower()
                for e in raw_entities
                if e and str(e).strip() and len(str(e).strip()) >= 2
            ]
        return memories if isinstance(memories, list) else None, entities

    # Legacy format: [...]  (bare array)
    if isinstance(parsed, list):
        return parsed, []

    return None, []


# ---------------------------------------------------------------------------
# Normalize and validate extracted memories
# ---------------------------------------------------------------------------

_VALID_TYPES = frozenset({"decision", "preference", "milestone", "problem", "fact"})


def _normalize_memories(raw: List[Dict[str, Any]], min_confidence: float) -> List[Dict[str, Any]]:
    """Validate and normalize LLM-extracted memories to standard format."""
    results = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        content = item.get("content", "").strip()
        if not content or len(content) < 5:
            continue

        # Normalize type (support both "type" and "memory_type" keys)
        mtype = str(item.get("type") or item.get("memory_type") or "fact").lower().strip()
        if mtype not in _VALID_TYPES:
            mtype = "fact"

        # Normalize confidence
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        if confidence < min_confidence:
            continue

        results.append({
            "content": content,
            "memory_type": mtype,
            "confidence": round(confidence, 4),
            "source": "llm",
        })

    return results


# ---------------------------------------------------------------------------
# Truncate text to fit within context limits
# ---------------------------------------------------------------------------

def _truncate_text(text: str, max_chars: int = 6000) -> str:
    """Truncate text to max_chars, trying to break at sentence boundaries."""
    if len(text) <= max_chars:
        return text

    # Try to break at the last sentence boundary within limit
    truncated = text[:max_chars]
    # Look for last sentence-ending punctuation
    for sep in ("。", "！", "？", ".", "!", "?", "\n"):
        last_sep = truncated.rfind(sep)
        if last_sep > max_chars * 0.5:  # Don't cut more than half
            return truncated[:last_sep + 1].strip()

    return truncated.strip() + "..."


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_memories_llm(
    text: str,
    *,
    min_confidence: float = 0.3,
    max_chars: int = 6000,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Extract memories and entities using LLM.

    Args:
        text: Conversation text to extract from.
        min_confidence: Minimum confidence threshold.
        max_chars: Max characters to send to LLM.

    Returns:
        Tuple of (memories, entities).
        - memories: List of memory dicts with keys: content, memory_type, confidence, source.
        - entities: List of entity strings extracted from the text.

    Raises:
        RuntimeError: If no LLM provider is available.
    """
    from agent.auxiliary_client import call_llm

    truncated = _truncate_text(text, max_chars)

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": EXTRACTION_USER_TEMPLATE.format(text=truncated)},
    ]

    try:
        response = call_llm(
            task="memory_extract",
            messages=messages,
            temperature=0.1,
            max_tokens=2000,
        )
        raw_content = response.choices[0].message.content
        if not isinstance(raw_content, str):
            raw_content = str(raw_content) if raw_content else ""
    except RuntimeError as e:
        raise  # No provider — let caller fall back to regex
    except Exception as e:
        logger.warning("LLM memory extraction failed: %s", e)
        return [], []

    if not raw_content.strip():
        return [], []

    raw_memories, entities = _parse_extraction_response(raw_content)
    if raw_memories is None:
        logger.debug("LLM memory extraction returned non-JSON: %s", raw_content[:200])
        return [], []

    return _normalize_memories(raw_memories, min_confidence), entities


def extract_memories_with_fallback(
    text: str,
    *,
    min_confidence: float = 0.3,
    prefer_llm: bool = True,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Extract memories and entities using LLM, falling back to regex.

    Args:
        text: Conversation text.
        min_confidence: Minimum confidence threshold.
        prefer_llm: If True, try LLM first then regex fallback.

    Returns:
        Tuple of (memories, entities).
        When LLM is used, entities come from LLM extraction.
        When regex fallback is used, entities come from regex extract_entities.
    """
    if prefer_llm:
        try:
            llm_memories, llm_entities = extract_memories_llm(
                text, min_confidence=min_confidence
            )
            if llm_memories:
                logger.debug("LLM extracted %d memories, %d entities", len(llm_memories), len(llm_entities))
                return llm_memories, llm_entities
            # LLM returned no memories but might have entities — still return them
            if llm_entities:
                return [], llm_entities
        except RuntimeError:
            logger.debug("No LLM provider for memory extraction, using regex fallback")
        except Exception as e:
            logger.warning("LLM extraction error, falling back to regex: %s", e)

    # Regex fallback — extract entities via regex
    from .extractor import extract_memories, extract_entities
    memories = extract_memories(text, min_confidence=min_confidence)
    entities = extract_entities(text)
    return memories, entities
