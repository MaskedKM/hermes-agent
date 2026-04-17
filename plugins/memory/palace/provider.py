"""PalaceMemoryProvider — MemoryProvider implementation for the Memory Palace plugin.

Exposes 4 tools (palace_store, palace_recall, palace_mine, palace_graph) plus
hooks for system prompt injection, prefetch, auto-extraction, session-end
batching, built-in memory mirroring, and pre-compression extraction.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    palace:
      auto_extract: true
      session_extract: true
      min_confidence: 0.3
      mirror_builtin: true
      db_path: $HERMES_HOME/palace/drawers.db
      kg_path: $HERMES_HOME/palace/knowledge_graph.db
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from .store import PalaceStore
from .extractor import extract_memories as _extract_memories_regex
from .extractor import extract_entities as _extract_entities
from .llm_extractor import extract_memories_with_fallback
from .queue import ExtractionQueue, detect_correction, detect_feedback_signal
from .knowledge_graph import PalaceKnowledgeGraph
from .miner import mine_project, mine_sessions, mine_status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory type → wing/room mapping
# ---------------------------------------------------------------------------

_MEMORY_TYPE_WING = {
    "decision": "decisions",
    "preference": "user",
    "milestone": "project",
    "problem": "problems",
    "emotional": "user",
}

_MEMORY_TYPE_ROOM = {
    "decision": "architecture",
    "preference": "preferences",
    "milestone": "achievements",
    "problem": "debugging",
    "emotional": "feelings",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}
        return all_config.get("plugins", {}).get("palace", {}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PALACE_STORE_SCHEMA = {
    "name": "palace_store",
    "description": (
        "Store knowledge in the Memory Palace — a hierarchical, searchable long-term memory.\n\n"
        "The Palace organizes knowledge into wings (broad categories) and rooms (specific topics).\n"
        "Use this to save facts, decisions, discoveries, and context for future sessions.\n\n"
        "ACTIONS:\n"
        "• add — Store a piece of knowledge. Specify wing + room for organization.\n"
        "• get — Retrieve a specific drawer by ID.\n"
        "• delete — Remove a drawer.\n"
        "• list — List drawers, optionally filtered by wing/room.\n"
        "• duplicate_check — Check if similar content already exists.\n\n"
        "WHEN TO USE:\n"
        "• User shares preferences, decisions, or facts worth remembering\n"
        "• Important discoveries or solutions during debugging\n"
        "• Project architecture decisions or configuration choices\n"
        "• Any information that would be useful in future sessions"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "get", "delete", "list", "duplicate_check"],
            },
            "content": {"type": "string", "description": "Knowledge content to store (required for 'add')."},
            "wing": {"type": "string", "description": "Wing (broad category): user, project, tech, decisions, problems, general."},
            "room": {"type": "string", "description": "Room (specific topic within wing), e.g. 'python', 'deploy', 'api'."},
            "hall": {"type": "string", "description": "Hall (optional sub-room for finer grouping)."},
            "importance": {"type": "number", "description": "Importance 1-5 (default 3). Higher = more prominent in recall."},
            "drawer_id": {"type": "string", "description": "Drawer ID (required for 'get'/'delete')."},
            "limit": {"type": "integer", "description": "Max results for 'list' (default 20)."},
            "offset": {"type": "integer", "description": "Offset for pagination (default 0)."},
        },
        "required": ["action"],
    },
}

PALACE_RECALL_SCHEMA = {
    "name": "palace_recall",
    "description": (
        "Search and recall knowledge from the Memory Palace.\n\n"
        "ACTIONS:\n"
        "• search — Full-text search across all drawers. Returns ranked results.\n"
        "• top — Get the highest-importance drawers (L1 Essential Story).\n"
        "• map — View the palace structure (wings → rooms → counts).\n\n"
        "Use before answering questions about past decisions, preferences, or project context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "top", "map"]},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "wing": {"type": "string", "description": "Filter to specific wing."},
            "room": {"type": "string", "description": "Filter to specific room."},
            "limit": {"type": "integer", "description": "Max results (default 5 for search, 8 for top)."},
        },
        "required": ["action"],
    },
}

PALACE_MINE_SCHEMA = {
    "name": "palace_mine",
    "description": (
        "Mine knowledge from project files or past sessions into the Memory Palace.\n\n"
        "ACTIONS:\n"
        "• project — Scan a project directory, chunk files, store as drawers.\n"
        "• sessions — Mine recent Hermes sessions for Q&A pairs.\n"
        "• status — Show mining statistics.\n\n"
        "Mining is incremental — already-mined sources are skipped unless modified."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["project", "sessions", "status"]},
            "project_path": {"type": "string", "description": "Project directory path (for 'project' action)."},
            "wing": {"type": "string", "description": "Wing to store under (default: 'project' or 'hermes_sessions')."},
            "max_sessions": {"type": "integer", "description": "Max sessions to mine (default 10)."},
        },
        "required": ["action"],
    },
}

PALACE_GRAPH_SCHEMA = {
    "name": "palace_graph",
    "description": (
        "Temporal knowledge graph for tracking entity relationships over time.\n\n"
        "ACTIONS:\n"
        "• add_entity — Register an entity (person, project, tool, etc.).\n"
        "• add_triple — Add a relationship: subject → predicate → object.\n"
        "• query — Query relationships for an entity (outgoing/incoming/both).\n"
        "• timeline — Get full timeline for an entity.\n"
        "• search — Search entities by name.\n"
        "• register_alias — Add an alias for entity disambiguation.\n"
        "• resolve — Resolve a name/alias to canonical entity name.\n"
        "• stats — Graph statistics.\n"
        "• registry — View entity registry for disambiguation.\n\n"
        "Use for tracking who worked on what, project dependencies, tool relationships."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add_entity", "add_triple", "query", "timeline", "search",
                         "register_alias", "resolve", "stats", "registry"],
            },
            "name": {"type": "string", "description": "Entity name."},
            "entity_type": {"type": "string", "description": "Entity type: person, project, tool, concept, etc."},
            "subject": {"type": "string", "description": "Subject entity (for add_triple/query)."},
            "predicate": {"type": "string", "description": "Relationship type (for add_triple/query): uses, works_on, depends_on, prefers, etc."},
            "object": {"type": "string", "description": "Object entity (for add_triple)."},
            "direction": {"type": "string", "description": "Query direction: outgoing, incoming, both (default: both)."},
            "alias": {"type": "string", "description": "Alias to register (for register_alias)."},
            "query": {"type": "string", "description": "Search query (for search action)."},
            "limit": {"type": "integer", "description": "Max results (default 10)."},
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Entity/triple extraction patterns for on_session_end
# ---------------------------------------------------------------------------

_TRIPLE_PATTERNS = [
    re.compile(r"(\w[\w\s]+?)\s+(uses?|used|prefer|prefers|works? on|worked on|depends? on|depend on)\s+(\w[\w\s]+)", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class PalaceMemoryProvider(MemoryProvider):
    """Memory Palace provider — hierarchical knowledge storage with FTS5 search + knowledge graph."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store: PalaceStore | None = None
        self._kg: PalaceKnowledgeGraph | None = None
        self._session_id: str | None = None
        self._hermes_home: str | None = None
        self._agent_context: str = "primary"
        self._extract_queue: ExtractionQueue | None = None
        self._write_memory: Callable[[str, str], None] | None = None  # injected by memory_manager

    # -- Core lifecycle -------------------------------------------------------

    @property
    def name(self) -> str:
        return "palace"

    def is_available(self) -> bool:
        return True  # SQLite is always available

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home", "")
        self._hermes_home = hermes_home
        self._session_id = session_id
        self._agent_context = kwargs.get("agent_context", "primary")

        # Resolve DB paths — expand $HERMES_HOME if present
        from hermes_constants import get_hermes_home as _ghh
        _home = hermes_home or str(_ghh())
        default_db = str(Path(_home) / "palace" / "drawers.db")
        default_kg = str(Path(_home) / "palace" / "knowledge_graph.db")

        db_path = self._config.get("db_path", default_db)
        kg_path = self._config.get("kg_path", default_kg)

        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)
        if isinstance(kg_path, str):
            kg_path = kg_path.replace("$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)

        self._store = PalaceStore(db_path)
        self._kg = PalaceKnowledgeGraph(kg_path)

        # Vector store (Phase 1: hybrid search)
        self._vector_store = None
        emb_config = self._config.get("embedding") or {}
        if emb_config.get("enabled", True):
            # Derive API key from zai/vision config if not explicitly set
            if not emb_config.get("api_key"):
                # Try to get from auxiliary.vision or providers.zai config (same Zhipu API)
                try:
                    from hermes_constants import get_hermes_home
                    import yaml
                    cfg_path = get_hermes_home() / "config.yaml"
                    with open(cfg_path) as f:
                        full = yaml.safe_load(f) or {}
                    for path in (("auxiliary", "vision"), ("providers", "zai")):
                        section = full
                        for key in path:
                            section = section.get(key, {}) if isinstance(section, dict) else {}
                        api_key = section.get("api_key") if isinstance(section, dict) else None
                        if api_key:
                            emb_config["api_key"] = api_key
                            break
                except Exception:
                    pass
            if emb_config.get("api_key"):
                try:
                    from .embedding_client import EmbeddingClient
                    from .vector_store import VectorStore
                    client = EmbeddingClient(emb_config)
                    dim = int(emb_config.get("dimension", 2048))
                    self._vector_store = VectorStore(
                        conn=self._store.conn,
                        embedding_client=client,
                        dimension=dim,
                    )
                    emb_count = self._vector_store.get_embedding_count()
                    logger.info("VectorStore initialized (dim=%d, embeddings=%d)",
                                dim, emb_count)
                except Exception as e:
                    logger.warning("VectorStore init failed (fts-only mode): %s", e)
                    self._vector_store = None
            else:
                logger.info("No embedding API key configured — hybrid search disabled")

        # Debounce queue for LLM extraction
        extract_mode = self._config.get("extract_mode", "regex")
        debounce_seconds = float(self._config.get("debounce_seconds", 30))
        if extract_mode in ("llm", "hybrid"):
            self._extract_queue = ExtractionQueue(
                debounce_seconds=debounce_seconds,
                extract_fn=extract_memories_with_fallback,
                store_fn=self._store_debounced_memories,
            )

        logger.info("Memory Palace initialized (session=%s, extract_mode=%s)", session_id, extract_mode)

    def shutdown(self) -> None:
        # Flush debounced extraction queue before closing store
        if self._extract_queue is not None:
            self._extract_queue.stop()
            self._extract_queue = None
        if self._store is not None:
            # Auto-trim if configured
            max_drawers = int(self._config.get("max_drawers", 0))
            if max_drawers > 0:
                try:
                    trimmed = self._store.trim_drawers(max_drawers=max_drawers)
                    if trimmed:
                        logger.info("Palace auto-trimmed %d drawers", trimmed)
                except Exception as e:
                    logger.warning("Palace trim failed: %s", e)
            self._store.close()
            self._store = None
        if self._kg is not None:
            self._kg.close()
            self._kg = None

    # -- System prompt --------------------------------------------------------

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""

        stats = self._store.stats()
        total = stats["total_drawers"]
        wings = stats["wings_count"]

        if total == 0:
            return (
                "# Memory Palace\n"
                "Active. Empty palace — proactively store knowledge worth remembering.\n"
                "Use palace_store to save facts, decisions, preferences, discoveries.\n"
                "Use palace_recall to search and recall past knowledge."
            )

        # Get L1 Essential Story
        l1 = self._store.get_top_importance()
        lines = [
            "# Memory Palace",
            f"Active. {total} drawers across {wings} wings.",
        ]
        if l1:
            lines.append("## Essential Story")
            lines.append(l1)
        lines.append(
            "Use palace_store to save knowledge, palace_recall to search, "
            "palace_mine to import files/sessions, palace_graph for relationships."
        )
        return "\n".join(lines)

    # -- Prefetch -------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._store or not query or not query.strip():
            return ""
        try:
            # Use hybrid search (FTS5 + vector RRF) if vector store available
            results = self._store.search_hybrid(
                query.strip(), limit=5, vector_store=self._vector_store
            )
            if not results:
                return ""
            lines = ["## Memory Palace"]
            for r in results:
                wing = r.get("wing", "")
                room = r.get("room", "")
                content = r.get("content", "")
                # Truncate long content
                if len(content) > 150:
                    content = content[:150] + "…"
                loc = f"[{wing}/{room}] " if wing else ""
                lines.append(f"- {loc}{content}")

            # Record recall events for Dreaming (append-only, ~0.1ms per INSERT)
            if session_id:
                for r in results:
                    try:
                        self._store.record_recall(
                            drawer_id=r.get("id", ""),
                            session_id=session_id,
                            query=query.strip(),
                            bm25_rank=r.get("rank", 0.0),
                        )
                    except Exception:
                        pass  # Non-critical: don't break prefetch on recall error

            # Phase 2: co-occurrence expansion
            try:
                entities = _extract_entities(query.strip())
                if entities:
                    related = []
                    for ent in entities[:3]:
                        related.extend(self._store.find_related_entities(ent, limit=3))
                    if related:
                        # Deduplicate and search expanded terms
                        expanded_terms = list(dict.fromkeys(related))[:5]
                        expanded_query = " OR ".join(expanded_terms)
                        expanded_results = self._store.search_fts(expanded_query, limit=3)
                        for r in expanded_results:
                            if r.get("id") not in {res.get("id") for res in results}:
                                content = r.get("content", "")
                                if len(content) > 150:
                                    content = content[:150] + "…"
                                wing = r.get("wing", "")
                                room = r.get("room", "")
                                loc = f"[{wing}/{room}] " if wing else ""
                                lines.append(f"- {loc}{content}")
            except Exception as e:
                logger.debug("Palace prefetch co-occurrence expansion failed: %s", e)

            return "\n".join(lines)
        except Exception as e:
            logger.debug("Palace prefetch failed: %s", e)
            return ""

    # -- Sync turn (auto-extract) ---------------------------------------------

    # -- Reverse index: Palace → Memory --------------------------------------

    def _sync_index_to_memory(
        self,
        content: str,
        wing: str,
        room: str,
        importance: float,
        source_type: str,
    ) -> None:
        """Write an INDEX line to Memory for high-importance auto-extracted drawers.

        Only triggers when importance >= 4.0 and source_type is auto_extract
        or session_extract. Requires _write_memory callback to be injected.
        """
        if self._write_memory is None:
            return
        if importance < 4.0:
            return
        if source_type not in ("auto_extract", "session_extract"):
            return

        # Generate keywords from wing, room, and first words of content
        keywords = set()
        keywords.add(wing)
        keywords.add(room)
        # Add first 2 meaningful words from content
        words = [w for w in content.split()[:6] if len(w) > 2]
        keywords.update(words[:2])

        snippet = content[:40].replace("\n", " ")
        kw_str = ", ".join(sorted(keywords)[:4])
        index_line = f"[INDEX] {snippet} (Palace {wing}/{room}, 搜索: {kw_str})"

        try:
            self._write_memory("memory", index_line)
            logger.debug("Palace → Memory INDEX: %s", snippet)
        except Exception as e:
            logger.debug("Palace reverse index write failed: %s", e)

    def _refine_session_to_memory_rules(
        self, memories: List[Dict[str, Any]]
    ) -> None:
        """Distill high-importance session memories into Memory [RULE] or [INDEX].

        Called at the end of on_session_end(). Uses the _write_memory callback
        injected by MemoryManager (Phase 1).

        Rules:
        - preference + importance >= 4.0 → [RULE] line in Memory
        - decision + importance >= 4.5 → [INDEX] line in Memory
        - All other types → skip (already handled by _sync_index_to_memory)
        """
        if self._write_memory is None:
            return

        for mem in memories:
            mtype = mem.get("memory_type", "general")
            conf = mem.get("confidence", 0.4)
            importance = min(5.0, round(conf * 5, 1))
            content = mem.get("content", "")

            if not content or not content.strip():
                continue

            if mtype == "preference" and importance >= 4.0:
                snippet = content[:60].replace("\n", " ")
                rule_line = f"[RULE] {snippet}"
                try:
                    self._write_memory("memory", rule_line)
                    logger.debug("Session → Memory RULE: %s", snippet)
                except Exception as e:
                    logger.debug("Session rule write failed: %s", e)

            elif mtype == "decision" and importance >= 4.5:
                snippet = content[:60].replace("\n", " ")
                index_line = f"[INDEX] {snippet} (Palace decisions/architecture)"
                try:
                    self._write_memory("memory", index_line)
                    logger.debug("Session → Memory INDEX: %s", snippet)
                except Exception as e:
                    logger.debug("Session index write failed: %s", e)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._config.get("auto_extract", True):
            return
        if self._agent_context != "primary":
            return
        if not self._store:
            return

        min_conf = float(self._config.get("min_confidence", 0.3))
        extract_mode = self._config.get("extract_mode", "regex")
        combined = f"{user_content}\n{assistant_content}"

        # Detect user correction signals
        has_correction = detect_correction(user_content)

        # --- Phase 1: Regex extraction (always, immediate) --------------------
        if extract_mode in ("regex", "hybrid"):
            memories = _extract_memories_regex(combined, min_confidence=min_conf)
        else:
            memories = []

        stored = 0
        stored_ids = []
        for mem in memories:
            mtype = mem.get("memory_type", "general")
            wing = _MEMORY_TYPE_WING.get(mtype, "general")
            room = _MEMORY_TYPE_ROOM.get(mtype, "general")
            raw_conf = mem.get("confidence", 0.3)
            importance = min(5.0, round(raw_conf * 5, 1))
            # Boost importance for correction signals
            if has_correction and mtype in ("decision", "fact", "preference"):
                importance = min(5.0, importance + 1.0)
                raw_conf = min(1.0, raw_conf + 0.2)
            # Semantic dedup check: skip near-duplicate content
            try:
                from .semantic_dedup import SemanticDeduplicator as SD
                if self._vector_store and self._embedding_client:
                    sd = SD(vector_store=self._vector_store)
                    dup = sd.find_duplicate(
                        mem["content"],
                        embed_fn=self._embedding_client.embed_text,
                    )
                    if dup.is_duplicate:
                        logger.debug("Semantic dedup: skipping duplicate (sim=%.2f, match=%s)",
                                     dup.similarity, dup.best_match_id)
                        continue
            except Exception as e:
                logger.debug("Semantic dedup check failed: %s", e)
            try:
                drawer_id = self._store.add_drawer(
                    content=mem["content"],
                    wing=wing,
                    room=room,
                    importance=importance,
                    confidence=raw_conf,
                    memory_type=mtype,
                    source_type="auto_extract",
                )
                stored_ids.append(drawer_id)
                stored += 1
                # Sync high-importance auto drawers back to Memory INDEX
                self._sync_index_to_memory(
                    content=mem["content"], wing=wing, room=room,
                    importance=importance, source_type="auto_extract",
                )
            except Exception as e:
                logger.debug("Palace sync_turn store failed: %s", e)

        # Extract co-occurrence entities from this turn
        try:
            entities = _extract_entities(combined)
            if len(entities) >= 2:
                self._store.add_cooccurrence(
                    entities=entities,
                    session_id=self._session_id or "",
                    context=combined[:200],
                )
        except Exception as e:
            logger.debug("Palace sync_turn co-occurrence failed: %s", e)

        # --- Feedback pre-check: keyword-based fast path ----------------------
        # LLM-based feedback detection happens in debounced extraction.
        # Keyword pre-check serves as fallback when LLM is unavailable.
        keyword_feedback = detect_feedback_signal(user_content)
        # Only apply keyword feedback immediately if no LLM queue configured
        if keyword_feedback is not None and self._extract_queue is None:
            try:
                from .feedback import FeedbackReactor
                reactor = FeedbackReactor(self._store)
                reactor.apply_feedback(keyword_feedback, relevant_drawer_ids=stored_ids or None)
                logger.debug("Palace keyword feedback applied (no LLM queue): %.2f", keyword_feedback)
            except Exception as e:
                logger.debug("Palace keyword feedback reactor failed: %s", e)

        if stored:
            logger.info("Palace auto-extracted %d memories from turn (regex)", stored)

        # --- Phase 2: LLM extraction (debounced) ------------------------------
        if self._extract_queue is not None:
            # Filter: only user input + final assistant response
            filtered = self._filter_turn_content(user_content, assistant_content)
            if filtered:
                self._extract_queue.enqueue(
                    filtered,
                    min_confidence=min_conf,
                    has_correction=has_correction,
                    keyword_feedback=keyword_feedback,
                )

    @staticmethod
    def _filter_turn_content(user_content: str, assistant_content: str) -> str:
        """Filter turn content: strip tool_call artifacts, keep meaningful text.

        Removes lines that look like tool call/results, keeping user input
        and the assistant's final natural-language response.
        """
        import re

        def _clean(text: str) -> str:
            # Remove tool-call JSON blocks (common artifact)
            text = re.sub(r'\{["\']tool_call["\'].*?\}', '', text, flags=re.DOTALL)
            # Remove tool result blocks
            text = re.sub(r'Tool output \(truncated\).*', '', text, flags=re.DOTALL)
            # Remove very long single lines (>500 chars, likely tool output)
            lines = text.split('\n')
            lines = [l for l in lines if len(l.strip()) < 500]
            return '\n'.join(lines).strip()

        cleaned_user = _clean(user_content)
        cleaned_asst = _clean(assistant_content)

        if not cleaned_user and not cleaned_asst:
            return ""

        parts = []
        if cleaned_user:
            parts.append(f"User: {cleaned_user}")
        if cleaned_asst:
            parts.append(f"Assistant: {cleaned_asst}")
        return "\n".join(parts)

    def _store_debounced_memories(
        self,
        memories: list,
        entities: list,
        has_correction: bool,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store LLM-extracted memories from the debounce queue.

        Also applies Cognee-style feedback reactor: if feedback signal is
        detected, adjust importance of targeted (or recent) drawers.
        """
        if not self._store:
            return

        min_conf = float(self._config.get("min_confidence", 0.3))
        stored_ids: list[str] = []
        stored = 0
        for mem in memories:
            mtype = mem.get("memory_type", mem.get("type", "fact"))
            wing = _MEMORY_TYPE_WING.get(mtype, "general")
            room = _MEMORY_TYPE_ROOM.get(mtype, "general")
            raw_conf = mem.get("confidence", 0.5)
            importance = min(5.0, round(raw_conf * 5, 1))
            # Boost for correction signals
            if has_correction and mtype in ("decision", "fact", "preference"):
                importance = min(5.0, importance + 1.0)
                raw_conf = min(1.0, raw_conf + 0.2)
            try:
                # check_duplicate avoids re-storing regex-extracted memories
                existing = self._store.check_duplicate(mem["content"])
                if existing:
                    continue
                did = self._store.add_drawer(
                    content=mem["content"],
                    wing=wing,
                    room=room,
                    importance=importance,
                    confidence=raw_conf,
                    memory_type=mtype,
                    source_type="auto_extract_llm",
                )
                stored_ids.append(did)
                stored += 1
            except Exception as e:
                logger.debug("Palace debounced store failed: %s", e)

        # Co-occurrence from LLM-extracted entities
        if len(entities) >= 2:
            try:
                self._store.add_cooccurrence(
                    entities=entities,
                    session_id=self._session_id or "",
                    context="",
                )
            except Exception:
                pass

        # --- Cognee-style feedback reactor ---
        if feedback and feedback.get("signal") in ("positive", "negative"):
            try:
                from .feedback import FeedbackReactor
                signal = feedback["signal"]
                strength = feedback.get("strength", 0.5)
                targets = feedback.get("targets", [])
                # Convert to signed strength
                signed = strength if signal == "positive" else -strength
                # Resolve target drawer IDs (Cognee: used_graph_element_ids)
                relevant_ids = self._resolve_feedback_targets(targets, stored_ids)
                reactor = FeedbackReactor(self._store)
                n = reactor.apply_feedback(signed, relevant_drawer_ids=relevant_ids)
                logger.info(
                    "Palace feedback reactor: signal=%s strength=%.2f targets=%s adjusted=%d drawers",
                    signal, strength, targets, n,
                )
            except Exception as e:
                logger.debug("Palace debounced feedback reactor failed: %s", e)

        if stored:
            logger.info("Palace debounced LLM extraction: %d memories stored", stored)

    def _resolve_feedback_targets(
        self,
        targets: list[str],
        fallback_ids: list[str],
    ) -> Optional[list[str]]:
        """Resolve feedback targets to drawer IDs.

        Cognee-style: if LLM identified specific topics, search drawers
        by those keywords.  Otherwise fall back to recently stored IDs.

        Returns list of drawer IDs or None (let reactor pick recent).
        """
        if not targets:
            return fallback_ids or None

        try:
            matched_ids = []
            for keyword in targets:
                results = self._store.search_fts(keyword, limit=3)
                for r in results:
                    did = r.get("id") if isinstance(r, dict) else None
                    if did and did not in matched_ids:
                        matched_ids.append(did)
            if matched_ids:
                return matched_ids
        except Exception as e:
            logger.debug("Feedback target resolution failed: %s", e)

        return fallback_ids or None

    # -- Tools ----------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            PALACE_STORE_SCHEMA,
            PALACE_RECALL_SCHEMA,
            PALACE_MINE_SCHEMA,
            PALACE_GRAPH_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        dispatch = {
            "palace_store": self._handle_palace_store,
            "palace_recall": self._handle_palace_recall,
            "palace_mine": self._handle_palace_mine,
            "palace_graph": self._handle_palace_graph,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            return tool_error(f"Unknown tool: {tool_name}")
        return handler(args)

    # -- palace_store handler -------------------------------------------------

    def _handle_palace_store(self, args: dict) -> str:
        try:
            action = args["action"]
            store = self._store

            if action == "add":
                content = args.get("content", "")
                if not content:
                    return tool_error("'content' is required for add action")
                drawer_id = store.add_drawer(
                    content=content,
                    wing=args.get("wing", "general"),
                    room=args.get("room", "general"),
                    hall=args.get("hall", ""),
                    importance=float(args.get("importance", 3.0)),
                )
                return json.dumps({"drawer_id": drawer_id, "status": "added"})

            elif action == "get":
                drawer_id = args.get("drawer_id", "")
                if not drawer_id:
                    return tool_error("'drawer_id' is required for get action")
                drawer = store.get_drawer(drawer_id)
                if drawer is None:
                    return json.dumps({"error": "Drawer not found", "drawer_id": drawer_id})
                return json.dumps({"drawer": drawer})

            elif action == "delete":
                drawer_id = args.get("drawer_id", "")
                if not drawer_id:
                    return tool_error("'drawer_id' is required for delete action")
                deleted = store.delete_drawer(drawer_id)
                return json.dumps({"deleted": deleted, "drawer_id": drawer_id})

            elif action == "list":
                drawers = store.list_drawers(
                    wing=args.get("wing"),
                    room=args.get("room"),
                    limit=int(args.get("limit", 20)),
                    offset=int(args.get("offset", 0)),
                )
                return json.dumps({"drawers": drawers, "count": len(drawers)})

            elif action == "duplicate_check":
                content = args.get("content", "")
                if not content:
                    return tool_error("'content' is required for duplicate_check action")
                dup = store.check_duplicate(content)
                if not dup:
                    return json.dumps({"duplicate": False})
                return json.dumps({"duplicate": True, "matches": dup})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- palace_recall handler ------------------------------------------------

    def _handle_palace_recall(self, args: dict) -> str:
        try:
            action = args["action"]
            store = self._store

            if action == "search":
                query = args.get("query", "")
                if not query:
                    return tool_error("'query' is required for search action")
                results = store.search_hybrid(
                    query,
                    wing=args.get("wing"),
                    room=args.get("room"),
                    limit=int(args.get("limit", 5)),
                    vector_store=self._vector_store,
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "top":
                n = int(args.get("limit", 8))
                l1 = store.get_top_importance(n=n)
                if not l1:
                    return json.dumps({"essential_story": "", "message": "No drawers stored yet"})
                return json.dumps({"essential_story": l1})

            elif action == "map":
                taxonomy = store.get_taxonomy()
                return json.dumps({"taxonomy": taxonomy})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- palace_mine handler --------------------------------------------------

    def _handle_palace_mine(self, args: dict) -> str:
        try:
            action = args["action"]
            store = self._store

            if action == "project":
                project_path = args.get("project_path", "")
                if not project_path:
                    return tool_error("'project_path' is required for project mining")
                wing = args.get("wing", "project")
                result = mine_project(store, project_path, wing=wing)
                return json.dumps(result)

            elif action == "sessions":
                state_db = str(Path(self._hermes_home) / "state.db") if self._hermes_home else None
                wing = args.get("wing", "hermes_sessions")
                max_sessions = int(args.get("max_sessions", 10))
                result = mine_sessions(store, state_db_path=state_db, wing=wing, max_sessions=max_sessions)
                return json.dumps(result)

            elif action == "status":
                result = mine_status(store)
                return json.dumps(result)

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- palace_graph handler -------------------------------------------------

    def _handle_palace_graph(self, args: dict) -> str:
        try:
            action = args["action"]
            kg = self._kg

            if action == "add_entity":
                name = args.get("name", "")
                if not name:
                    return tool_error("'name' is required for add_entity")
                eid = kg.add_entity(
                    name=name,
                    entity_type=args.get("entity_type", "unknown"),
                )
                return json.dumps({"entity_id": eid, "status": "added"})

            elif action == "add_triple":
                subject = args.get("subject", "")
                predicate = args.get("predicate", "")
                obj = args.get("object", "")
                if not subject or not predicate or not obj:
                    return tool_error("'subject', 'predicate', and 'object' are required for add_triple")
                tid = kg.add_triple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                )
                return json.dumps({"triple_id": tid, "status": "added"})

            elif action == "query":
                entity = args.get("subject", "")
                if not entity:
                    return tool_error("'subject' is required for query action")
                results = kg.query_triples(
                    entity=entity,
                    direction=args.get("direction", "both"),
                    predicate=args.get("predicate"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "timeline":
                name = args.get("name", "") or args.get("subject", "")
                if not name:
                    return tool_error("'name' or 'subject' is required for timeline")
                results = kg.get_entity_timeline(name, limit=int(args.get("limit", 20)))
                return json.dumps({"results": results, "count": len(results)})

            elif action == "search":
                query = args.get("query", "")
                if not query:
                    return tool_error("'query' is required for search action")
                results = kg.search_entities(query, limit=int(args.get("limit", 10)))
                return json.dumps({"results": results, "count": len(results)})

            elif action == "register_alias":
                name = args.get("name", "")
                alias = args.get("alias", "")
                if not name or not alias:
                    return tool_error("'name' and 'alias' are required for register_alias")
                kg.register_entity_alias(name=name, alias=alias)
                return json.dumps({"status": "registered", "name": name, "alias": alias})

            elif action == "resolve":
                name = args.get("name", "") or args.get("query", "")
                if not name:
                    return tool_error("'name' or 'query' is required for resolve")
                resolved = kg.resolve_entity(name)
                if resolved is None:
                    return json.dumps({"resolved": None, "message": "Entity not found"})
                return json.dumps({"resolved": resolved})

            elif action == "stats":
                result = kg.stats()
                return json.dumps(result)

            elif action == "registry":
                registry_type = args.get("entity_type")
                results = kg.get_registry(registry_type=registry_type)
                return json.dumps({"results": results, "count": len(results)})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- Hooks ----------------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._config.get("session_extract", True):
            return
        if not self._store or not messages:
            return

        min_conf = float(self._config.get("min_confidence", 0.3))
        # Use a higher threshold for session-end batch extraction
        session_min_conf = max(min_conf, 0.4)

        # Collect all user + assistant text
        texts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                texts.append(content)

        combined = "\n\n".join(texts)
        if not combined.strip():
            return

        # Use LLM extraction with regex fallback
        memories, session_entities = extract_memories_with_fallback(
            combined, min_confidence=session_min_conf, prefer_llm=True
        )
        stored = 0
        new_drawer_ids = []
        for mem in memories:
            mtype = mem.get("memory_type", "general")
            wing = _MEMORY_TYPE_WING.get(mtype, "general")
            room = _MEMORY_TYPE_ROOM.get(mtype, "general")
            importance = min(5.0, round(mem.get("confidence", 0.4) * 5, 1))
            raw_conf = mem.get("confidence", 0.4)
            try:
                drawer_id = self._store.add_drawer(
                    content=mem["content"],
                    wing=wing,
                    room=room,
                    importance=importance,
                    confidence=raw_conf,
                    memory_type=mtype,
                    source_type="session_extract",
                )
                new_drawer_ids.append(drawer_id)
                stored += 1
                # Sync high-importance session drawers back to Memory INDEX
                self._sync_index_to_memory(
                    content=mem["content"], wing=wing, room=room,
                    importance=importance, source_type="session_extract",
                )
            except Exception as e:
                logger.debug("Palace session_end store failed: %s", e)

        # Phase 2: Distill high-importance preferences/decisions into Memory rules
        self._refine_session_to_memory_rules(memories)

        # Build cross-references: link new drawers to semantically related existing ones
        for did in new_drawer_ids:
            try:
                drawer = self._store.get_drawer(did)
                if not drawer:
                    continue
                candidates = self._store.check_duplicate(drawer["content"], limit=3)
                if isinstance(candidates, list):
                    for cand in candidates:
                        cand_id = cand.get("id", "")
                        if cand_id and cand_id != did:
                            self._store.add_link(did, cand_id)
            except Exception as e:
                logger.debug("Palace session_end link building failed: %s", e)

        # Extract co-occurrence entities from the full session
        # Prefer LLM-extracted entities; fall back to regex if empty
        try:
            entities = session_entities if session_entities else _extract_entities(combined)
            if len(entities) >= 2:
                self._store.add_cooccurrence(
                    entities=entities,
                    session_id=self._session_id or "",
                    context=combined[:300],
                    weight=1.5,  # Session-end co-occurrences get higher weight
                )
        except Exception as e:
            logger.debug("Palace session_end co-occurrence failed: %s", e)

        # Extract potential entities and triples
        if self._kg:
            for text in texts:
                for pattern in _TRIPLE_PATTERNS:
                    for match in pattern.finditer(text):
                        subj, pred, obj = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
                        try:
                            self._kg.add_triple(subject=subj, predicate=pred, object=obj)
                        except Exception as e:
                            logger.debug("Palace session_end triple extraction failed: %s", e)

        if stored:
            logger.info("Palace session-end extracted %d memories", stored)

    _SKIP_MIRROR_TAGS = frozenset({"INDEX", "RULE", "ACTIVE", "WORKAROUND", "ENV"})

    def _classify_content(self, content: str, target: str) -> tuple:
        """Classify Memory content into Palace wing/room.

        Uses the same wing/room taxonomy as _MEMORY_TYPE_WING/ROOM.
        """
        if target == "user":
            return "user", "preferences"

        tag_match = re.match(r'\[(\w+)\]', content)
        tag = tag_match.group(1).upper() if tag_match else ""
        lower = content.lower()

        # Infra patterns
        if tag in ("GATEWAY",) or "gateway" in lower:
            return "infra", "gateway"
        if tag in ("MIHOMO", "PROXY") or any(k in lower for k in ("proxy", "mihomo", "clash")):
            return "infra", "proxy"
        if tag.startswith("WEIXIN") or "weixin" in lower or "wechat" in lower:
            return "infra", "weixin"

        # Cron patterns
        if tag in ("DAILY", "BRIEFING", "CRON") or any(k in lower for k in ("briefing", "cron", "schedule")):
            return "cron", "jobs"

        # Tech patterns
        if tag in ("ENV",) or any(k in lower for k in ("instance", "ubuntu", "provider", "python", "node")):
            return "tech", "environment"

        # Fallback
        return "general", "builtin_mirror"

    def on_memory_write(self, action: str, target: str, content: str,
                        old_text: str = "") -> None:
        """Mirror Memory writes to Palace with intelligent classification."""
        if not self._config.get("mirror_builtin", True):
            return
        if action not in ("add", "replace") or not self._store or not content:
            return

        # Filter out Memory-only tags
        tag_match = re.match(r'\[(\w+)\]', content)
        if tag_match and tag_match.group(1).upper() in self._SKIP_MIRROR_TAGS:
            return

        wing, room = self._classify_content(content, target)

        try:
            if action == "replace" and old_text:
                # Find and update the matching drawer in Palace
                matches = self._store.search_fts(old_text, limit=5)
                if matches:
                    best = matches[0]
                    self._store.delete_drawer(best["id"])
                    self._store.add_drawer(
                        content=content,
                        wing=wing,
                        room=room,
                        importance=best.get("importance", 4.0),
                        memory_type="builtin_mirror",
                        source_type="builtin_mirror",
                    )
                    logger.debug("Palace mirror replaced drawer %d via old_text", best["id"])
                    return

            # Use check_duplicate to avoid duplicates with sync_turn/on_session_end
            dup = self._store.check_duplicate(content)
            if dup:
                logger.debug("Palace mirror skipped (duplicate): %s...", content[:60])
                return

            self._store.add_drawer(
                content=content,
                wing=wing,
                room=room,
                importance=4.0,
                memory_type="builtin_mirror",
                source_type="builtin_mirror",
            )
        except Exception as e:
            logger.debug("Palace memory_write mirror failed: %s", e)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract memories from messages about to be compressed.

        Does NOT store to Palace — that's on_session_end's job.
        Only returns a summary string for injection into the compressed context,
        so the model retains awareness of key facts after compression.
        """
        if not messages:
            return ""

        # Collect user + assistant text (skip system/tool noise)
        texts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                texts.append(content)

        combined = "\n\n".join(texts)
        if not combined.strip():
            return ""

        # Use higher confidence threshold for mid-conversation extraction
        # to avoid polluting Palace with transient dialogue noise.
        # Only LLM extraction here — regex is too noisy for this context.
        try:
            memories, _entities = extract_memories_with_fallback(
                combined, min_confidence=0.6, prefer_llm=True
            )
        except Exception:
            return ""

        if not memories:
            return ""

        lines = ["## Memory Palace — Key memories from this conversation:"]
        for mem in memories[:8]:
            mtype = mem.get("memory_type", "general")
            lines.append(f"- [{mtype}] {mem['content']}")
        return "\n".join(lines)

    # -- Config ---------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "auto_extract",
                "description": "Auto-extract memories from each conversation turn",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "session_extract",
                "description": "Extract memories when session ends",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "min_confidence",
                "description": "Minimum confidence threshold for auto-extraction (0.0-1.0)",
                "default": "0.3",
            },
            {
                "key": "mirror_builtin",
                "description": "Mirror built-in memory writes to the palace",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "db_path",
                "description": "Path to the drawers database",
                "default": "$HERMES_HOME/palace/drawers.db",
            },
            {
                "key": "kg_path",
                "description": "Path to the knowledge graph database",
                "default": "$HERMES_HOME/palace/knowledge_graph.db",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path) as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["palace"] = values
            with open(config_path, "w") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception as e:
            logger.debug("Palace save_config failed: %s", e)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the Memory Palace provider with the plugin system."""
    config = _load_plugin_config()
    provider = PalaceMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
