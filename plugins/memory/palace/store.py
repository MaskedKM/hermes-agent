"""PalaceStore — SQLite + FTS5 storage layer for the Memory Palace plugin.

Manages a SQLite database with FTS5 full-text search for storing 'drawers'
(pieces of knowledge) organized into wings, rooms, and halls.
"""

from __future__ import annotations

import hashlib
import json
import os
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# -- jieba tokenizer for Chinese FTS support ----------------------------
_jieba = None
_JIEBA_INIT = False


def _get_jieba():
    """Lazy-load jieba. Returns the module or None if unavailable."""
    global _jieba, _JIEBA_INIT
    if _JIEBA_INIT:
        return _jieba
    _JIEBA_INIT = True
    try:
        import jieba as _jb
        _jieba = _jb
        # Suppress jieba init log noise
        _jb.setLogLevel(logging.WARNING)
        logger.debug("jieba tokenizer loaded for Chinese FTS support")
    except ImportError:
        logger.debug("jieba not available — Chinese FTS will use LIKE fallback")
    return _jieba


def _has_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _tokenize(text: str) -> str:
    """Tokenize text for FTS index.

    Uses jieba for Chinese text (produces space-separated tokens).
    For non-Chinese text, returns the original (FTS handles it natively).
    Also preserves original English/code words so they remain searchable.
    """
    jb = _get_jieba()
    if jb is None or not _has_chinese(text):
        return text

    # Extract segments: Chinese runs get jieba'd, non-Chinese passes through
    parts = []
    for seg in re.split(r"([\u4e00-\u9fff]+)", text):
        if re.match(r"[\u4e00-\u9fff]+", seg):
            # Chinese segment: jieba cut, filter stop words & short tokens
            tokens = [w for w in jb.cut(seg) if len(w.strip()) >= 2 and w.strip() not in _STOP_WORDS]
            parts.append(" ".join(tokens))
        elif seg.strip():
            parts.append(seg)
    return " ".join(parts)


def _load_stop_words() -> frozenset[str]:
    """Load Chinese stop words from ~/.hermes/palace/stop_words.txt.

    Returns empty frozenset if file is missing or unreadable.
    """
    path = os.path.expanduser("~/.hermes/palace/stop_words.txt")
    try:
        with open(path, encoding="utf-8") as f:
            words = set()
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    words.add(line)
            return frozenset(words)
    except (FileNotFoundError, OSError):
        return frozenset()


# Loaded once at module import; restart gateway to pick up edits.
_STOP_WORDS = _load_stop_words()

# FTS5 special characters that must be escaped in MATCH expressions.
# Note: '*' is NOT included here because prefix queries (word*) use it intentionally.
_FTS_SPECIAL = re.compile(r'(["()\\:\-])')


def _fts_escape(term: str) -> str:
    """Escape a token for safe use in FTS5 MATCH expressions.

    Strategy: wrap each token in double-quotes so FTS5 treats it as a
    literal phrase, not an operator.  Double-quotes inside are escaped
    as "" (FTS5 convention).
    """
    # For individual tokens (no spaces), just wrap in quotes
    if " " not in term:
        escaped_inner = term.replace('"', '""')
        return f'"{escaped_inner}"'
    # For multi-word terms, escape each word individually
    words = term.split()
    escaped_words = [f'"{w.replace(chr(34), chr(34)*2)}"' for w in words if w]
    return " ".join(escaped_words)


class PalaceStore:
    """Core storage backend for the Memory Palace plugin.

    Uses SQLite with WAL mode and FTS5 for full-text search.
    The caller is responsible for managing the lifecycle (call ``close()``).
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._open()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open (or create) the database and ensure schema is ready."""
        path = Path(self._db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        self._conn = conn
        self._create_tables()
        self._auto_backup()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _auto_backup(self) -> None:
        """Create a lightweight backup if the last backup is >24h old."""
        import sqlite3 as _sqlite3
        from datetime import timedelta

        backup_dir = Path(self._db_path).parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        backups = sorted(backup_dir.glob("drawers_*.db"), reverse=True)
        if backups:
            last_mtime = datetime.fromtimestamp(backups[0].stat().st_mtime, tz=timezone.utc)
            if datetime.now(tz=timezone.utc) - last_mtime < timedelta(hours=24):
                return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"drawers_{ts}.db"
        try:
            conn = _sqlite3.connect(self._db_path)
            backup_conn = _sqlite3.connect(str(backup_path))
            conn.backup(backup_conn)
            backup_conn.close()
            conn.close()

            for old in backups[6:]:
                old.unlink()

            logger.info("Palace backup created: %s", backup_path)
        except Exception as e:
            logger.debug("Palace auto-backup failed: %s", e)

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("PalaceStore is closed")
        return self._conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        c = self.conn
        c.executescript("""
            CREATE TABLE IF NOT EXISTS drawers (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                wing TEXT NOT NULL,
                room TEXT NOT NULL,
                hall TEXT DEFAULT '',
                source_file TEXT DEFAULT '',
                source_type TEXT DEFAULT 'manual',
                chunk_index INTEGER DEFAULT 0,
                added_by TEXT DEFAULT 'hermes',
                filed_at TEXT DEFAULT (datetime('now')),
                importance REAL DEFAULT 3.0,
                confidence REAL DEFAULT 0.5,
                archived INTEGER DEFAULT 0,
                memory_type TEXT DEFAULT 'general',
                source_mtime REAL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_drawers_wing ON drawers(wing);
            CREATE INDEX IF NOT EXISTS idx_drawers_room ON drawers(room);
            CREATE INDEX IF NOT EXISTS idx_drawers_wing_room ON drawers(wing, room);
            CREATE INDEX IF NOT EXISTS idx_drawers_importance ON drawers(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_drawers_source_file ON drawers(source_file);
        """)

        # Safe ALTER: add columns that may not exist yet
        for col_sql in [
            "ALTER TABLE drawers ADD COLUMN superseded_by TEXT DEFAULT NULL",
            "ALTER TABLE drawers ADD COLUMN tokens TEXT DEFAULT ''",
            "ALTER TABLE drawers ADD COLUMN confidence REAL DEFAULT 0.5",
            "ALTER TABLE drawers ADD COLUMN archived INTEGER DEFAULT 0",
            "ALTER TABLE drawers ADD COLUMN source_session_id TEXT DEFAULT NULL",
            "ALTER TABLE drawers ADD COLUMN feedback_weight REAL DEFAULT 0.5",
        ]:
            try:
                self.conn.execute(col_sql)
            except Exception:
                pass  # Column already exists

        # Create indexes on columns added by ALTER (must come after ALTER)
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_drawers_confidence ON drawers(confidence DESC)",
            "CREATE INDEX IF NOT EXISTS idx_drawers_archived ON drawers(archived)",
        ]:
            try:
                self.conn.execute(idx_sql)
            except Exception:
                pass  # Column or index may not exist yet

        self.conn.commit()

        # FTS5 table with tokens column for Chinese support.
        # We must drop and recreate if schema changed (FTS5 doesn't support ALTER).
        self._ensure_fts_schema()

        # Create remaining tables
        self.conn.executescript("""
            -- 增强一: drawer_links (Drawer 交叉引用)
            CREATE TABLE IF NOT EXISTS drawer_links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (source_id, target_id)
            );
            CREATE INDEX IF NOT EXISTS idx_dlinks_source ON drawer_links(source_id);
            CREATE INDEX IF NOT EXISTS idx_dlinks_target ON drawer_links(target_id);

            -- 增强二: cooccurrence (共现提取)
            CREATE TABLE IF NOT EXISTS cooccurrence_groups (
                id TEXT PRIMARY KEY,
                entities TEXT NOT NULL,
                context TEXT DEFAULT '',
                session_id TEXT DEFAULT '',
                weight REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS cooccurrence_items (
                entity TEXT NOT NULL,
                group_id TEXT NOT NULL,
                PRIMARY KEY (entity, group_id)
            );
            CREATE INDEX IF NOT EXISTS idx_co_items_entity ON cooccurrence_items(entity);

            CREATE TABLE IF NOT EXISTS mined_sources (
                source_key TEXT PRIMARY KEY,
                mtime REAL,
                mined_at TEXT DEFAULT (datetime('now'))
            );

            -- Dreaming: recall events (search hit tracking)
            CREATE TABLE IF NOT EXISTS recall_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drawer_id TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT '',
                query TEXT DEFAULT '',
                bm25_rank REAL NOT NULL DEFAULT 0.0,
                recalled_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_recall_events_drawer ON recall_events(drawer_id);
            CREATE INDEX IF NOT EXISTS idx_recall_events_session ON recall_events(session_id);
            CREATE INDEX IF NOT EXISTS idx_recall_events_time ON recall_events(recalled_at);

            -- Dreaming: phase signals
            CREATE TABLE IF NOT EXISTS dreaming_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drawer_id TEXT NOT NULL,
                phase TEXT NOT NULL CHECK(phase IN ('light', 'deep', 'rem')),
                signal_strength REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_dreaming_signals_drawer ON dreaming_signals(drawer_id);
            CREATE INDEX IF NOT EXISTS idx_dreaming_signals_phase ON dreaming_signals(phase);

            -- Feedback loop: feedback events
            CREATE TABLE IF NOT EXISTS feedback_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drawer_id TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT '',
                feedback_type TEXT NOT NULL CHECK(feedback_type IN (
                    'correction',
                    'affirmation',
                    'implicit_negative'
                )),
                signal_strength REAL NOT NULL DEFAULT 0.0,
                context TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (drawer_id) REFERENCES drawers(id)
            );
            CREATE INDEX IF NOT EXISTS idx_feedback_drawer ON feedback_events(drawer_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_time ON feedback_events(created_at);

            -- Dreaming: run logs
            CREATE TABLE IF NOT EXISTS dreaming_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase TEXT NOT NULL CHECK(phase IN ('light', 'deep', 'rem', 'recovery')),
                status TEXT NOT NULL DEFAULT 'success',
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                finished_at TEXT,
                duration_ms INTEGER,
                report_json TEXT,
                memory_health REAL,
                drawers_active INTEGER,
                drawers_processed INTEGER,
                drawers_merged INTEGER DEFAULT 0,
                drawers_promoted INTEGER DEFAULT 0,
                drawers_archived INTEGER DEFAULT 0
            );
        """)
        self.conn.commit()
        logger.debug("PalaceStore schema ensured at %s", self._db_path)

    def _ensure_fts_schema(self) -> None:
        """Ensure the FTS5 virtual table has the correct schema (with tokens column).

        FTS5 does not support ALTER TABLE, so we must drop and recreate when
        the schema changes. This is safe because we use content= table sync.
        """
        # Check current FTS schema
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='drawers_fts'"
        ).fetchone()

        target_has_tokens = "tokens" in (row[0] or "").lower() if row else False

        if target_has_tokens:
            return  # Schema already correct

        # Need to rebuild FTS with tokens column
        logger.info("Palace FTS schema upgrade: adding tokens column for Chinese support")

        # Drop old FTS (triggers first, then table)
        self.conn.executescript("""
            DROP TRIGGER IF EXISTS drawers_ai;
            DROP TRIGGER IF EXISTS drawers_ad;
            DROP TRIGGER IF EXISTS drawers_au;
            DROP TABLE IF EXISTS drawers_fts;
        """)

        # Create new FTS with tokens column
        self.conn.execute("""
            CREATE VIRTUAL TABLE drawers_fts USING fts5(
                content, tokens, wing, room, hall, source_file UNINDEXED,
                content='drawers', content_rowid='rowid'
            );
        """)

        # Recreate triggers (include tokens)
        self.conn.executescript("""
            CREATE TRIGGER drawers_ai AFTER INSERT ON drawers BEGIN
                INSERT INTO drawers_fts(rowid, content, tokens, wing, room, hall, source_file)
                VALUES (new.rowid, new.content, new.tokens, new.wing, new.room, new.hall, new.source_file);
            END;

            CREATE TRIGGER drawers_ad AFTER DELETE ON drawers BEGIN
                INSERT INTO drawers_fts(drawers_fts, rowid, content, tokens, wing, room, hall, source_file)
                VALUES ('delete', old.rowid, old.content, old.tokens, old.wing, old.room, old.hall, old.source_file);
            END;

            CREATE TRIGGER drawers_au AFTER UPDATE ON drawers BEGIN
                INSERT INTO drawers_fts(drawers_fts, rowid, content, tokens, wing, room, hall, source_file)
                VALUES ('delete', old.rowid, old.content, old.tokens, old.wing, old.room, old.hall, old.source_file);
                INSERT INTO drawers_fts(rowid, content, tokens, wing, room, hall, source_file)
                VALUES (new.rowid, new.content, new.tokens, new.wing, new.room, new.hall, new.source_file);
            END;
        """)

        self.conn.commit()

        # Backfill tokens for existing drawers
        self._backfill_tokens()

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_drawer_id(content: str, wing: str, room: str) -> str:
        """Generate a deterministic drawer ID."""
        digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:16]
        return f"drawer_{wing}_{room}_{digest}"

    def _backfill_tokens(self) -> None:
        """Fill the tokens column for all existing drawers that have empty tokens."""
        rows = self.conn.execute(
            "SELECT rowid, content FROM drawers WHERE tokens = '' OR tokens IS NULL"
        ).fetchall()
        if not rows:
            return
        count = 0
        for row in rows:
            tokens = _tokenize(row["content"])
            self.conn.execute(
                "UPDATE drawers SET tokens = ? WHERE rowid = ?",
                (tokens, row["rowid"]),
            )
            count += 1
        self.conn.commit()
        if count:
            logger.info("Palace backfilled tokens for %d drawers", count)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def add_drawer(
        self,
        content: str,
        wing: str,
        room: str = "general",
        hall: str = "",
        source_file: str = "",
        source_type: str = "manual",
        chunk_index: int = 0,
        importance: float = 3.0,
        confidence: float = 0.5,
        memory_type: str = "general",
        source_mtime: float = 0,
    ) -> str:
        """Insert (or replace) a drawer. Returns the drawer ID."""
        drawer_id = self._make_drawer_id(content, wing, room)
        tokens = _tokenize(content)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO drawers
                (id, content, tokens, wing, room, hall, source_file, source_type,
                 chunk_index, added_by, filed_at, importance, confidence, memory_type,
                 source_mtime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'hermes', datetime('now'), ?, ?, ?, ?)
            """,
            (
                drawer_id, content, tokens, wing, room, hall, source_file, source_type,
                chunk_index, importance, confidence, memory_type, source_mtime,
            ),
        )
        self.conn.commit()
        logger.debug("add_drawer %s (wing=%s room=%s)", drawer_id, wing, room)
        return drawer_id

    def get_drawer(self, drawer_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single drawer by ID. Returns a dict or None."""
        row = self.conn.execute(
            "SELECT * FROM drawers WHERE id = ?", (drawer_id,)
        ).fetchone()
        return dict(row) if row else None

    def delete_drawer(self, drawer_id: str) -> bool:
        """Delete a drawer (and its FTS entry via trigger). Also cleans up drawer_links."""
        # Clean up drawer_links first
        self.conn.execute(
            "DELETE FROM drawer_links WHERE source_id = ? OR target_id = ?",
            (drawer_id, drawer_id),
        )
        cur = self.conn.execute("DELETE FROM drawers WHERE id = ?", (drawer_id,))
        self.conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.debug("delete_drawer %s (links cleaned)", drawer_id)
        return deleted

    def batch_update_drawers(self, updates: list[dict]) -> int:
        """Batch-update drawer fields.  Each *update* must contain ``id``
        plus the columns to change (e.g. ``importance``)."""
        if not updates:
            return 0
        count = 0
        for u in updates:
            did = u.pop("id", None)
            if not did or not u:
                continue
            set_clause = ", ".join(f"{k} = ?" for k in u)
            vals = list(u.values()) + [did]
            cur = self.conn.execute(
                f"UPDATE drawers SET {set_clause} WHERE id = ?", vals
            )
            count += cur.rowcount
        if count:
            self.conn.commit()
        return count

    def set_feedback_weight(self, drawer_id: str, weight: float) -> None:
        """Persist the feedback EMA weight for a drawer."""
        self.conn.execute(
            "UPDATE drawers SET feedback_weight = ? WHERE id = ?",
            (weight, drawer_id),
        )
        self.conn.commit()

    def list_drawers(
        self,
        wing: str | None = None,
        room: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Paginated listing with optional wing/room filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if wing is not None:
            clauses.append("wing = ?")
            params.append(wing)
        if room is not None:
            clauses.append("room = ?")
            params.append(room)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM drawers{where} ORDER BY filed_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_fts(
        self,
        query: str,
        wing: str | None = None,
        room: str | None = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """FTS5 BM25 full-text search with optional wing/room filter.

        For Chinese queries, automatically searches the tokens column
        (jieba-pre-tokenized) in addition to the raw content column.
        """
        # Build match clause: for Chinese, search tokens column; otherwise content
        if _has_chinese(query):
            tokenized = _tokenize(query)
            # Split tokenized output into individual tokens and OR them.
            # FTS5 treats spaces as implicit AND; we want OR for Chinese terms
            # so "记忆增强" (tokenized to "记忆 增强") matches drawers containing
            # either "记忆" or "增强".
            tokens_list = tokenized.split()
            if tokens_list:
                escaped = [_fts_escape(t) for t in tokens_list]
                or_expr = " OR ".join(escaped)
                match_param = f"tokens:({or_expr})"
            else:
                match_param = query  # fallback
            match_clause = "drawers_fts MATCH ?"
        else:
            match_clause = "drawers_fts MATCH ?"
            # Escape FTS special chars in plain English queries too
            match_param = _fts_escape(query)

        clauses: list[str] = [match_clause, "d.archived = 0"]
        params: list[Any] = [match_param]

        if wing is not None:
            clauses.append("d.wing = ?")
            params.append(wing)
        if room is not None:
            clauses.append("d.room = ?")
            params.append(room)

        where = " AND ".join(clauses)
        sql = f"""
            SELECT d.id, d.content, d.wing, d.room, d.importance,
                   d.confidence, rank AS rank
            FROM drawers_fts AS fts
            JOIN drawers AS d ON d.rowid = fts.rowid
            WHERE {where}
            ORDER BY rank, (d.importance * d.confidence) DESC
            LIMIT ?
        """
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()

        # If Chinese search found nothing, try LIKE fallback on content
        if not rows and _has_chinese(query):
            like_clauses = ["d.content LIKE ?", "d.archived = 0"]
            like_params: list[Any] = [f"%{query}%"]
            if wing is not None:
                like_clauses.append("d.wing = ?")
                like_params.append(wing)
            if room is not None:
                like_clauses.append("d.room = ?")
                like_params.append(room)
            like_where = " AND ".join(like_clauses)
            like_sql = f"""
                SELECT d.id, d.content, d.wing, d.room, d.importance,
                       d.confidence, 0.0 AS rank
                FROM drawers AS d
                WHERE {like_where}
                ORDER BY (d.importance * d.confidence) DESC
                LIMIT ?
            """
            like_params.append(limit)
            rows = self.conn.execute(like_sql, like_params).fetchall()

        return [dict(r) for r in rows]

    def check_duplicate(self, content: str, limit: int = 5) -> list[Dict[str, Any]]:
        """Check if similar content exists using FTS5 prefix search on first few words.

        Args:
            content: Text to check against existing drawers.
            limit: Max results to return (default 5).

        Returns:
            List of matching drawer dicts, or empty list if no matches.
        """
        # Take first few words as a prefix query for fuzzy duplicate check
        # Strip FTS5 special characters ([ ] " *) to avoid syntax errors
        words = [w.strip("[]\"*") for w in content.strip().split()[:6] if w.strip("[]\"*")]
        if not words:
            return []
        prefix_query = " ".join(w + "*" for w in words)
        return self.search_fts(prefix_query, limit=limit)

    # ------------------------------------------------------------------
    # Top-importance / L1 Essential Story
    # ------------------------------------------------------------------

    def get_top_importance(
        self,
        n: int = 8,
        max_chars: int = 1200,
        snippet_limit: int = 200,
    ) -> str:
        """Generate the L1 Essential Story from the highest-importance drawers.

        Groups by room, truncates snippets, and respects a character budget.
        Ranking uses importance * confidence boosted by recall frequency.
        """
        rows = self.conn.execute(
            """
            SELECT d.content, d.wing, d.room, d.importance, d.confidence,
                   d.source_file, COALESCE(re.recall_count, 0) AS recall_count
            FROM drawers d
            LEFT JOIN (
                SELECT drawer_id, COUNT(*) AS recall_count
                FROM recall_events
                GROUP BY drawer_id
            ) re ON re.drawer_id = d.id
            WHERE d.archived = 0
            ORDER BY (d.importance * d.confidence
                      * (1.0 + LN(1.0 + COALESCE(re.recall_count, 0)) * 0.3)) DESC
            LIMIT ?
            """,
            (n * 3,),  # over-fetch so grouping has enough material
        ).fetchall()

        if not rows:
            return ""

        seen_rooms: set[str] = set()
        parts: list[str] = []
        total_len = 0

        for row in rows:
            room = row["room"]
            if room in seen_rooms:
                continue
            seen_rooms.add(room)

            snippet = row["content"].strip()
            if len(snippet) > snippet_limit:
                snippet = snippet[:snippet_limit] + "…"
            line = f"[{row['wing']}/{room}] (★{row['importance']}) {snippet}"
            line_len = len(line) + 1  # +1 for newline

            if total_len + line_len > max_chars:
                parts.append("… (use palace_search for more)")
                break

            parts.append(line)
            total_len += line_len

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Taxonomy
    # ------------------------------------------------------------------

    def list_wings(self) -> List[Dict[str, Any]]:
        """Return all wings with drawer counts, sorted by count desc."""
        rows = self.conn.execute(
            "SELECT wing, COUNT(*) as count FROM drawers GROUP BY wing ORDER BY count DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def list_rooms(self, wing: str) -> List[Dict[str, Any]]:
        """Return rooms within a wing with drawer counts."""
        rows = self.conn.execute(
            "SELECT room, COUNT(*) as count FROM drawers WHERE wing = ? GROUP BY room",
            (wing,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_taxonomy(self) -> List[Dict[str, Any]]:
        """Full wing → room → count tree."""
        rows = self.conn.execute(
            """
            SELECT wing, room, COUNT(*) as count
            FROM drawers
            GROUP BY wing, room
            ORDER BY wing, count DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Mining tracking
    # ------------------------------------------------------------------

    def file_already_mined(self, source_key: str) -> bool:
        """Check whether a source has already been mined."""
        row = self.conn.execute(
            "SELECT 1 FROM mined_sources WHERE source_key = ?", (source_key,)
        ).fetchone()
        return row is not None

    def get_mined_mtime(self, source_key: str) -> float:
        """Get the mtime stored for a mined source. Returns 0 if not found."""
        row = self.conn.execute(
            "SELECT mtime FROM mined_sources WHERE source_key = ?", (source_key,)
        ).fetchone()
        return row[0] if row else 0.0

    def mark_mined(self, source_key: str, mtime: float = 0) -> None:
        """Record that a source has been mined."""
        self.conn.execute(
            "INSERT OR REPLACE INTO mined_sources (source_key, mtime) VALUES (?, ?)",
            (source_key, mtime),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the palace."""
        total = self.conn.execute("SELECT COUNT(*) FROM drawers").fetchone()[0]
        archived = self.conn.execute("SELECT COUNT(*) FROM drawers WHERE archived = 1").fetchone()[0]
        wings = self.conn.execute("SELECT COUNT(DISTINCT wing) FROM drawers").fetchone()[0]
        rooms = self.conn.execute("SELECT COUNT(DISTINCT room) FROM drawers").fetchone()[0]
        latest = self.conn.execute(
            "SELECT filed_at FROM drawers ORDER BY filed_at DESC LIMIT 1"
        ).fetchone()
        return {
            "total_drawers": total,
            "active_drawers": total - archived,
            "archived_drawers": archived,
            "wings_count": wings,
            "rooms_count": rooms,
            "latest_filed_at": latest[0] if latest else None,
        }

    def trim_drawers(self, max_drawers: int = 500) -> int:
        """Archive lowest-scoring drawers when total exceeds max_drawers.

        Score = confidence * importance + time_decay.
        Archives the bottom 10% of excess drawers (soft delete).
        Archived drawers are excluded from FTS and search by default.

        Returns the number of newly archived drawers.
        """
        if max_drawers <= 0:
            return 0

        active_count = self.conn.execute(
            "SELECT COUNT(*) FROM drawers WHERE archived = 0"
        ).fetchone()[0]

        if active_count <= max_drawers:
            return 0

        excess = active_count - max_drawers
        # Archive bottom 10% of excess, minimum 1
        to_archive = max(1, excess // 10)

        # Calculate time_decay: older drawers get slight penalty
        # filed_at is ISO text; we use julianday for numeric comparison
        rows = self.conn.execute(
            """
            SELECT rowid FROM drawers
            WHERE archived = 0
            ORDER BY (
                confidence * importance
                - (julianday('now') - julianday(filed_at)) * 0.01
            ) ASC
            LIMIT ?
            """,
            (to_archive,),
        ).fetchall()

        if not rows:
            return 0

        rowids = [r["rowid"] for r in rows]
        placeholders = ",".join("?" * len(rowids))

        # Mark as archived
        self.conn.execute(
            f"UPDATE drawers SET archived = 1 WHERE rowid IN ({placeholders})",
            rowids,
        )

        # Remove from FTS (content= sync doesn't auto-remove on UPDATE)
        for rid in rowids:
            self.conn.execute(
                "INSERT INTO drawers_fts(drawers_fts, rowid) VALUES ('delete', ?)",
                (rid,),
            )

        self.conn.commit()
        logger.info(
            "Palace trimmed %d drawers (active: %d → %d, max: %d)",
            len(rowids), active_count, active_count - len(rowids), max_drawers,
        )
        return len(rowids)

    # ------------------------------------------------------------------
    # 增强一: Drawer Links (交叉引用)
    # ------------------------------------------------------------------

    def add_link(self, source_id: str, target_id: str, weight: float = 1.0) -> None:
        """Add a bidirectional link between two drawers.

        Silently skips self-links and duplicates.
        """
        if source_id == target_id:
            return
        self.conn.execute(
            """INSERT OR IGNORE INTO drawer_links (source_id, target_id, weight)
               VALUES (?, ?, ?)""",
            (source_id, target_id, weight),
        )
        self.conn.execute(
            """INSERT OR IGNORE INTO drawer_links (source_id, target_id, weight)
               VALUES (?, ?, ?)""",
            (target_id, source_id, weight),
        )
        self.conn.commit()
        logger.debug("add_link %s <-> %s (weight=%.1f)", source_id, target_id, weight)

    def search_fts_with_links(
        self,
        query: str,
        wing: str | None = None,
        room: str | None = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """FTS5 search + link expansion.

        First does a normal FTS5 search, then fetches linked drawers for
        each result. Deduplicates by drawer ID.
        """
        # Phase 1: direct FTS5 match
        direct = self.search_fts(query, wing=wing, room=room, limit=limit)
        seen_ids = {r["id"] for r in direct}
        results = list(direct)

        if len(results) >= limit:
            return results

        # Phase 2: expand via links
        id_list = list(seen_ids)
        ph = ",".join("?" for _ in id_list)
        linked_rows = self.conn.execute(
            f"""
            SELECT DISTINCT
                CASE WHEN dl.source_id IN ({ph}) THEN dl.target_id ELSE dl.source_id END AS linked_id
            FROM drawer_links dl
            WHERE dl.source_id IN ({ph}) OR dl.target_id IN ({ph})
            """,
            id_list + id_list + id_list,
        ).fetchall()

        remaining = limit - len(results)
        for row in linked_rows:
            if remaining <= 0:
                break
            lid = row["linked_id"]
            if lid in seen_ids:
                continue
            seen_ids.add(lid)
            drawer = self.get_drawer(lid)
            if drawer:
                results.append(drawer)
                remaining -= 1

        return results

    # ------------------------------------------------------------------
    # 增强二: Cooccurrence (共现提取)
    # ------------------------------------------------------------------

    def add_cooccurrence(
        self,
        entities: List[str],
        context: str = "",
        session_id: str = "",
        weight: float = 1.0,
    ) -> None:
        """Record a co-occurrence group of entities.

        Skips groups with <2 entities. Deduplicates entity names (lowercase).
        """
        if not entities or len(entities) < 2:
            return
        # Normalize and deduplicate
        normalized = list(dict.fromkeys(e.lower().strip() for e in entities if e.strip()))
        if len(normalized) < 2:
            return

        group_id = hashlib.md5(
            "|".join(sorted(normalized)).encode("utf-8")
        ).hexdigest()[:16]
        group_id = f"co_{group_id}"

        entities_json = json.dumps(normalized, ensure_ascii=False)
        self.conn.execute(
            """INSERT OR IGNORE INTO cooccurrence_groups
               (id, entities, context, session_id, weight)
               VALUES (?, ?, ?, ?, ?)""",
            (group_id, entities_json, context, session_id, weight),
        )
        for entity in normalized:
            self.conn.execute(
                """INSERT OR IGNORE INTO cooccurrence_items (entity, group_id)
                   VALUES (?, ?)""",
                (entity, group_id),
            )
        self.conn.commit()
        logger.debug("add_cooccurrence %s (entities=%d)", group_id, len(normalized))

    def find_related_entities(self, entity: str, limit: int = 10) -> List[str]:
        """Find entities that co-occur with the given entity.

        Returns a deduplicated list of related entity names, sorted by
        frequency (most common first).
        """
        entity_lower = entity.lower().strip()
        rows = self.conn.execute(
            """
            SELECT ci.entity, COUNT(*) as freq
            FROM cooccurrence_items ci
            JOIN cooccurrence_items ci2 ON ci.group_id = ci2.group_id
            WHERE ci2.entity = ? AND ci.entity != ?
            GROUP BY ci.entity
            ORDER BY freq DESC
            LIMIT ?
            """,
            (entity_lower, entity_lower, limit),
        ).fetchall()
        return [r["entity"] for r in rows]

    # ------------------------------------------------------------------
    # P2 预留: 记忆进化
    # ------------------------------------------------------------------

    def mark_superseded(self, old_id: str, new_id: str) -> bool:
        """Mark an old drawer as superseded by a new one (P2 预留).

        Returns True if the update was applied.
        """
        cur = self.conn.execute(
            "UPDATE drawers SET superseded_by = ? WHERE id = ? AND superseded_by IS NULL",
            (new_id, old_id),
        )
        self.conn.commit()
        if cur.rowcount > 0:
            logger.debug("mark_superseded %s -> %s", old_id, new_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Dreaming: Recall event tracking
    # ------------------------------------------------------------------

    def record_recall(self, drawer_id: str, session_id: str,
                      query: str = "", bm25_rank: float = 0.0) -> None:
        """Record a search hit event (append-only, for Dreaming scoring)."""
        self.conn.execute(
            "INSERT INTO recall_events (drawer_id, session_id, query, bm25_rank) "
            "VALUES (?, ?, ?, ?)",
            (drawer_id, session_id, query, bm25_rank),
        )
        self.conn.commit()

    def get_recall_stats(self, drawer_id: str) -> Dict[str, Any]:
        """Get recall statistics for a drawer. Returns zero values if no records."""
        row = self.conn.execute(
            """
            SELECT drawer_id,
                   COUNT(*) as recall_count,
                   COUNT(DISTINCT session_id) as recall_sessions,
                   COUNT(DISTINCT date(recalled_at)) as recall_days,
                   MAX(recalled_at) as last_recalled_at,
                   AVG(bm25_rank) as avg_bm25_rank
            FROM recall_events
            WHERE drawer_id = ?
            GROUP BY drawer_id
            """,
            (drawer_id,),
        ).fetchone()
        if not row:
            return {
                "recall_count": 0, "recall_sessions": 0, "recall_days": 0,
                "last_recalled_at": None, "avg_bm25_rank": 0.0,
            }
        return dict(row)

    def get_total_recall_count(self) -> int:
        """Get total recall_events count (for cold start detection)."""
        return self.conn.execute("SELECT COUNT(*) FROM recall_events").fetchone()[0]

    def cleanup_old_recall_events(self, max_age_days: int = 90) -> int:
        """Delete recall_events older than max_age_days. Returns deleted count."""
        cur = self.conn.execute(
            "DELETE FROM recall_events WHERE julianday('now') - julianday(recalled_at) > ?",
            (max_age_days,),
        )
        self.conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Dreaming: Drawer management helpers
    # ------------------------------------------------------------------

    def merge_drawers(self, source_id: str, target_id: str) -> bool:
        """Merge source drawer into target (simplified, for 0.95 threshold dedup).

        1. Redirect drawer_links from source to target
        2. Reassign recall_events from source to target
        3. Mark source as superseded_by = target_id
        Does NOT merge content (at 0.95 threshold, differences are negligible).
        """
        # Redirect links
        self.conn.execute(
            "UPDATE drawer_links SET source_id = ? WHERE source_id = ?",
            (target_id, source_id),
        )
        self.conn.execute(
            "UPDATE drawer_links SET target_id = ? WHERE target_id = ?",
            (target_id, source_id),
        )
        # Reassign recall events
        self.conn.execute(
            "UPDATE recall_events SET drawer_id = ? WHERE drawer_id = ?",
            (target_id, source_id),
        )
        # Mark superseded
        cur = self.conn.execute(
            "UPDATE drawers SET superseded_by = ? WHERE id = ? AND superseded_by IS NULL",
            (target_id, source_id),
        )
        self.conn.commit()
        if cur.rowcount > 0:
            logger.debug("merge_drawers %s -> %s", source_id, target_id)
            return True
        return False

    def get_recent_drawers(self, lookback_days: int = 2,
                           limit: int = 200) -> List[Dict[str, Any]]:
        """Get recent drawers (by filed_at, using julianday)."""
        rows = self.conn.execute(
            """
            SELECT * FROM drawers
            WHERE archived = 0 AND superseded_by IS NULL
              AND julianday('now') - julianday(filed_at) <= ?
            ORDER BY filed_at DESC LIMIT ?
            """,
            (lookback_days, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_active_drawers(self, include_superseded: bool = False,
                           wing: str | None = None,
                           limit: int = 500) -> List[Dict[str, Any]]:
        """Get active drawers (non-archived, optionally excluding superseded)."""
        clauses = ["archived = 0"]
        params: list[Any] = []
        if not include_superseded:
            clauses.append("superseded_by IS NULL")
        if wing is not None:
            clauses.append("wing = ?")
            params.append(wing)
        where = " WHERE " + " AND ".join(clauses)
        sql = f"SELECT * FROM drawers{where} ORDER BY importance * confidence DESC, filed_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_link_count(self, drawer_id: str) -> int:
        """Get link count for a drawer (bidirectional)."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM drawer_links WHERE source_id = ? OR target_id = ?",
            (drawer_id, drawer_id),
        ).fetchone()
        return row[0]

    def bulk_update_scores(self, updates: List[Dict[str, Any]]) -> int:
        """Batch update importance/confidence/memory_type for drawers.
        
        Each update dict: {"id": "...", "importance": 4.0, "confidence": 0.9}
        Only provided keys are updated.
        """
        allowed = {"importance", "confidence", "memory_type"}
        count = 0
        for u in updates:
            drawer_id = u.get("id")
            if not drawer_id:
                continue
            sets = []
            vals = []
            for k in allowed:
                if k in u:
                    sets.append(f"{k} = ?")
                    vals.append(u[k])
            if not sets:
                continue
            vals.append(drawer_id)
            sql = f"UPDATE drawers SET {', '.join(sets)} WHERE id = ?"
            cur = self.conn.execute(sql, vals)
            count += cur.rowcount
        self.conn.commit()
        return count

    def archive_drawer(self, drawer_id: str) -> bool:
        """Archive a drawer (soft delete, remove from FTS)."""
        # Get rowid first for FTS removal
        row = self.conn.execute(
            "SELECT rowid FROM drawers WHERE id = ?", (drawer_id,)
        ).fetchone()
        if not row:
            return False
        rid = row["rowid"]
        self.conn.execute("UPDATE drawers SET archived = 1 WHERE id = ?", (drawer_id,))
        # Remove from FTS (content= sync doesn't auto-remove on UPDATE)
        self.conn.execute(
            "INSERT INTO drawers_fts(drawers_fts, rowid) VALUES ('delete', ?)", (rid,)
        )
        self.conn.commit()
        logger.debug("archive_drawer %s", drawer_id)
        return True

    def get_dreaming_state(self) -> Dict[str, Any]:
        """Get Dreaming global state for scoring decisions."""
        active = self.conn.execute(
            "SELECT COUNT(*) FROM drawers WHERE archived = 0 AND superseded_by IS NULL"
        ).fetchone()[0]
        high_imp = self.conn.execute(
            "SELECT COUNT(*) FROM drawers WHERE archived = 0 AND importance >= 4.0"
        ).fetchone()[0]
        archived = self.conn.execute(
            "SELECT COUNT(*) FROM drawers WHERE archived = 1"
        ).fetchone()[0]
        superseded = self.conn.execute(
            "SELECT COUNT(*) FROM drawers WHERE superseded_by IS NOT NULL"
        ).fetchone()[0]
        avg_stats = self.conn.execute(
            "SELECT AVG(importance) as avg_imp, AVG(confidence) as avg_conf "
            "FROM drawers WHERE archived = 0"
        ).fetchone()
        total_recalls = self.get_total_recall_count()

        # Days since first dreaming run
        first_run = self.conn.execute(
            "SELECT started_at FROM dreaming_runs ORDER BY started_at ASC LIMIT 1"
        ).fetchone()
        days_since_first = 0
        if first_run:
            try:
                days_since_first = self.conn.execute(
                    "SELECT CAST(julianday('now') - julianday(?) AS INTEGER)",
                    (first_run[0],),
                ).fetchone()[0]
            except Exception:
                days_since_first = 0

        return {
            "active_count": active,
            "high_importance_count": high_imp,
            "avg_importance": round(avg_stats["avg_imp"] or 0, 3),
            "avg_confidence": round(avg_stats["avg_conf"] or 0, 3),
            "total_recall_count": total_recalls,
            "archived_count": archived,
            "superseded_count": superseded,
            "days_since_first_dreaming_run": days_since_first,
        }

    # ------------------------------------------------------------------
    # Dreaming: Signal & run logging
    # ------------------------------------------------------------------

    def add_dreaming_signal(self, drawer_id: str, phase: str,
                            strength: float = 1.0) -> None:
        """Record a dreaming phase signal for a drawer."""
        self.conn.execute(
            "INSERT INTO dreaming_signals (drawer_id, phase, signal_strength) "
            "VALUES (?, ?, ?)",
            (drawer_id, phase, strength),
        )
        self.conn.commit()

    def save_dreaming_run(self, phase: str, report: Dict[str, Any],
                          status: str = "success") -> int:
        """Save a dreaming run log. Returns run_id."""
        self.conn.execute(
            """
            INSERT INTO dreaming_runs
                (phase, status, finished_at, duration_ms, report_json,
                 memory_health, drawers_active, drawers_processed,
                 drawers_merged, drawers_promoted, drawers_archived)
            VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                phase, status,
                report.get("duration_ms", 0), json.dumps(report, ensure_ascii=False, default=str),
                report.get("memory_health"),
                report.get("drawers_active"),
                report.get("drawers_processed", 0),
                report.get("drawers_merged", 0),
                report.get("drawers_promoted", 0),
                report.get("drawers_archived", 0),
            ),
        )
        self.conn.commit()
        cur = self.conn.execute("SELECT last_insert_rowid()")
        return cur.fetchone()[0]

    def get_recent_dreaming_runs(self, phase: str | None = None,
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """Query recent dreaming runs."""
        if phase:
            rows = self.conn.execute(
                "SELECT * FROM dreaming_runs WHERE phase = ? ORDER BY started_at DESC LIMIT ?",
                (phase, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM dreaming_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_last_dreaming_run(self, phase: str) -> Dict[str, Any] | None:
        """Get the most recent run for a given phase."""
        row = self.conn.execute(
            "SELECT * FROM dreaming_runs WHERE phase = ? ORDER BY started_at DESC LIMIT 1",
            (phase,),
        ).fetchone()
        return dict(row) if row else None

    def get_dreaming_signal_counts(self, drawer_id: str) -> Dict[str, int]:
        """Get dreaming signal counts per phase for a drawer."""
        rows = self.conn.execute(
            "SELECT phase, COUNT(*) as cnt FROM dreaming_signals "
            "WHERE drawer_id = ? GROUP BY phase",
            (drawer_id,),
        ).fetchall()
        result = {"light": 0, "deep": 0, "rem": 0}
        for r in rows:
            result[r["phase"]] = r["cnt"]

    # ------------------------------------------------------------------
    # 增强三: Hybrid Search (FTS5 + Vector RRF)
    # ------------------------------------------------------------------

    def search_hybrid(
        self,
        query: str,
        wing: str | None = None,
        room: str | None = None,
        limit: int = 5,
        mode: str = "auto",
        vector_store=None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: FTS5 BM25 + vector kNN with RRF fusion.

        Args:
            query: Search query string.
            wing: Optional wing filter.
            room: Optional room filter.
            limit: Max results to return.
            mode: "auto" | "fts" | "vector" | "hybrid".
                auto: hybrid if vector_store available and has embeddings,
                      else fts fallback.
                fts: pure FTS5 (same as search_fts).
                vector: pure vector kNN (debug).
                hybrid: always try both, merge with RRF.
            vector_store: Optional VectorStore instance for vector search.

        Returns:
            List of drawer dicts, sorted by RRF score descending.
        """
        # Validate mode
        mode = mode.lower()
        if mode not in ("auto", "fts", "vector", "hybrid"):
            mode = "auto"

        # Determine effective mode
        if mode == "auto":
            if vector_store is None or vector_store.get_embedding_count() == 0:
                mode = "fts"
            else:
                mode = "hybrid"

        # Pure FTS mode
        if mode == "fts":
            return self.search_fts_with_links(query, wing=wing, room=room, limit=limit)

        # Pure vector mode (debug)
        if mode == "vector":
            if vector_store is None:
                return []
            return self._search_vector_only(query, vector_store, limit=limit)

        # Hybrid mode: FTS5 + Vector kNN → RRF fusion
        return self._search_rrf(query, wing=wing, room=room, limit=limit,
                                vector_store=vector_store)

    def _search_vector_only(
        self,
        query: str,
        vector_store,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Pure vector search — for debugging."""
        if not vector_store._client.available:
            return []

        try:
            q_emb = vector_store._client.embed_text(query[:500])
        except Exception as e:
            logger.warning("Vector search: embedding failed: %s", e)
            return []

        vec_results = vector_store.search_vectors(q_emb, k=limit)
        results = []
        for vr in vec_results:
            drawer = self.get_drawer(vr["drawer_id"])
            if drawer:
                drawer["_vec_distance"] = vr["distance"]
                results.append(drawer)
        return results

    def _search_rrf(
        self,
        query: str,
        wing: str | None,
        room: str | None,
        limit: int,
        vector_store,
        k: int = 60,
        fetch_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """RRF fusion of FTS5 and vector search results.

        RRF formula: score(d) = w_fts / (k + fts_rank) + w_vec / (k + vec_rank)
        where ranks start at 1.
        """
        # Phase 1: FTS5 BM25 (fetch more than limit for RRF)
        fts_results = self.search_fts(query, wing=wing, room=room, limit=fetch_k)
        fts_rank_map: Dict[str, int] = {}
        for i, r in enumerate(fts_results):
            fts_rank_map[r["id"]] = i + 1  # rank from 1

        # Phase 2: Vector kNN
        vec_rank_map: Dict[str, int] = {}
        vec_dist_map: Dict[str, float] = {}
        if vector_store and vector_store._client.available:
            try:
                q_emb = vector_store._client.embed_text(query[:500])
                vec_results = vector_store.search_vectors(q_emb, k=fetch_k)
                for i, vr in enumerate(vec_results):
                    vec_rank_map[vr["drawer_id"]] = i + 1
                    vec_dist_map[vr["drawer_id"]] = vr["distance"]
            except Exception as e:
                logger.warning("RRF vector search failed (fts-only fallback): %s", e)

        # Phase 3: Merge with RRF
        all_ids = set(fts_rank_map.keys()) | set(vec_rank_map.keys())
        scored: List[Tuple[str, float]] = []
        for did in all_ids:
            score = 0.0
            if did in fts_rank_map:
                score += 1.0 / (k + fts_rank_map[did])
            if did in vec_rank_map:
                score += 1.0 / (k + vec_rank_map[did])
            scored.append((did, score))

        # Sort by RRF score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Fetch full drawer dicts for top results
        results: List[Dict[str, Any]] = []
        for did, score in scored[:limit]:
            drawer = self.get_drawer(did)
            if drawer:
                drawer["_rrf_score"] = round(score, 6)
                if did in fts_rank_map:
                    drawer["_fts_rank"] = fts_rank_map[did]
                if did in vec_rank_map:
                    drawer["_vec_rank"] = vec_rank_map[did]
                    drawer["_vec_distance"] = vec_dist_map[did]
                results.append(drawer)

        # Phase 4: Link expansion if we have room
        if len(results) < limit:
            seen_ids = {r["id"] for r in results}
            id_list = list(seen_ids)
            ph = ",".join("?" for _ in id_list)
            linked_rows = self.conn.execute(
                f"""
                SELECT DISTINCT
                    CASE WHEN dl.source_id IN ({ph}) THEN dl.target_id
                         ELSE dl.source_id END AS linked_id
                FROM drawer_links dl
                WHERE dl.source_id IN ({ph}) OR dl.target_id IN ({ph})
                """,
                id_list + id_list + id_list,
            ).fetchall()

            remaining = limit - len(results)
            for row in linked_rows:
                if remaining <= 0:
                    break
                lid = row["linked_id"]
                if lid in seen_ids:
                    continue
                seen_ids.add(lid)
                drawer = self.get_drawer(lid)
                if drawer:
                    drawer["_rrf_score"] = 0.0
                    results.append(drawer)
                    remaining -= 1

        return results

    def get_feedback_weight(self, drawer_id: str) -> float:
        """Get feedback_weight for a drawer. Returns 0.5 (neutral) if column missing."""
        try:
            row = self.conn.execute(
                "SELECT feedback_weight FROM drawers WHERE id = ?",
                (drawer_id,),
            ).fetchone()
            return row["feedback_weight"] if row else 0.5
        except Exception:
            return 0.5  # Column may not exist yet (Phase 2 not applied)

    def set_feedback_weight(self, drawer_id: str, weight: float) -> None:
        """Set feedback_weight for a drawer. No-op if column doesn't exist."""
        try:
            self.conn.execute(
                "UPDATE drawers SET feedback_weight = ? WHERE id = ?",
                (weight, drawer_id),
            )
            self.conn.commit()
        except Exception as e:
            logger.debug("set_feedback_weight failed (column may not exist): %s", e)
