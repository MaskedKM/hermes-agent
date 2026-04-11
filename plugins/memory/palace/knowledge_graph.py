"""
Temporal knowledge graph with entity_registry for disambiguation.

Tracks entity relationships over time, supporting time-travel queries
(e.g. "what did Alice work on in January?").

Uses a SEPARATE database from drawers.db (knowledge_graph.db).
"""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    type       TEXT DEFAULT 'unknown',
    properties TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS triples (
    id            TEXT PRIMARY KEY,
    subject       TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    object        TEXT NOT NULL,
    valid_from    TEXT DEFAULT (datetime('now')),
    valid_to      TEXT DEFAULT NULL,
    confidence    REAL DEFAULT 1.0,
    source_closet TEXT DEFAULT '',
    source_file   TEXT DEFAULT '',
    extracted_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS entity_registry (
    name         TEXT NOT NULL,
    registry_type TEXT NOT NULL,
    aliases      TEXT DEFAULT '[]',
    properties   TEXT DEFAULT '{}',
    confidence   REAL DEFAULT 1.0,
    source       TEXT DEFAULT '',
    updated_at   TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (name, registry_type)
);

CREATE INDEX IF NOT EXISTS idx_triples_subject  ON triples(subject);
CREATE INDEX IF NOT EXISTS idx_triples_object   ON triples(object);
CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
CREATE INDEX IF NOT EXISTS idx_triples_valid    ON triples(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_entities_name    ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type    ON entities(type);
"""


class PalaceKnowledgeGraph:
    """Temporal knowledge graph backed by SQLite."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.debug("Knowledge graph opened at %s", db_path)

    # ------------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_id(name: str) -> str:
        """Normalize entity name to a stable ID."""
        return name.lower().replace(" ", "_").replace("'", "")

    @staticmethod
    def _triple_id(subject: str, predicate: str, obj: str,
                   valid_from: str = "") -> str:
        """Generate a deterministic triple ID (unique per version)."""
        raw = f"{subject}_{predicate}_{obj}_{valid_from}"
        digest = hashlib.md5(raw.encode()).hexdigest()[:8]
        return f"t_{subject}_{predicate}_{digest}"

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    def add_entity(self, name: str, entity_type: str = "unknown",
                   properties: dict = None) -> str:
        """Insert an entity (idempotent). Returns entity_id."""
        eid = self._entity_id(name)
        props = json.dumps(properties or {})
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR IGNORE INTO entities (id, name, type, properties, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (eid, name, entity_type, props, now),
        )
        self._conn.commit()
        return eid

    def get_entity(self, name: str) -> Optional[dict]:
        """Lookup entity by normalized name."""
        eid = self._entity_id(name)
        row = self._conn.execute(
            "SELECT * FROM entities WHERE id = ?", (eid,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["properties"] = json.loads(d["properties"])
        return d

    def search_entities(self, query: str, limit: int = 10) -> list:
        """LIKE search on entity name."""
        rows = self._conn.execute(
            "SELECT * FROM entities WHERE name LIKE ? LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["properties"] = json.loads(d["properties"])
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Triple operations
    # ------------------------------------------------------------------

    def add_triple(self, subject: str, predicate: str, object: str,
                   valid_from: str = None, confidence: float = 1.0,
                   source_closet: str = "", source_file: str = "",
                   entity_types: dict = None) -> str:
        """Add a temporal triple. Auto-creates subject/object entities.

        Dedup: if the same (subject, predicate, object) already exists with
        valid_to IS NULL, the old triple is invalidated before inserting.

        Returns the new triple_id.
        """
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(object)
        entity_types = entity_types or {}
        now = datetime.now(timezone.utc).isoformat()
        vf = valid_from or now

        # Auto-create entities
        self.add_entity(subject, entity_types.get(subject, "unknown"))
        self.add_entity(object, entity_types.get(object, "unknown"))

        # Dedup: invalidate existing active triple with same s/p/o
        self._conn.execute(
            "UPDATE triples SET valid_to = ? "
            "WHERE subject = ? AND predicate = ? AND object = ? AND valid_to IS NULL",
            (now, sub_id, predicate, obj_id),
        )

        # Insert new triple
        tid = self._triple_id(sub_id, predicate, obj_id, vf)
        self._conn.execute(
            "INSERT OR IGNORE INTO triples "
            "(id, subject, predicate, object, valid_from, valid_to, "
            " confidence, source_closet, source_file, extracted_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)",
            (tid, sub_id, predicate, obj_id, vf, confidence, source_closet, source_file, now),
        )
        self._conn.commit()
        return tid

    def invalidate_triple(self, triple_id: str = None, subject: str = None,
                          predicate: str = None, object: str = None) -> bool:
        """Set valid_to = now on matching active triple(s).

        Lookup by triple_id OR by subject+predicate+object.
        Returns True if any row was updated.
        """
        now = datetime.now(timezone.utc).isoformat()

        if triple_id:
            cur = self._conn.execute(
                "UPDATE triples SET valid_to = ? WHERE id = ? AND valid_to IS NULL",
                (now, triple_id),
            )
        elif subject and predicate and object:
            sub_id = self._entity_id(subject)
            obj_id = self._entity_id(object)
            cur = self._conn.execute(
                "UPDATE triples SET valid_to = ? "
                "WHERE subject = ? AND predicate = ? AND object = ? AND valid_to IS NULL",
                (now, sub_id, predicate, obj_id),
            )
        else:
            return False

        self._conn.commit()
        return cur.rowcount > 0

    def query_triples(self, entity: str, direction: str = "outgoing",
                      predicate: str = None, as_of: str = None,
                      limit: int = 10) -> list:
        """Query triples involving *entity*.

        direction: 'outgoing' (entity is subject), 'incoming' (entity is object),
                   or 'both'.
        as_of: ISO date for time-travel query (only triples valid at that point).
        """
        eid = self._entity_id(entity)

        # Build WHERE clauses
        conditions = []
        params: list = []

        if direction == "outgoing":
            conditions.append("t.subject = ?")
            params.append(eid)
        elif direction == "incoming":
            conditions.append("t.object = ?")
            params.append(eid)
        else:  # both
            conditions.append("(t.subject = ? OR t.object = ?)")
            params.extend([eid, eid])

        if predicate:
            conditions.append("t.predicate = ?")
            params.append(predicate)

        if as_of:
            conditions.append("t.valid_from <= ?")
            conditions.append("(t.valid_to IS NULL OR t.valid_to > ?)")
            params.extend([as_of, as_of])
        else:
            # Default: only active (currently valid) triples
            conditions.append("t.valid_to IS NULL")

        where = " AND ".join(conditions)
        sql = (
            f"SELECT t.*, "
            f"  s.name AS subject_name, "
            f"  o.name AS object_name "
            f"FROM triples t "
            f"LEFT JOIN entities s ON t.subject = s.id "
            f"LEFT JOIN entities o ON t.object = o.id "
            f"WHERE {where} "
            f"ORDER BY t.valid_from DESC LIMIT ?"
        )
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_entity_timeline(self, entity_name: str, limit: int = 20) -> list:
        """Get all triples involving entity, ordered by valid_from DESC."""
        eid = self._entity_id(entity_name)
        sql = (
            "SELECT t.*, "
            "  s.name AS subject_name, "
            "  o.name AS object_name "
            "FROM triples t "
            "LEFT JOIN entities s ON t.subject = s.id "
            "LEFT JOIN entities o ON t.object = o.id "
            "WHERE t.subject = ? OR t.object = ? "
            "ORDER BY t.valid_from DESC LIMIT ?"
        )
        rows = self._conn.execute(sql, (eid, eid, limit)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Entity Registry (disambiguation)
    # ------------------------------------------------------------------

    def register_entity_alias(self, name: str, alias: str,
                              registry_type: str = "person") -> None:
        """Add *alias* to the entity_registry entry for *name*."""
        row = self._conn.execute(
            "SELECT aliases FROM entity_registry WHERE name = ? AND registry_type = ?",
            (name, registry_type),
        ).fetchone()

        now = datetime.now(timezone.utc).isoformat()

        if row is None:
            aliases = json.dumps([alias])
            self._conn.execute(
                "INSERT INTO entity_registry "
                "(name, registry_type, aliases, updated_at) VALUES (?, ?, ?, ?)",
                (name, registry_type, aliases, now),
            )
        else:
            aliases = json.loads(row["aliases"])
            if alias not in aliases:
                aliases.append(alias)
            self._conn.execute(
                "UPDATE entity_registry SET aliases = ?, updated_at = ? "
                "WHERE name = ? AND registry_type = ?",
                (json.dumps(aliases), now, name, registry_type),
            )

        self._conn.commit()

    def resolve_entity(self, name_or_alias: str) -> Optional[str]:
        """Resolve a name or alias to a canonical entity name.

        1. Check entity_registry for exact match on name or alias.
        2. Fallback: entities table exact match on normalized name.
        Returns canonical name or None.
        """
        # 1. Check registry name
        row = self._conn.execute(
            "SELECT name FROM entity_registry WHERE name = ?", (name_or_alias,)
        ).fetchone()
        if row:
            return row["name"]

        # 2. Check registry aliases
        rows = self._conn.execute(
            "SELECT name, aliases FROM entity_registry"
        ).fetchall()
        for r in rows:
            aliases = json.loads(r["aliases"])
            if name_or_alias in aliases:
                return r["name"]

        # 3. Fallback: entities table
        eid = self._entity_id(name_or_alias)
        row = self._conn.execute(
            "SELECT name FROM entities WHERE id = ?", (eid,)
        ).fetchone()
        if row:
            return row["name"]

        return None

    def get_registry(self, registry_type: str = None) -> list:
        """Query entity_registry, optionally filtered by type."""
        if registry_type:
            rows = self._conn.execute(
                "SELECT * FROM entity_registry WHERE registry_type = ?",
                (registry_type,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM entity_registry"
            ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["aliases"] = json.loads(d["aliases"])
            d["properties"] = json.loads(d["properties"])
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary counts."""
        entity_count = self._conn.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()[0]
        triple_count = self._conn.execute(
            "SELECT COUNT(*) FROM triples"
        ).fetchone()[0]
        active_count = self._conn.execute(
            "SELECT COUNT(*) FROM triples WHERE valid_to IS NULL"
        ).fetchone()[0]
        return {
            "entity_count": entity_count,
            "triple_count": triple_count,
            "active_triple_count": active_count,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.debug("Knowledge graph closed")
