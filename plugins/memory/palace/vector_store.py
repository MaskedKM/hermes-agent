"""Vector store — sqlite-vec operations for drawer embeddings.

Manages the vec0 virtual table, drawer_vector_map mapping table,
and embedding_cache for deduplication.

Design notes:
- vec0 does not support INSERT OR REPLACE (issue #259).
  We use DELETE + INSERT for upserts.
- vec0 uses implicit rowid as PK; we maintain a separate
  drawer_vector_map table for drawer_id ↔ rowid mapping.
- Embeddings are cached by content hash to avoid redundant API calls.
"""

from __future__ import annotations

import logging
import sqlite3
import sqlite_vec
from typing import Any, Dict, List, Optional, Tuple

from .embedding_client import (
    EmbeddingClient,
    content_hash,
    deserialize_f32,
    serialize_f32,
)

logger = logging.getLogger(__name__)


class VectorStore:
    """SQLite-vec backed vector store for drawer embeddings."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedding_client: EmbeddingClient,
        dimension: int = 2048,
    ) -> None:
        self._conn = conn
        self._client = embedding_client
        self._dimension = dimension
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create vec0 virtual table and mapping tables if not exist."""
        # Load sqlite-vec extension
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)

        # vec0 virtual table — note: dimension is baked into CREATE TABLE
        try:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS drawer_vectors "
                f"USING vec0(embedding float[{self._dimension}])"
            )
        except Exception as e:
            # If table exists with different dimension, we need to drop and recreate
            logger.warning("vec0 table creation issue (may need migration): %s", e)

        self._conn.executescript("""
            -- drawer_id → rowid mapping
            CREATE TABLE IF NOT EXISTS drawer_vector_map (
                drawer_id TEXT PRIMARY KEY,
                vector_rowid INTEGER NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            -- Embedding cache: avoid re-embedding identical content
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        self._conn.commit()
        logger.debug("VectorStore tables ensured (dim=%d)", self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._client.model

    def upsert_embedding(self, drawer_id: str, embedding: List[float]) -> None:
        """Insert or update an embedding for a drawer.

        Uses DELETE + INSERT workaround because vec0 doesn't support
        INSERT OR REPLACE (issue #259).
        """
        blob = serialize_f32(embedding)

        existing = self._conn.execute(
            "SELECT vector_rowid FROM drawer_vector_map WHERE drawer_id = ?",
            (drawer_id,),
        ).fetchone()

        if existing:
            rowid = existing["vector_rowid"]
            self._conn.execute(
                "DELETE FROM drawer_vectors WHERE rowid = ?", (rowid,)
            )
            self._conn.execute(
                "INSERT INTO drawer_vectors(rowid, embedding) VALUES (?, vec_f32(?))",
                (rowid, blob),
            )
        else:
            cursor = self._conn.execute(
                "INSERT INTO drawer_vectors(embedding) VALUES (vec_f32(?))",
                (blob,),
            )
            rowid = cursor.lastrowid
            self._conn.execute(
                "INSERT INTO drawer_vector_map (drawer_id, vector_rowid, model) "
                "VALUES (?, ?, ?)",
                (drawer_id, rowid, self._client.model),
            )

        self._conn.commit()

    def delete_embedding(self, drawer_id: str) -> bool:
        """Delete an embedding for a drawer. Returns True if deleted."""
        existing = self._conn.execute(
            "SELECT vector_rowid FROM drawer_vector_map WHERE drawer_id = ?",
            (drawer_id,),
        ).fetchone()

        if not existing:
            return False

        rowid = existing["vector_rowid"]
        self._conn.execute(
            "DELETE FROM drawer_vectors WHERE rowid = ?", (rowid,)
        )
        self._conn.execute(
            "DELETE FROM drawer_vector_map WHERE drawer_id = ?", (drawer_id,)
        )
        self._conn.commit()
        return True

    def get_embedding(self, drawer_id: str) -> Optional[List[float]]:
        """Get the embedding for a drawer. Returns None if not found."""
        row = self._conn.execute(
            """
            SELECT v.embedding
            FROM drawer_vectors v
            JOIN drawer_vector_map m ON m.vector_rowid = v.rowid
            WHERE m.drawer_id = ?
            """,
            (drawer_id,),
        ).fetchone()

        if not row:
            return None
        return deserialize_f32(row["embedding"])

    def has_embedding(self, drawer_id: str) -> bool:
        """Check if a drawer has an embedding."""
        row = self._conn.execute(
            "SELECT 1 FROM drawer_vector_map WHERE drawer_id = ?",
            (drawer_id,),
        ).fetchone()
        return row is not None

    def search_vectors(
        self,
        query_embedding: List[float],
        k: int = 20,
    ) -> List[Dict[str, Any]]:
        """kNN search. Returns list of {drawer_id, distance} dicts."""
        blob = serialize_f32(query_embedding)

        rows = self._conn.execute(
            """
            SELECT m.drawer_id, v.distance
            FROM drawer_vectors v
            JOIN drawer_vector_map m ON m.vector_rowid = v.rowid
            WHERE v.embedding MATCH vec_f32(?)
              AND k = ?
            ORDER BY v.distance
            """,
            (blob, k),
        ).fetchall()

        return [dict(r) for r in rows]

    def get_missing_embeddings(
        self, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Get drawers that don't have embeddings yet.

        Returns list of {id, content} dicts.
        """
        rows = self._conn.execute(
            """
            SELECT d.id, d.content
            FROM drawers d
            LEFT JOIN drawer_vector_map m ON m.drawer_id = d.id
            WHERE m.drawer_id IS NULL
              AND d.archived = 0
            ORDER BY d.importance DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_embedding_count(self) -> int:
        """Count total embeddings stored."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM drawer_vector_map"
        ).fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Look up cached embedding by content hash."""
        h = content_hash(text)
        row = self._conn.execute(
            "SELECT embedding FROM embedding_cache WHERE content_hash = ?",
            (h,),
        ).fetchone()
        if row:
            return deserialize_f32(row["embedding"])
        return None

    def put_cached_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding for future reuse."""
        h = content_hash(text)
        blob = serialize_f32(embedding)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO embedding_cache (content_hash, model, embedding)
            VALUES (?, ?, ?)
            """,
            (h, self._client.model, blob),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def batch_embed(
        self,
        items: List[Dict[str, Any]],
        *,
        content_key: str = "content",
        id_key: str = "id",
        on_progress=None,
    ) -> int:
        """Embed a batch of items, using cache for known content.

        Args:
            items: List of dicts, each must have content_key and id_key.
            content_key: Key to read text from (default "content").
            id_key: Key to read drawer_id from (default "id").
            on_progress: Optional callback(completed, total) for progress.

        Returns:
            Number of items successfully embedded.
        """
        if not items or not self._client.available:
            return 0

        # Separate cached vs uncached
        to_embed: List[Dict[str, Any]] = []
        cached_count = 0

        for item in items:
            text = item[content_key][:500]  # Truncate for hash consistency
            cached = self.get_cached_embedding(text)
            if cached is not None:
                self.upsert_embedding(item[id_key], cached)
                cached_count += 1
            else:
                to_embed.append(item)

        if not to_embed:
            logger.info("All %d items served from embedding cache", len(items))
            return cached_count

        # Batch embed uncached items
        texts = [item[content_key][:500] for item in to_embed]
        try:
            embeddings = self._client.embed_texts(texts)
        except Exception as e:
            logger.error("Batch embedding failed: %s", e)
            return cached_count

        embedded = 0
        for item, emb in zip(to_embed, embeddings):
            try:
                text = item[content_key][:500]
                self.upsert_embedding(item[id_key], emb)
                self.put_cached_embedding(text, emb)
                embedded += 1
                if on_progress:
                    on_progress(cached_count + embedded, len(items))
            except Exception as e:
                logger.warning("Failed to embed %s: %s", item[id_key], e)

        total = cached_count + embedded
        logger.info(
            "Batch embed: %d total (%d cached, %d new)",
            total, cached_count, embedded,
        )
        return total
