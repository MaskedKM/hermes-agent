#!/usr/bin/env python3
"""Batch-embed all existing Palace drawers into the vector store.

Usage:
    python -m plugins.memory.palace.embed_all [--batch-size N] [--dry-run]

Also supports embedding individual drawers on add (called from provider.py).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Allow running as standalone script or module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from plugins.memory.palace.embedding_client import EmbeddingClient
from plugins.memory.palace.store import PalaceStore
from plugins.memory.palace.vector_store import VectorStore

logger = logging.getLogger(__name__)

_DEFAULT_DB = os.path.expanduser("~/.hermes/palace/drawers.db")
_BATCH_SIZE = 20  # API calls per batch (each call can handle up to 2048 texts)


def get_embedding_config() -> dict:
    """Read embedding config from hermes config.yaml.

    Tries plugins.palace.embedding first, then falls back to
    deriving api_key from vision config (same Zhipu API).
    """
    import yaml

    config_path = os.path.expanduser("~/.hermes/config.yaml")
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        emb = cfg.get("plugins", {}).get("palace", {}).get("embedding", {})
        # Derive API key from other configs if not explicitly set
        if not emb.get("api_key"):
            # Try: plugins.palace.embedding → auxiliary.vision → providers.zai
            for path in (
                ("auxiliary", "vision"),
                ("providers", "zai"),
            ):
                section = cfg
                for key in path:
                    section = section.get(key, {}) if isinstance(section, dict) else {}
                api_key = section.get("api_key") if isinstance(section, dict) else None
                if api_key:
                    emb.setdefault("api_key", api_key)
                    break
        return emb
    except Exception as e:
        logger.warning("Failed to read config: %s", e)
        return {}


def embed_all_drawers(
    db_path: str = _DEFAULT_DB,
    batch_size: int = _BATCH_SIZE,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Embed all drawers that don't have embeddings yet.

    Returns:
        (total, embedded, skipped) counts.
    """
    # Init store
    store = PalaceStore(db_path)

    # Get embedding config
    emb_config = get_embedding_config()
    if not emb_config.get("api_key"):
        logger.error("No embedding API key configured. Set plugins.memory_palace.embedding.api_key in config.yaml")
        return 0, 0, 0

    # Init embedding client and vector store
    client = EmbeddingClient(emb_config)
    if not client.available:
        logger.error("Embedding client not available (missing API key)")
        return 0, 0, 0

    dim = int(emb_config.get("dimension", 2048))
    vs = VectorStore(conn=store.conn, embedding_client=client, dimension=dim)

    # Find drawers without embeddings
    rows = store.conn.execute("""
        SELECT d.id, d.content
        FROM drawers d
        LEFT JOIN drawer_vector_map m ON m.drawer_id = d.id
        WHERE m.drawer_id IS NULL
          AND d.archived = 0
          AND d.content IS NOT NULL
          AND length(d.content) > 0
    """).fetchall()

    total = len(rows)
    if total == 0:
        print("All drawers already have embeddings.")
        return 0, 0, 0

    print(f"Found {total} drawers without embeddings (dim={dim})")
    if dry_run:
        for r in rows[:5]:
            print(f"  - {r['id']}: {r['content'][:60]}...")
        if total > 5:
            print(f"  ... and {total - 5} more")
        return total, 0, total

    # Process in batches
    embedded = 0
    skipped = 0

    def on_progress(done: int, total_items: int) -> None:
        print(f"  Progress: {done}/{total_items}", end="\r")

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        items = [{"id": r["id"], "content": r["content"]} for r in batch]

        try:
            count = vs.batch_embed(items, on_progress=on_progress)
            embedded += count
        except Exception as e:
            logger.error("Batch %d failed: %s", i // batch_size, e)
            skipped += len(batch)

        # Rate limit: small pause between batches
        if i + batch_size < total:
            time.sleep(0.5)

    print(f"\nDone: {embedded} embedded, {skipped} skipped out of {total}")
    return total, embedded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-embed Palace drawers")
    parser.add_argument("--db", default=_DEFAULT_DB, help="Path to drawers.db")
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE, help="Items per API batch")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be embedded")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    embed_all_drawers(db_path=args.db, batch_size=args.batch_size, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
