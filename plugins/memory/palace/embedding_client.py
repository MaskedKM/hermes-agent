"""Embedding client — OpenAI-compatible interface.

Uses Zhipu (智谱) embedding-3 model (2048d, free tier).
"""

from __future__ import annotations

import hashlib
import logging
import struct
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum texts per batch for Zhipu embedding API
_MAX_BATCH_SIZE = 2048


def serialize_f32(vector: List[float]) -> bytes:
    """Serialize a float list to little-endian f32 bytes (for sqlite-vec)."""
    return struct.pack(f"{len(vector)}f", *vector)


def deserialize_f32(data: bytes) -> List[float]:
    """Deserialize f32 bytes back to float list."""
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


def content_hash(text: str, max_chars: int = 500) -> str:
    """SHA256 hash of text[:max_chars] for embedding cache key."""
    return hashlib.sha256(text[:max_chars].encode("utf-8")).hexdigest()


class EmbeddingClient:
    """Embedding client using OpenAI-compatible interface.

    Default: Zhipu embedding-3 (2048d, free).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.base_url = config.get("base_url", "https://open.bigmodel.cn/api/paas/v4")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "embedding-3")
        self.dimension = int(config.get("dimension", 2048))
        self.timeout = int(config.get("timeout", 30))
        self._client = None
        self._last_error: Optional[str] = None

    def _get_client(self):
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding with auto-chunking.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            RuntimeError: If API call fails after retries.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        # Process in chunks to respect API batch limits
        for i in range(0, len(texts), _MAX_BATCH_SIZE):
            chunk = texts[i : i + _MAX_BATCH_SIZE]
            embeddings = self._call_api(chunk)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_text(self, text: str) -> List[float]:
        """Single text embedding."""
        results = self.embed_texts([text])
        return results[0] if results else []

    def _call_api(self, texts: List[str], max_retries: int = 2) -> List[List[float]]:
        """Call the embedding API with retry logic."""
        client = self._get_client()

        for attempt in range(max_retries + 1):
            try:
                response = client.embeddings.create(
                    input=texts,
                    model=self.model,
                )
                # Sort by index to ensure correct ordering
                data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in data]
            except Exception as e:
                self._last_error = str(e)
                logger.warning(
                    "Embedding API call failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))

        raise RuntimeError(f"Embedding API failed after {max_retries + 1} attempts: {self._last_error}")

    @property
    def available(self) -> bool:
        """Check if client is configured (has API key)."""
        return bool(self.api_key)

    def test_connection(self) -> bool:
        """Test if the embedding API is reachable."""
        try:
            result = self.embed_text("test")
            return len(result) == self.dimension
        except Exception as e:
            logger.warning("Embedding connection test failed: %s", e)
            return False
