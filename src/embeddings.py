"""Embedding generation module using local sentence-transformers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING


@dataclass
class ChunkEmbedding:
    """Result of embedding a single chunk."""

    path: str
    chunk_index: int
    start_line: int
    end_line: int
    content: str
    description: str
    path_embedding: list[float]
    content_embedding: list[float] | None
    description_embedding: list[float] | None
    mtime: float

if TYPE_CHECKING:
    from src.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# Models that require a prefix on search queries (not on document content).
# Key: model name suffix (matched against the end of model_name).
# Value: prefix string to prepend to queries.
QUERY_PREFIX_BY_MODEL: dict[str, str] = {
    "CodeRankEmbed": "Represent this query for searching relevant code: ",
}

# Models that need trust_remote_code=True when loading.
TRUST_REMOTE_CODE_MODELS: set[str] = {
    "CodeRankEmbed",
}


def _query_prefix_for(model_name: str) -> str:
    """Return the query prefix for a model, or empty string if none needed."""
    for suffix, prefix in QUERY_PREFIX_BY_MODEL.items():
        if model_name.endswith(suffix):
            return prefix
    return ""


def _needs_trust_remote_code(model_name: str) -> bool:
    """Check if a model needs trust_remote_code=True."""
    return any(model_name.endswith(s) for s in TRUST_REMOTE_CODE_MODELS)


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        ...

    @property
    def query_prefix(self) -> str:
        """Prefix to prepend to search queries (empty for most models)."""
        return ""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class LocalEmbedding(EmbeddingProvider):
    """Local embedding using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._query_prefix = _query_prefix_for(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def query_prefix(self) -> str:
        return self._query_prefix

    def _load_model(self):
        if self._model is None:
            logger.debug(f"Loading SentenceTransformer model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            kwargs = {}
            if _needs_trust_remote_code(self.model_name):
                kwargs["trust_remote_code"] = True
            self._model = SentenceTransformer(self.model_name, **kwargs)
            logger.debug(f"Model loaded: {self.model_name}")
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def dimension(self) -> int:
        model = self._load_model()
        return model.get_sentence_embedding_dimension()


class EmbeddingService:
    """High-level service for generating embeddings from files."""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    async def embed_file(self, path: str, content: str, description: str = "") -> dict:
        """Generate embeddings for a file or chunk's path, content, and description."""
        texts = [path]
        if content:
            texts.append(content)
        if description:
            texts.append(description)

        embeddings = await self.provider.embed(texts)

        result = {
            "path": path,
            "path_embedding": embeddings[0],
            "content_embedding": None,
            "content": content,
            "description": description,
            "description_embedding": None,
        }

        idx = 1
        if content:
            result["content_embedding"] = embeddings[idx]
            idx += 1
        if description:
            result["description_embedding"] = embeddings[idx]

        return result

    async def embed_chunks_batch(
        self,
        chunks: list[tuple[str, int, int, int, str, str, float]],
    ) -> list[ChunkEmbedding]:
        """Embed multiple chunks in a single batch call.

        Args:
            chunks: List of tuples (path, chunk_index, start_line, end_line,
                    content, description, mtime).

        Returns:
            List of ChunkEmbedding results with embeddings filled in.

        Groups all texts (paths, contents, descriptions) into one embed() call,
        then distributes results back to chunks.
        """
        if not chunks:
            return []

        # Build text lists, tracking positions
        texts: list[str] = []
        positions: list[tuple[int, str]] = []  # (chunk_idx, field_type)

        for i, (path, _idx, _start, _end, content, desc, _mtime) in enumerate(chunks):
            positions.append((i, "path"))
            texts.append(path)

            if content:
                positions.append((i, "content"))
                texts.append(content)

            if desc:
                positions.append((i, "description"))
                texts.append(desc)

        # Single batch embed call
        embeddings = await self.provider.embed(texts)

        # Distribute embeddings back to chunks
        embed_map: dict[int, dict[str, list[float]]] = {i: {} for i in range(len(chunks))}

        for (chunk_idx, field), embedding in zip(positions, embeddings):
            embed_map[chunk_idx][field] = embedding

        results: list[ChunkEmbedding] = []
        for i, (path, idx, start, end, content, desc, mtime) in enumerate(chunks):
            results.append(ChunkEmbedding(
                path=path,
                chunk_index=idx,
                start_line=start,
                end_line=end,
                content=content,
                description=desc,
                path_embedding=embed_map[i]["path"],
                content_embedding=embed_map[i].get("content"),
                description_embedding=embed_map[i].get("description"),
                mtime=mtime,
            ))

        return results

    async def embed_query(
        self, query: str, cache: VectorStore | None = None
    ) -> tuple[list[float], bool]:
        """Generate embedding for a search query.

        Applies the model's query prefix if one is configured (e.g. CodeRankEmbed
        requires a specific instruction prefix on queries but not on documents).

        Returns:
            Tuple of (embedding, cache_hit).
        """
        if cache:
            cached = await cache.get_cached_query(query)
            if cached is not None:
                return cached, True

        prefixed = self.provider.query_prefix + query
        embeddings = await self.provider.embed([prefixed])
        embedding = embeddings[0]

        if cache:
            await cache.cache_query(query, embedding)

        return embedding, False
