"""Vector store module using LanceDB."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import lancedb
import pyarrow as pa

if TYPE_CHECKING:
    from .reranker import Reranker


class StorageError(Exception):
    """Raised when vector store operations fail."""

    pass


@dataclass
class SearchResult:
    path: str
    chunk_index: int
    start_line: int
    end_line: int
    content: str
    description: str
    score: float
    content_score: float
    description_score: float


_CATEGORY_WEIGHTS = {
    "code": 1.0,
    "test": 0.8,
    "docs": 0.6,
}

_DOCS_EXTENSIONS = {".md", ".rst", ".txt"}


def _classify_file(path: str) -> str:
    """Classify a file path as 'code', 'test', or 'docs'."""
    parts = path.replace("\\", "/").split("/")
    filename = parts[-1] if parts else ""

    # Test detection: path contains tests/ or test/, or filename starts with
    # test_ or ends with _test.py
    for part in parts[:-1]:
        if part in ("tests", "test"):
            return "test"
    if filename.startswith("test_") or filename.endswith("_test.py"):
        return "test"

    # Docs detection: extension or docs/ prefix
    ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
    if ext in _DOCS_EXTENSIONS:
        return "docs"
    if parts and parts[0] == "docs":
        return "docs"

    return "code"


def _apply_category_bias(results: list[SearchResult]) -> list[SearchResult]:
    """Apply score multipliers by file category so the downstream sort respects them."""
    biased = []
    for r in results:
        weight = _CATEGORY_WEIGHTS.get(_classify_file(r.path), 1.0)
        biased.append(SearchResult(
            path=r.path,
            chunk_index=r.chunk_index,
            start_line=r.start_line,
            end_line=r.end_line,
            content=r.content,
            description=r.description,
            score=r.score * weight,
            content_score=r.content_score,
            description_score=r.description_score,
        ))
    return biased


def _deduplicate_by_file(results: list[SearchResult]) -> list[SearchResult]:
    """Keep only the highest-scoring chunk per file."""
    best: dict[str, SearchResult] = {}
    for r in results:
        if r.path not in best or r.score > best[r.path].score:
            best[r.path] = r
    return sorted(best.values(), key=lambda x: x.score, reverse=True)


def _rerank_results(
    results: list[SearchResult],
    query_text: str,
    reranker: Reranker,
) -> list[SearchResult]:
    """Re-score results using a cross-encoder and sort by new scores."""
    if not results:
        return results

    documents = [r.content for r in results]
    scores = reranker.rerank(query_text, documents)

    reranked = []
    for r, new_score in zip(results, scores):
        reranked.append(SearchResult(
            path=r.path,
            chunk_index=r.chunk_index,
            start_line=r.start_line,
            end_line=r.end_line,
            content=r.content,
            description=r.description,
            score=new_score,
            content_score=r.content_score,
            description_score=r.description_score,
        ))
    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked


def _rrf_merge(
    semantic_results: list[SearchResult],
    fts_results: list[SearchResult],
    semantic_weight: float = 1.0,
    fts_weight: float = 1.0,
    k: int = 60,
) -> list[SearchResult]:
    """Merge two ranked lists using weighted Reciprocal Rank Fusion.

    For each result, computes:
        score = w_sem / (k + rank_sem) + w_fts / (k + rank_fts)

    Results are keyed by (path, chunk_index) so the same chunk from both
    lists gets fused. Results appearing in only one list get a score
    contribution from that list only.
    """
    # Build lookup: (path, chunk_index) -> (SearchResult, semantic_rank, fts_rank)
    merged: dict[tuple[str, int], dict] = {}

    for rank, r in enumerate(semantic_results, start=1):
        key = (r.path, r.chunk_index)
        merged[key] = {
            "result": r,
            "sem_rank": rank,
            "fts_rank": None,
            "content_score": r.content_score,
            "description_score": r.description_score,
        }

    for rank, r in enumerate(fts_results, start=1):
        key = (r.path, r.chunk_index)
        if key in merged:
            merged[key]["fts_rank"] = rank
            # Keep description_score from semantic side (already set)
        else:
            merged[key] = {
                "result": r,
                "sem_rank": None,
                "fts_rank": rank,
                "content_score": r.content_score,
                "description_score": r.description_score,
            }

    out = []
    for key, entry in merged.items():
        score = 0.0
        if entry["sem_rank"] is not None:
            score += semantic_weight / (k + entry["sem_rank"])
        if entry["fts_rank"] is not None:
            score += fts_weight / (k + entry["fts_rank"])

        r = entry["result"]
        out.append(SearchResult(
            path=r.path,
            chunk_index=r.chunk_index,
            start_line=r.start_line,
            end_line=r.end_line,
            content=r.content,
            description=r.description,
            score=score,
            content_score=entry["content_score"],
            description_score=entry["description_score"],
        ))

    return out


class VectorStore:
    """LanceDB-backed vector store for file embeddings."""

    TABLE_NAME = "files"
    SEARCH_CACHE_TABLE = "searches"

    def __init__(self, db_path: Path, dimension: int, reranker: Reranker | None = None):
        self.db_path = db_path
        self.dimension = dimension
        self.reranker = reranker
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None
        self._search_cache: lancedb.table.Table | None = None

    def _get_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("path", pa.string()),
            pa.field("chunk_index", pa.int64()),
            pa.field("start_line", pa.int64()),
            pa.field("end_line", pa.int64()),
            pa.field("content", pa.string()),
            pa.field("fts_content", pa.string()),
            pa.field("description", pa.string()),
            pa.field("mtime", pa.float64()),
            pa.field("path_embedding", pa.list_(pa.float32(), self.dimension)),
            pa.field("content_embedding", pa.list_(pa.float32(), self.dimension)),
            pa.field("description_embedding", pa.list_(pa.float32(), self.dimension)),
        ])

    def _get_search_cache_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("query", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), self.dimension)),
            pa.field("hit_count", pa.int64()),
            pa.field("created_at", pa.float64()),
            pa.field("last_used_at", pa.float64()),
        ])

    async def connect(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.db_path))
        except Exception as e:
            raise StorageError(f"Failed to connect to database at {self.db_path}: {e}") from e

        try:
            self._table = self._db.open_table(self.TABLE_NAME)
        except (FileNotFoundError, ValueError):
            self._table = self._db.create_table(
                self.TABLE_NAME,
                schema=self._get_schema(),
            )

        try:
            self._search_cache = self._db.open_table(self.SEARCH_CACHE_TABLE)
        except (FileNotFoundError, ValueError):
            self._search_cache = self._db.create_table(
                self.SEARCH_CACHE_TABLE,
                schema=self._get_search_cache_schema(),
            )

        # Create FTS index on content for hybrid search
        self._ensure_fts_index()

    def _ensure_fts_index(self) -> None:
        """Create FTS index on content column if not exists."""
        if self._table is None:
            return

        try:
            self._table.create_fts_index(
                "fts_content",
                replace=True,
                with_position=False,
                language="English",
                stem=True,
                remove_stop_words=True,
            )
        except Exception:
            # FTS index creation may fail on empty table or if not supported
            pass

    async def upsert(
        self,
        path: str,
        chunk_index: int,
        start_line: int,
        end_line: int,
        content: str,
        path_embedding: list[float],
        content_embedding: list[float] | None,
        description: str = "",
        description_embedding: list[float] | None = None,
        mtime: float = 0.0,
        fts_content: str | None = None,
    ) -> None:
        """Insert or update a chunk's embeddings."""
        if self._table is None:
            raise RuntimeError("Not connected to database")

        # Insert new entry (caller handles deletion when needed)
        data = [{
            "path": path,
            "chunk_index": chunk_index,
            "start_line": start_line,
            "end_line": end_line,
            "content": content,
            "fts_content": fts_content or content,
            "description": description,
            "mtime": mtime,
            "path_embedding": path_embedding,
            "content_embedding": content_embedding or path_embedding,
            "description_embedding": description_embedding or path_embedding,
        }]
        self._table.add(data)

    async def upsert_batch(self, chunks: list[dict]) -> None:
        """Insert multiple chunks in a single LanceDB add() call.

        Each chunk dict should have: path, chunk_index, start_line, end_line,
        content, description, mtime, path_embedding, content_embedding,
        description_embedding.
        """
        if self._table is None:
            raise RuntimeError("Not connected to database")

        if not chunks:
            return

        data = []
        for c in chunks:
            data.append({
                "path": c["path"],
                "chunk_index": c["chunk_index"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "content": c["content"],
                "fts_content": c.get("fts_content") or c["content"],
                "description": c["description"],
                "mtime": c["mtime"],
                "path_embedding": c["path_embedding"],
                "content_embedding": c["content_embedding"] or c["path_embedding"],
                "description_embedding": c["description_embedding"] or c["path_embedding"],
            })

        self._table.add(data)

    async def delete(self, path: str) -> None:
        """Delete all chunks for a file from the store."""
        if self._table is None:
            raise RuntimeError("Not connected to database")

        try:
            self._table.delete(f'path = "{path}"')
        except Exception:
            pass  # Entry might not exist

    async def get_file_mtime(self, path: str) -> float | None:
        """Get the stored mtime for a file. Returns None if not indexed."""
        if self._table is None:
            raise RuntimeError("Not connected to database")

        try:
            results = self._table.search().where(
                f'path = "{path}"', prefilter=True
            ).limit(1).to_list()
        except Exception:
            return None

        if not results:
            return None

        return results[0].get("mtime")

    async def _search_semantic_raw(
        self,
        query_embedding: list[float],
        limit: int = 10,
        description_weight: float = 0.3,
    ) -> list[SearchResult]:
        """Raw semantic search: content + description embeddings, no post-processing."""
        if self._table is None:
            raise RuntimeError("Not connected to database")

        content_results = (
            self._table.search(query_embedding, vector_column_name="content_embedding")
            .metric("cosine")
            .limit(limit * 2)
            .to_list()
        )

        desc_results = (
            self._table.search(query_embedding, vector_column_name="description_embedding")
            .metric("cosine")
            .limit(limit * 2)
            .to_list()
        )

        desc_scores: dict[str, float] = {}
        for r in desc_results:
            desc_scores[r["path"]] = 1 / (1 + r["_distance"])

        content_weight = 1 - description_weight
        results = []
        for r in content_results:
            content_score = 1 / (1 + r["_distance"])
            description_score = desc_scores.get(r["path"], 0.0)
            combined = (content_weight * content_score) + (description_weight * description_score)

            results.append(SearchResult(
                path=r["path"],
                chunk_index=r["chunk_index"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                content=r["content"],
                description=r.get("description", ""),
                score=combined,
                content_score=content_score,
                description_score=description_score,
            ))

        return results

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        description_weight: float = 0.3,
        query_text: str = "",
    ) -> list[SearchResult]:
        """Search for similar chunks using content and description embeddings.

        Args:
            query_embedding: The query vector to search with.
            limit: Maximum number of results to return.
            description_weight: Weight for description similarity (0-1).
                Content weight is (1 - description_weight).
            query_text: Original query text for reranking (optional).
        """
        fetch = limit * 5 if self.reranker and query_text else limit
        results = await self._search_semantic_raw(query_embedding, fetch, description_weight)
        results = _apply_category_bias(results)
        results.sort(key=lambda x: x.score, reverse=True)
        results = _deduplicate_by_file(results)
        if self.reranker and query_text:
            results = _rerank_results(results, query_text, self.reranker)
        return results[:limit]

    async def _search_fts_raw(
        self,
        query_text: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Raw full-text search: BM25 results, no post-processing."""
        if self._table is None:
            raise RuntimeError("Not connected to database")

        try:
            results = (
                self._table.search(query_text, query_type="fts")
                .limit(limit)
                .to_list()
            )
        except Exception:
            return []

        return [
            SearchResult(
                path=r["path"],
                chunk_index=r["chunk_index"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                content=r["content"],
                description=r.get("description", ""),
                score=r.get("_score", 0.0),
                content_score=r.get("_score", 0.0),
                description_score=0.0,
            )
            for r in results
        ]

    async def search_fts(
        self,
        query_text: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Full-text search only (BM25).

        Args:
            query_text: The text query to search for.
            limit: Maximum number of results to return.
        """
        fetch = limit * 5 if self.reranker else limit * 3
        fts_results = await self._search_fts_raw(query_text, fetch)
        results = _deduplicate_by_file(_apply_category_bias(fts_results))
        if self.reranker:
            results = _rerank_results(results, query_text, self.reranker)
        return results[:limit]

    async def search_hybrid(
        self,
        query_embedding: list[float],
        query_text: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Hybrid search: vector + FTS with custom weighted RRF.

        Runs semantic and FTS searches separately, then merges their ranked
        lists using Reciprocal Rank Fusion with tunable weights.

        Args:
            query_embedding: The query vector to search with.
            query_text: The text query for FTS.
            limit: Maximum number of results to return.
        """
        fetch = limit * 5 if self.reranker else limit * 3
        sem_results = await self._search_semantic_raw(query_embedding, limit=fetch)
        fts_results = await self._search_fts_raw(query_text, limit=fetch)

        merged = _rrf_merge(sem_results, fts_results, semantic_weight=1.2, fts_weight=0.5)
        merged = _apply_category_bias(merged)
        merged.sort(key=lambda x: x.score, reverse=True)
        merged = _deduplicate_by_file(merged)
        if self.reranker:
            merged = _rerank_results(merged, query_text, self.reranker)
        return merged[:limit]

    async def list_all(self) -> dict[str, int]:
        """List all indexed file paths with their chunk counts."""
        if self._table is None:
            raise RuntimeError("Not connected to database")

        table = self._table.to_arrow()
        paths = table.column("path").to_pylist()
        # Count chunks per file
        counts: dict[str, int] = {}
        for path in paths:
            counts[path] = counts.get(path, 0) + 1
        return counts

    async def get_all_mtimes(self) -> dict[str, float]:
        """Get all stored (path, mtime) pairs in a single query.

        Returns a dict mapping path -> mtime. Much faster than individual lookups.
        """
        if self._table is None:
            raise RuntimeError("Not connected to database")

        table = self._table.to_arrow()
        paths = table.column("path").to_pylist()
        mtimes = table.column("mtime").to_pylist()

        # Deduplicate - files may have multiple chunks, all with same mtime
        result: dict[str, float] = {}
        for path, mtime in zip(paths, mtimes):
            if path not in result:
                result[path] = mtime
        return result

    async def get_cached_query(self, query: str) -> list[float] | None:
        """Get cached embedding for a query. Returns None on cache miss."""
        if self._search_cache is None:
            return None

        try:
            results = self._search_cache.search().where(
                f'query = "{query}"', prefilter=True
            ).limit(1).to_list()
        except Exception:
            return None

        if not results:
            return None

        # Update hit count and last_used_at
        row = results[0]
        now = time.time()
        self._search_cache.delete(f'query = "{query}"')
        self._search_cache.add([{
            "query": query,
            "embedding": row["embedding"],
            "hit_count": row["hit_count"] + 1,
            "created_at": row["created_at"],
            "last_used_at": now,
        }])

        return row["embedding"]

    async def cache_query(self, query: str, embedding: list[float]) -> None:
        """Cache a query embedding."""
        if self._search_cache is None:
            return

        now = time.time()
        self._search_cache.add([{
            "query": query,
            "embedding": embedding,
            "hit_count": 0,
            "created_at": now,
            "last_used_at": now,
        }])

    async def get_cache_stats(self) -> dict:
        """Get search cache statistics."""
        if self._search_cache is None:
            return {"unique_queries": 0, "total_hits": 0}

        try:
            table = self._search_cache.to_arrow()
            unique_queries = table.num_rows
            total_hits = sum(table.column("hit_count").to_pylist())
            return {"unique_queries": unique_queries, "total_hits": total_hits}
        except Exception:
            return {"unique_queries": 0, "total_hits": 0}

    def get_db_size(self) -> int:
        """Get total size of database files in bytes."""
        if not self.db_path.exists():
            return 0
        total = 0
        for f in self.db_path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    async def close(self) -> None:
        self._table = None
        self._search_cache = None
        self._db = None
