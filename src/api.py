"""FastAPI HTTP server for querying file embeddings."""

import time
from importlib.metadata import version
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .embeddings import EmbeddingService
from .errors import SearchError
from .project_config import ProjectConfig
from .vectorstore import StorageError, VectorStore


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    search_type: Literal["hybrid", "semantic", "full-text"] = Field(
        default="hybrid",
        description="Search mode: hybrid (default), semantic, or full-text",
    )


class SearchResult(BaseModel):
    path: str
    start_line: int
    end_line: int
    content: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str


class StatsResponse(BaseModel):
    indexed_files: int
    total_chunks: int
    index_size_bytes: int
    avg_query_latency_ms: float | None
    startup_duration_ms: float | None
    files: dict[str, int]  # path -> chunk count


class StatusResponse(BaseModel):
    state: str  # starting, indexing, ready, error
    total: int
    indexed: int
    percent: int
    error: str | None = None


def create_app(
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    progress: object | None = None,
    stats: object | None = None,
    project_config: ProjectConfig | None = None,
) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Giddyanne",
        description="Semantic file search via embeddings",
        version=version("giddyanne"),
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/status", response_model=StatusResponse)
    async def status():
        """Get current server status and indexing progress."""
        if progress is None:
            return StatusResponse(state="ready", total=0, indexed=0, percent=100)
        return StatusResponse(**progress.to_dict())

    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest):
        """Search indexed files by semantic similarity."""
        # Check if server is still indexing
        if progress is not None and progress.state == "indexing":
            raise HTTPException(
                status_code=503,
                detail=f"Server is indexing ({progress.percent}% complete)",
            )

        start = time.perf_counter()
        cache_hit = False
        try:
            # Route to appropriate search method based on search_type
            if request.search_type == "full-text":
                # FTS doesn't need embedding
                results = await vector_store.search_fts(
                    request.query,
                    limit=request.limit,
                )
            else:
                # Both semantic and hybrid need embedding
                query_embedding, cache_hit = await embedding_service.embed_query(request.query)
                if request.search_type == "hybrid":
                    results = await vector_store.search_hybrid(
                        query_embedding,
                        request.query,
                        limit=request.limit,
                    )
                else:  # semantic
                    results = await vector_store.search(
                        query_embedding,
                        limit=request.limit,
                    )
            return SearchResponse(
                results=[
                    SearchResult(
                        path=r.path,
                        start_line=r.start_line,
                        end_line=r.end_line,
                        content=r.content,
                        score=r.score,
                    )
                    for r in results
                ],
                query=request.query,
            )
        except StorageError as e:
            raise HTTPException(status_code=503, detail=f"Storage unavailable: {e}")
        except SearchError as e:
            raise HTTPException(status_code=500, detail=f"Search failed: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if stats is not None:
                stats.record_search(
                    time.perf_counter() - start,
                    cache_hit=cache_hit,
                    query=request.query,
                )

    @app.get("/stats", response_model=StatsResponse)
    async def get_stats():
        """Get statistics about indexed files."""
        files = await vector_store.list_all()
        return StatsResponse(
            indexed_files=len(files),
            total_chunks=sum(files.values()),
            index_size_bytes=vector_store.get_db_size(),
            avg_query_latency_ms=stats.avg_query_latency_ms() if stats else None,
            startup_duration_ms=stats.startup_duration_ms() if stats else None,
            files=files,
        )

    @app.get("/sitemap")
    async def sitemap(verbose: bool = False):
        """List all indexed file paths."""
        files = await vector_store.list_all()
        paths = sorted(files.keys())

        config_paths = None
        if project_config is not None:
            config_paths = [
                {"path": pc.path, "description": pc.description}
                for pc in project_config.paths
            ]

        if not verbose:
            result = {"files": paths, "count": len(paths)}
            if config_paths is not None:
                result["paths"] = config_paths
            return result

        mtimes = await vector_store.get_all_mtimes()
        result = {
            "files": [
                {"path": p, "chunks": files[p], "mtime": mtimes.get(p, 0.0)}
                for p in paths
            ],
            "count": len(paths),
        }
        if config_paths is not None:
            result["paths"] = config_paths
        return result

    @app.post("/reindex")
    async def reindex():
        """Trigger a full reindex (placeholder)."""
        # This will be implemented in http_main.py with access to the indexer
        return {"status": "reindex triggered"}

    return app
