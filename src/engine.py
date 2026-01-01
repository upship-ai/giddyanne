"""Shared indexing engine for HTTP and MCP servers."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.chunker import chunk_content
from src.embeddings import EmbeddingService, LocalEmbedding
from src.languages import detect_language
from src.project_config import FileFilter, ProjectConfig
from src.vectorstore import VectorStore
from src.watcher import EventType, FileEvent

logger = logging.getLogger(__name__)

# Tuning constants for parallel indexing
FILE_READ_CONCURRENCY = 16  # Max concurrent file reads
EMBED_BATCH_SIZE = 32  # Chunks per embedding batch

# Module-level executor for file I/O
_file_executor = ThreadPoolExecutor(max_workers=FILE_READ_CONCURRENCY)


async def read_file_async(path: Path) -> tuple[Path, str, float]:
    """Read file content in thread pool to avoid blocking event loop.

    Returns (path, content, mtime).
    """
    loop = asyncio.get_running_loop()

    def _read():
        mtime = path.stat().st_mtime
        content = path.read_text(errors="ignore")
        return (path, content, mtime)

    return await loop.run_in_executor(_file_executor, _read)


class IndexingProgress:
    """Tracks indexing state and progress."""

    def __init__(self):
        self.state: str = "starting"  # starting, indexing, ready, error
        self.total: int = 0
        self.indexed: int = 0
        self.error: str | None = None

    @property
    def percent(self) -> int:
        if self.total == 0:
            return 0
        return int(self.indexed / self.total * 100)

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "total": self.total,
            "indexed": self.indexed,
            "percent": self.percent,
            "error": self.error,
        }


class StatsTracker:
    """Tracks operation stats and appends per-operation log lines."""

    def __init__(self, log_path: Path, root_path: Path | None = None):
        self.log_path = log_path
        self.root_path = root_path
        self._embed_count = 0
        self._search_count = 0
        self._cache_hits = 0
        self._delete_count = 0
        self._search_durations: list[float] = []
        self._max_durations = 100  # Keep last N for rolling average
        self._start_time = time.perf_counter()
        self._startup_duration_ms: float | None = None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log("START", "server initialized")

    def _timestamp(self) -> str:
        return time.strftime("%H:%M:%S")

    def _log(self, op: str, detail: str) -> None:
        line = f"{self._timestamp()} {op:<6} {detail}\n"
        with self.log_path.open("a") as f:
            f.write(line)

    def _relative_path(self, file_path: str) -> str:
        """Convert absolute path to relative path from root."""
        if self.root_path:
            try:
                return str(Path(file_path).relative_to(self.root_path))
            except ValueError:
                pass
        return file_path

    def record_embed(
        self,
        duration: float,
        file_path: str,
        progress: IndexingProgress | None = None,
    ) -> None:
        self._embed_count += 1
        path = self._relative_path(file_path)
        if progress and progress.total > 0:
            pct = progress.percent
            self._log(
                "EMBED",
                f"{duration:.2f}s  {path}  [{progress.indexed}/{progress.total} {pct}%]",
            )
        else:
            self._log("EMBED", f"{duration:.2f}s  {path}  [{self._embed_count}]")

    def record_search(
        self, duration: float, cache_hit: bool = False, query: str = ""
    ) -> None:
        self._search_count += 1
        if cache_hit:
            self._cache_hits += 1
        self._search_durations.append(duration)
        if len(self._search_durations) > self._max_durations:
            self._search_durations.pop(0)
        cache_str = "hit" if cache_hit else "miss"
        q = query[:30] + "..." if len(query) > 30 else query
        self._log(
            "SEARCH",
            f'{duration:.2f}s  "{q}"  [cache:{cache_str} {self._cache_hits}/{self._search_count}]',
        )

    def avg_query_latency_ms(self) -> float | None:
        """Get average query latency in milliseconds. Returns None if no searches yet."""
        if not self._search_durations:
            return None
        return (sum(self._search_durations) / len(self._search_durations)) * 1000

    def record_delete(self, file_path: str) -> None:
        self._delete_count += 1
        path = self._relative_path(file_path)
        self._log("DELETE", f"{path}  [{self._delete_count}]")

    def record_startup_complete(self) -> None:
        """Record that initial indexing is complete."""
        self._startup_duration_ms = (time.perf_counter() - self._start_time) * 1000
        duration_secs = self._startup_duration_ms / 1000
        self._log("READY", f"[{duration_secs:.2f}s]")

    def startup_duration_ms(self) -> float | None:
        """Get startup duration in milliseconds. None if not yet complete."""
        return self._startup_duration_ms


class FileIndexer:
    """Handles indexing files when changes are detected."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        file_filter: FileFilter,
        stats: StatsTracker | None = None,
        progress: IndexingProgress | None = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.file_filter = file_filter
        self.stats = stats
        self.progress = progress or IndexingProgress()

    @property
    def root_path(self) -> Path:
        return self.file_filter.root_path

    @property
    def project_config(self) -> ProjectConfig:
        return self.file_filter.config

    async def _should_reindex(self, path: Path) -> tuple[bool, float]:
        """Check if a file needs re-indexing based on mtime.

        Returns (should_reindex, current_mtime).
        """
        try:
            current_mtime = path.stat().st_mtime
        except OSError:
            return False, 0.0

        stored_mtime = await self.vector_store.get_file_mtime(str(path))
        if stored_mtime is not None and stored_mtime == current_mtime:
            return False, current_mtime

        return True, current_mtime

    async def index_file(self, path: Path, mtime: float | None = None) -> None:
        """Index a single file by chunking and embedding each chunk."""
        if not self.file_filter.should_include(path):
            return

        start = time.perf_counter()
        if self.stats:
            self.stats._log("INDEX", self.stats._relative_path(str(path)))
        try:
            # Capture mtime if not provided
            if mtime is None:
                mtime = path.stat().st_mtime

            content = path.read_text(errors="ignore")
            description = self.file_filter.get_description(path)

            # Delete existing chunks for this file
            await self.vector_store.delete(str(path))

            # Chunk the content (language-aware if recognized)
            settings = self.project_config.settings
            language = detect_language(str(path))
            chunks = chunk_content(
                content,
                language=language,
                min_lines=settings.min_chunk_lines,
                max_lines=settings.max_chunk_lines,
                overlap=settings.overlap_lines,
            )

            # If no chunks (empty file), create a single chunk
            if not chunks:
                chunks = [type("Chunk", (), {
                    "start_line": 1,
                    "end_line": 1,
                    "content": content,
                })()]

            # Embed and store each chunk
            for idx, chunk in enumerate(chunks):
                logger.debug(
                    f"Embedding chunk {idx + 1}/{len(chunks)} for {path.name} "
                    f"(lines {chunk.start_line}-{chunk.end_line})"
                )
                result = await self.embedding_service.embed_file(
                    str(path), chunk.content, description
                )
                await self.vector_store.upsert(
                    path=result["path"],
                    chunk_index=idx,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=result["content"],
                    path_embedding=result["path_embedding"],
                    content_embedding=result["content_embedding"],
                    description=result["description"],
                    description_embedding=result["description_embedding"],
                    mtime=mtime,
                )

            if self.stats:
                self.stats.record_embed(
                    time.perf_counter() - start, str(path), self.progress
                )

            logger.info(f"Indexed: {path} ({len(chunks)} chunks)")
        except Exception as e:
            logger.error(f"Failed to index {path}: {e}")

    async def delete_file(self, path: Path) -> None:
        """Remove a file from the index."""
        await self.vector_store.delete(str(path))
        if self.stats:
            self.stats.record_delete(str(path))
        logger.info(f"Removed from index: {path}")

    async def handle_event(self, event: FileEvent) -> None:
        """Handle a file system event."""
        if event.is_directory:
            return

        if event.event_type == EventType.DELETED:
            await self.delete_file(event.path)
        elif event.event_type == EventType.MOVED:
            await self.delete_file(event.path)
            if event.dest_path:
                await self.index_file(event.dest_path)
        else:  # CREATED or MODIFIED
            # Check if file actually needs reindexing (mtime changed)
            should_reindex, mtime = await self._should_reindex(event.path)
            if should_reindex:
                await self.index_file(event.path, mtime=mtime)

    async def reconcile_index(self) -> int:
        """Remove stale files from index. Returns count of removed files.

        A file is stale if:
        - It no longer exists on disk, OR
        - It no longer matches the configured paths/patterns
        """
        start = time.perf_counter()
        indexed_files = await self.vector_store.list_all()
        list_time = time.perf_counter() - start
        logger.info(
            f"Reconcile: Listed {len(indexed_files)} indexed files in {list_time:.2f}s"
        )

        check_start = time.perf_counter()
        removed = 0
        already_processed: set[str] = set()

        for file_path_str in indexed_files:
            # Skip if already processed (defensive against duplicates in list)
            if file_path_str in already_processed:
                continue
            already_processed.add(file_path_str)

            file_path = Path(file_path_str)
            should_remove = False
            reason = ""

            # Check 1: Does file exist?
            if not file_path.exists():
                should_remove = True
                reason = "deleted"
            # Check 2: Does it match config? (use matches_path, not should_include)
            elif not self.file_filter.matches_path(file_path):
                should_remove = True
                reason = "excluded"

            if should_remove:
                await self.vector_store.delete(file_path_str)
                removed += 1
                logger.info(f"Removed stale ({reason}): {file_path}")

        check_time = time.perf_counter() - check_start
        total_time = time.perf_counter() - start
        logger.info(
            f"Reconcile complete in {total_time:.2f}s "
            f"(list:{list_time:.2f}s, check:{check_time:.2f}s, removed:{removed})"
        )
        return removed

    async def full_index(self) -> None:
        """Perform a full index of all configured paths."""
        logger.info(f"Starting full index of {self.root_path}")

        # Phase 1: Pre-scan to collect candidate files
        phase1_start = time.perf_counter()
        files_to_check: list[Path] = []
        for path_config in self.project_config.paths:
            target = self.root_path / path_config.path
            if target.is_file():
                if self.file_filter.should_include(target):
                    files_to_check.append(target)
            elif target.is_dir():
                for file_path in target.rglob("*"):
                    if self.file_filter.should_include(file_path):
                        files_to_check.append(file_path)

        phase1_time = time.perf_counter() - phase1_start
        logger.info(
            f"Phase 1: Found {len(files_to_check)} candidate files in {phase1_time:.2f}s"
        )

        # Phase 2: Filter to only files that need re-indexing
        phase2_start = time.perf_counter()

        # Load all stored mtimes in one bulk query (the slow part was N individual queries)
        stored_mtimes = await self.vector_store.get_all_mtimes()

        files_to_index: list[Path] = []
        for file_path in files_to_check:
            try:
                current_mtime = file_path.stat().st_mtime
            except OSError:
                continue  # File disappeared

            stored_mtime = stored_mtimes.get(str(file_path))
            if stored_mtime is None or stored_mtime != current_mtime:
                files_to_index.append(file_path)

        phase2_time = time.perf_counter() - phase2_start
        skipped = len(files_to_check) - len(files_to_index)
        self.progress.total = len(files_to_index)
        self.progress.state = "indexing"
        logger.info(
            f"Phase 2: Checked mtimes in {phase2_time:.2f}s - "
            f"{self.progress.total} to index, {skipped} unchanged"
        )

        # Phase 3: Parallel read + batch embed
        phase3_start = time.perf_counter()

        # Semaphore to limit concurrent file reads
        read_sem = asyncio.Semaphore(FILE_READ_CONCURRENCY)
        settings = self.project_config.settings

        async def read_and_chunk(path: Path) -> list[tuple]:
            """Read file and return list of chunk tuples."""
            async with read_sem:
                try:
                    path, content, mtime = await read_file_async(path)
                except Exception as e:
                    logger.error(f"Failed to read {path}: {e}")
                    return []

            description = self.file_filter.get_description(path)

            # Delete existing chunks
            await self.vector_store.delete(str(path))

            # Chunk content
            language = detect_language(str(path))
            chunks = chunk_content(
                content,
                language=language,
                min_lines=settings.min_chunk_lines,
                max_lines=settings.max_chunk_lines,
                overlap=settings.overlap_lines,
            )

            if not chunks:
                chunks = [type("Chunk", (), {
                    "start_line": 1, "end_line": 1, "content": content
                })()]

            return [
                (str(path), idx, c.start_line, c.end_line, c.content, description, mtime)
                for idx, c in enumerate(chunks)
            ]

        # Read all files in parallel
        all_chunk_lists = await asyncio.gather(
            *[read_and_chunk(p) for p in files_to_index],
            return_exceptions=True
        )

        # Flatten chunks, skip errors
        all_chunks: list[tuple] = []
        files_processed = 0
        for result in all_chunk_lists:
            if isinstance(result, Exception):
                logger.error(f"Read error: {result}")
                continue
            if result:
                all_chunks.extend(result)
                files_processed += 1
                self.progress.indexed = files_processed

        logger.info(
            f"Phase 3a: Read {files_processed} files, {len(all_chunks)} chunks "
            f"in {time.perf_counter() - phase3_start:.2f}s"
        )

        # Batch embed and upsert
        embed_start = time.perf_counter()
        total_batches = (len(all_chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
        for batch_num, batch_start in enumerate(
            range(0, len(all_chunks), EMBED_BATCH_SIZE), 1
        ):
            batch = all_chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
            batch_time = time.perf_counter()
            chunk_embeddings = await self.embedding_service.embed_chunks_batch(batch)

            # Batch upsert
            upsert_data = [
                {
                    "path": ce.path,
                    "chunk_index": ce.chunk_index,
                    "start_line": ce.start_line,
                    "end_line": ce.end_line,
                    "content": ce.content,
                    "description": ce.description,
                    "mtime": ce.mtime,
                    "path_embedding": ce.path_embedding,
                    "content_embedding": ce.content_embedding,
                    "description_embedding": ce.description_embedding,
                }
                for ce in chunk_embeddings
            ]
            await self.vector_store.upsert_batch(upsert_data)

            # Log batch progress
            if self.stats:
                batch_duration = time.perf_counter() - batch_time
                self.stats._log(
                    "BATCH",
                    f"{batch_duration:.2f}s  {len(batch)} chunks  "
                    f"[{batch_num}/{total_batches}]",
                )

        embed_time = time.perf_counter() - embed_start
        phase3_time = time.perf_counter() - phase3_start
        total_time = phase1_time + phase2_time + phase3_time
        self.progress.state = "ready"
        logger.info(
            f"Phase 3b: Embedded {len(all_chunks)} chunks in {embed_time:.2f}s"
        )
        logger.info(
            f"Full index complete in {total_time:.2f}s "
            f"(scan:{phase1_time:.2f}s, mtime:{phase2_time:.2f}s, embed:{phase3_time:.2f}s)"
        )


def create_embedding_provider(project_config: ProjectConfig):
    """Create local embedding provider using sentence-transformers."""
    return LocalEmbedding(model_name=project_config.settings.local_model)
