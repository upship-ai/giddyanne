"""MCP server entrypoint for Giddyanne."""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from src.embeddings import EmbeddingService
from src.engine import (
    FileIndexer,
    IndexingProgress,
    StatsTracker,
    create_embedding_provider,
)
from src.mcp_server import run_mcp_server
from src.project_config import ConfigError, FileFilter, ProjectConfig
from src.vectorstore import VectorStore
from src.watcher import FileWatcher

logging.basicConfig(level=logging.WARNING)  # Quiet for MCP stdio
logger = logging.getLogger(__name__)


async def main():
    # Get root path from environment or current directory
    # (Go CLI passes GIDDY_WATCH_PATH when spawning MCP server)
    root_path_str = os.environ.get("GIDDY_WATCH_PATH")
    root_path = Path(root_path_str).resolve() if root_path_str else Path.cwd().resolve()
    config_path = root_path / ".giddyanne.yaml"

    # Load project config
    if config_path.exists():
        project_config = ProjectConfig.load(config_path)
    else:
        project_config = ProjectConfig.default(root_path)

    # Initialize embedding provider
    provider = create_embedding_provider(project_config)
    embedding_service = EmbeddingService(provider)

    # Test embedding provider before indexing
    try:
        await provider.embed("test")
    except Exception as e:
        print(f"Warning: Embedding provider failed: {e}", file=sys.stderr)

    # Initialize stats tracker and progress
    # db_path includes model name for isolation between different embedding models
    safe_model = provider.model_name.replace("/", "-")
    base_path = Path(project_config.settings.db_path)
    db_path = root_path / base_path.parent / safe_model / base_path.name
    log_path = db_path.parent / "mcp.log"
    stats = StatsTracker(log_path, root_path)
    progress = IndexingProgress()

    # Initialize vector store
    vector_store = VectorStore(db_path, provider.dimension())
    await vector_store.connect()

    # Create file filter (shared between indexer and watcher)
    file_filter = FileFilter(root_path, project_config)

    # Index files
    indexer = FileIndexer(
        embedding_service,
        vector_store,
        file_filter,
        stats,
        progress,
    )

    # Reconcile index first (remove stale files)
    removed = await indexer.reconcile_index()
    if removed:
        logger.info(f"Removed {removed} stale files from index")

    await indexer.full_index()

    # Start file watcher for real-time updates
    watcher = FileWatcher(file_filter, indexer.handle_event)
    await watcher.start()

    # Run MCP server
    await run_mcp_server(embedding_service, vector_store, stats)


if __name__ == "__main__":
    # Become process group leader so we can kill all child processes on exit
    # (sentence-transformers/PyTorch spawn workers that need to be cleaned up)
    try:
        os.setpgrp()
    except OSError:
        pass  # Already a process group leader

    def handle_signal(signum, frame):
        # Kill entire process group to clean up embedding model workers
        os.killpg(os.getpid(), signal.SIGKILL)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Startup failed: {e}", file=sys.stderr)
        sys.exit(1)
