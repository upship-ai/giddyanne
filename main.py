"""Main entrypoint for Giddyanne HTTP server."""

import argparse
import asyncio
import atexit
import logging
import os
import signal
import sys
from pathlib import Path

import uvicorn

from src.api import create_app
from src.embeddings import EmbeddingService
from src.engine import (
    FileIndexer,
    IndexingProgress,
    StatsTracker,
    create_embedding_provider,
)
from src.project_config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    ConfigError,
    FileFilter,
    ProjectConfig,
)
from src.vectorstore import VectorStore
from src.watcher import FileWatcher

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Configure logging level and optional file output."""
    level = logging.DEBUG if verbose else logging.WARNING
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    # Basic config for console/stderr
    logging.basicConfig(level=level, format=fmt)

    # Add file handler if log_file specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(file_handler)

# PID file directory (for scanning)
PID_DIR = ".giddyanne"


def get_pid_file_path(root_path: Path, host: str, port: int) -> Path:
    """Get the PID file path. Omit host/port from name when using defaults."""
    host_part = "" if host == DEFAULT_HOST else host
    port_part = "" if port == DEFAULT_PORT else str(port)
    return root_path / f".giddyanne/{host_part}-{port_part}.pid"


def get_log_file_path(root_path: Path, host: str, port: int) -> Path:
    """Get the log file path. Omit host/port from name when using defaults."""
    host_part = "" if host == DEFAULT_HOST else host
    port_part = "" if port == DEFAULT_PORT else str(port)
    return root_path / f".giddyanne/{host_part}-{port_part}.log"


def get_db_path(root_path: Path, base_db_path: str, model_name: str) -> Path:
    """Get database path with model name included.

    Transforms e.g. '.giddyanne/vectors.lance' + 'all-MiniLM-L6-v2'
    into '.giddyanne/all-MiniLM-L6-v2/vectors.lance'
    """
    # Sanitize model name for filesystem (replace / with -)
    safe_model = model_name.replace("/", "-")
    base_path = Path(base_db_path)
    return root_path / base_path.parent / safe_model / base_path.name


def find_running_server(root_path: Path) -> tuple[str, int, int] | None:
    """Find a running server by scanning pid files. Returns (host, port, pid) or None."""
    pid_dir = root_path / PID_DIR
    if not pid_dir.exists():
        return None

    for pid_file in pid_dir.glob("*-*.pid"):
        try:
            # Parse host-port from filename (e.g., "-.pid", "-8080.pid", "localhost-.pid")
            stem = pid_file.stem
            parts = stem.rsplit("-", 1)
            if len(parts) != 2:
                continue
            host_part, port_part = parts
            # Empty parts mean defaults were used
            host = host_part if host_part else DEFAULT_HOST
            port = int(port_part) if port_part else DEFAULT_PORT

            # Read PID from file
            pid = int(pid_file.read_text().strip())

            # Check if process is still running
            try:
                os.kill(pid, 0)
                return host, port, pid
            except OSError:
                # Process not running, clean up stale PID file
                pid_file.unlink(missing_ok=True)
        except (ValueError, FileNotFoundError):
            continue

    return None


def write_pid_file(pid_path: Path) -> None:
    """Write current PID to PID file."""
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))


def remove_pid_file(pid_path: Path) -> None:
    """Remove PID file."""
    pid_path.unlink(missing_ok=True)


def find_available_port(start_port: int = 8765, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    import socket

    for offset in range(max_attempts):
        port = start_port + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")


def stop_server(root_path: Path) -> bool:
    """Stop running server. Returns True if stopped, False if not running."""
    result = find_running_server(root_path)
    if result is None:
        return False
    host, port, pid = result
    pid_path = get_pid_file_path(root_path, host, port)
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait briefly for process to exit
        for _ in range(10):
            try:
                os.kill(pid, 0)
                import time
                time.sleep(0.1)
            except OSError:
                break
        remove_pid_file(pid_path)
        return True
    except OSError:
        return False


def spawn_daemon(root_path: Path, port: int, host: str, verbose: bool = False) -> None:
    """Spawn a new background process instead of forking.

    Fork-based daemonization doesn't work with threading libraries
    like sentence-transformers and LanceDB.
    """
    import subprocess

    python = sys.executable
    script = Path(__file__).resolve()

    # Start a new detached process with explicit arguments
    cmd = [python, str(script), "--background", "--path", str(root_path),
           "--port", str(port), "--host", host]
    if verbose:
        cmd.append("--verbose")

    with open(os.devnull, "w") as devnull:
        subprocess.Popen(
            cmd,
            stdout=devnull,
            stderr=devnull,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(root_path),
        )


async def run_server(app, host: str, port: int, verbose: bool = False):
    """Run the FastAPI server."""
    log_level = "debug" if verbose else "info"
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server = uvicorn.Server(config)
    await server.serve()


async def main(
    root_path: Path, project_config: ProjectConfig, host: str, port: int, verbose: bool = False
):
    """Run the server with the given configuration."""
    # Log configuration
    logger.info(f"Watch path: {root_path}")
    logger.info(f"Listening on port: {port}")

    # Initialize stats tracker and progress
    log_path = get_log_file_path(root_path, host, port)
    stats = StatsTracker(log_path, root_path)
    progress = IndexingProgress()

    # Initialize embedding provider
    provider = create_embedding_provider(project_config)
    embedding_service = EmbeddingService(provider)

    # Initialize vector store (path includes model name for isolation)
    db_path = get_db_path(root_path, project_config.settings.db_path, provider.model_name)
    vector_store = VectorStore(db_path, provider.dimension())
    await vector_store.connect()

    # Create file filter (shared between indexer and watcher)
    file_filter = FileFilter(root_path, project_config)

    # Initialize indexer
    indexer = FileIndexer(
        embedding_service,
        vector_store,
        file_filter,
        stats,
        progress,
    )

    # Create API server (before indexing so /status is available)
    app = create_app(vector_store, embedding_service, progress, stats)

    # Start server and indexing concurrently
    async def run_indexing():
        # Reconcile index first (remove stale files)
        removed = await indexer.reconcile_index()
        if removed:
            logger.info(f"Removed {removed} stale files from index")

        await indexer.full_index()
        stats.record_startup_complete()

        # Set up file watcher after initial index
        watcher = FileWatcher(file_filter, indexer.handle_event)
        await watcher.start()
        logger.info("File watcher started")
        return watcher

    async def run_with_indexing():
        # Run indexing in background while server starts
        indexing_task = asyncio.create_task(run_indexing())
        watcher = None
        try:
            await run_server(app, host, port, verbose)
        except asyncio.CancelledError:
            pass  # Clean shutdown requested
        finally:
            # Get the watcher if indexing completed, otherwise cancel it
            if indexing_task.done():
                try:
                    watcher = indexing_task.result()
                except Exception:
                    pass  # Indexing failed, no watcher to stop
            else:
                indexing_task.cancel()
                try:
                    await indexing_task
                except asyncio.CancelledError:
                    pass

            # Clean up resources
            if watcher:
                await watcher.stop()
            await vector_store.close()

    await run_with_indexing()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Giddyanne HTTP server")
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run as a background daemon"
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help=argparse.SUPPRESS,  # Internal flag, not shown in help
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port to listen on (default: from config or find available)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: from config)"
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop a running daemon"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check if daemon is running"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Project root path (default: current directory)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.verbose)

    # Determine root path (CLI arg or current directory)
    root_path = args.path.resolve() if args.path else Path.cwd().resolve()

    # Handle --status (doesn't need config)
    if args.status:
        result = find_running_server(root_path)
        if result:
            host, port, pid = result
            print(f"Running (PID {pid}, {host}:{port})")
            sys.exit(0)
        else:
            print("Not running")
            sys.exit(1)

    # Handle --stop (doesn't need config)
    if args.stop:
        if stop_server(root_path):
            print("Server stopped")
            sys.exit(0)
        else:
            print("Server not running")
            sys.exit(1)

    # Check if already running
    existing = find_running_server(root_path)
    if existing:
        host, port, pid = existing
        print(f"Server already running (PID {pid}, {host}:{port})")
        sys.exit(1)

    # Load project config
    config_path = root_path / ".giddyanne.yaml"
    try:
        if config_path.exists():
            project_config = ProjectConfig.load(config_path)
        else:
            project_config = ProjectConfig.default(root_path)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine host and port (CLI args override config)
    host = args.host or project_config.settings.host
    base_port = args.port or project_config.settings.port

    # Always try to find an available port starting from the configured one
    try:
        port = find_available_port(base_port)
    except RuntimeError as e:
        print(f"Port error: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle --daemon: spawn a new background process and exit
    if args.daemon:
        spawn_daemon(root_path, port, host, args.verbose)
        print(f"Server starting on port {port}")
        sys.exit(0)

    # From here on, we're running the actual server (either foreground or --background)

    # Configure logging to write to log file (for giddy log)
    log_path = get_log_file_path(root_path, host, port)
    configure_logging(args.verbose, log_path)

    # Write PID file and register cleanup
    pid_path = get_pid_file_path(root_path, host, port)
    write_pid_file(pid_path)
    atexit.register(remove_pid_file, pid_path)

    # Become process group leader so we can kill all child processes on exit
    # (sentence-transformers/PyTorch spawn workers that need to be cleaned up)
    try:
        os.setpgrp()
    except OSError:
        pass  # Already a process group leader (e.g., started with start_new_session)

    # Run the server with proper signal handling
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Handle signals for clean shutdown - kill the entire process group
    # to ensure child processes (e.g., embedding model workers) are terminated
    def handle_signal(signum, frame):
        remove_pid_file(pid_path)
        # Kill entire process group (we're the leader due to start_new_session)
        os.killpg(os.getpid(), signal.SIGKILL)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        loop.run_until_complete(main(root_path, project_config, host, port, args.verbose))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Startup failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        loop.close()
        remove_pid_file(pid_path)
