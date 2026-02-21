"""MCP server entrypoint for Giddyanne."""

import asyncio
import logging
import os
import sys
from pathlib import Path

from src.mcp_server import run_mcp_server
from src.project_config import ConfigError, ProjectConfig
from src.startup import create_components, run_indexing, setup_signal_handlers, start_watcher

logging.basicConfig(level=logging.WARNING)  # Quiet for MCP stdio
logger = logging.getLogger(__name__)


async def main():
    # Get root path from environment or current directory
    # (Go CLI passes GIDDY_WATCH_PATH when spawning MCP server)
    root_path_str = os.environ.get("GIDDY_WATCH_PATH")
    root_path = Path(root_path_str).resolve() if root_path_str else Path.cwd().resolve()
    config_path = root_path / ".giddyanne.yaml"

    # Load project config and determine storage dir
    if config_path.exists():
        project_config = ProjectConfig.load(config_path)
        storage_dir = root_path / ".giddyanne"
    else:
        project_config = ProjectConfig.default(root_path)
        storage_dir = ProjectConfig.get_tmp_storage_dir(root_path)

    # Initialize all components
    c = await create_components(root_path, project_config, storage_dir, log_filename="mcp.log")

    # Test embedding provider before indexing
    try:
        await c.provider.embed("test")
    except Exception as e:
        print(f"Warning: Embedding provider failed: {e}", file=sys.stderr)

    # Index synchronously (MCP needs index ready before serving)
    await run_indexing(c)
    await start_watcher(c)

    await run_mcp_server(
        c.embedding_service, c.vector_store, c.stats, c.progress,
        project_config, root_path,
    )


if __name__ == "__main__":
    setup_signal_handlers()

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
