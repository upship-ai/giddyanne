"""MCP server for semantic codebase search."""

import time
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from .embeddings import EmbeddingService
from .project_config import ProjectConfig
from .vectorstore import VectorStore

if TYPE_CHECKING:
    from .engine import IndexingProgress, StatsTracker

# Will be initialized on startup
_embedding_service: EmbeddingService | None = None
_vector_store: VectorStore | None = None
_stats: "StatsTracker | None" = None
_progress: "IndexingProgress | None" = None
_project_config: ProjectConfig | None = None
_root_path: Path | None = None


def _append_file_tree(
    lines: list[str], prefix: str, file_paths: list[str], suffixes: dict[str, str]
) -> None:
    """Append tree-formatted file paths to lines with connectors."""
    tree: dict = {}
    for fp in sorted(file_paths):
        rel = fp[len(prefix):]
        if not rel:
            continue
        parts = rel.split("/")
        node = tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                node[part] = fp  # leaf: store full path for suffix lookup
            else:
                if part not in node or not isinstance(node[part], dict):
                    node[part] = {}
                node = node[part]

    def _walk(node: dict, indent: str) -> None:
        entries = sorted(node.keys())
        for i, name in enumerate(entries):
            last = i == len(entries) - 1
            connector = "\u2514\u2500\u2500 " if last else "\u251c\u2500\u2500 "
            child = node[name]
            if isinstance(child, dict):
                lines.append(f"{indent}{connector}{name}/")
                next_indent = indent + ("    " if last else "\u2502   ")
                _walk(child, next_indent)
            else:
                sfx = suffixes.get(child, "")
                lines.append(f"{indent}{connector}{name}{sfx}")

    _walk(tree, "")


def create_server() -> Server:
    """Create the MCP server with search tool."""
    server = Server("giddyanne")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="version",
                description="Return the giddyanne version.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="sitemap",
                description="List all indexed file paths. "
                "Shows what files are in the search index.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "verbose": {
                            "type": "boolean",
                            "description": "Include chunk counts and modification times",
                            "default": False,
                        },
                    },
                },
            ),
            Tool(
                name="status",
                description="Show indexing progress. "
                "Returns current state (starting/indexing/ready/error) and file counts.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="health",
                description="Server health check. Returns ok if the server is running.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="stats",
                description="Index statistics. "
                "Returns file count, chunks, index size, query latency, and startup time.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="search",
                description="Search the codebase by semantic similarity. "
                "Find files and code snippets that match a natural language query.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of what you're looking for",  # noqa: E501
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                        },
                        "search_type": {
                            "type": "string",
                            "enum": ["hybrid", "semantic", "full-text"],
                            "description": "Search mode: hybrid (default), semantic, or full-text",  # noqa: E501
                            "default": "hybrid",
                        },
                    },
                    "required": ["query"],
                },
            )
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[TextContent] | CallToolResult:
        if name == "version":
            return [TextContent(type="text", text=pkg_version("giddyanne"))]

        if name == "sitemap":
            if _vector_store is None:
                return CallToolResult(
                    content=[TextContent(type="text", text="Server not initialized")],
                    isError=True,
                )
            files = await _vector_store.list_all()
            # Make paths relative to project root
            root_prefix = str(_root_path) + "/" if _root_path else ""
            rel_files = {}
            for p, chunks in files.items():
                rel = p.removeprefix(root_prefix) if root_prefix else p
                rel_files[rel] = chunks
            files = rel_files
            all_paths = sorted(files.keys())
            verbose = arguments.get("verbose", False)

            suffixes: dict[str, str] = {}
            if verbose:
                raw_mtimes = await _vector_store.get_all_mtimes()
                # Make mtime keys relative too
                mtimes = {
                    k.removeprefix(root_prefix) if root_prefix else k: v
                    for k, v in raw_mtimes.items()
                }
                for p in all_paths:
                    suffixes[p] = (
                        f"  {files[p]} chunks  mtime {mtimes.get(p, 0.0):.0f}"
                    )

            lines: list[str] = []

            if _project_config and _project_config.paths:
                claimed: set[str] = set()
                for pc in _project_config.paths:
                    desc = f"    {pc.description}" if pc.description else ""
                    lines.append(f"{pc.path}{desc}")
                    matching = [p for p in all_paths if p.startswith(pc.path)]
                    claimed.update(matching)
                    _append_file_tree(lines, pc.path, matching, suffixes)
                other = [p for p in all_paths if p not in claimed]
                if other:
                    lines.append("Other files")
                    _append_file_tree(lines, "", other, suffixes)
            else:
                for p in all_paths:
                    lines.append(p + suffixes.get(p, ""))

            lines.append(f"\n{len(all_paths)} files indexed")

            return [TextContent(type="text", text="\n".join(lines))]

        if name == "status":
            if _progress is None:
                text = "Ready"
            else:
                d = _progress.to_dict()
                if d["state"] == "ready":
                    text = "Ready"
                elif d["state"] == "error":
                    text = f"Error: {d.get('error', 'unknown')}"
                else:
                    text = (
                        f"Indexing: {d['indexed']}/{d['total']} "
                        f"files ({d['percent']}%)"
                    )
            return [TextContent(type="text", text=text)]

        if name == "health":
            return [TextContent(type="text", text="ok")]

        if name == "stats":
            if _vector_store is None:
                return CallToolResult(
                    content=[TextContent(type="text", text="Server not initialized")],
                    isError=True,
                )
            files = await _vector_store.list_all()
            lines = [
                f"Indexed files: {len(files)}",
                f"Total chunks:  {sum(files.values())}",
                f"Index size:    {_vector_store.get_db_size()} bytes",
            ]
            if _stats:
                latency = _stats.avg_query_latency_ms()
                if latency is not None:
                    lines.append(f"Avg latency:   {latency:.1f}ms")
                startup = _stats.startup_duration_ms()
                if startup is not None:
                    lines.append(f"Startup time:  {startup:.0f}ms")
            return [TextContent(type="text", text="\n".join(lines))]

        if name != "search":
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True,
            )

        if _embedding_service is None or _vector_store is None:
            return CallToolResult(
                content=[TextContent(type="text", text="Server not initialized")],
                isError=True,
            )

        query = arguments.get("query", "")
        limit = arguments.get("limit", 10)
        search_type = arguments.get("search_type", "hybrid")

        if not query:
            return CallToolResult(
                content=[TextContent(type="text", text="Query is required")],
                isError=True,
            )

        try:
            start = time.perf_counter()
            cache_hit = False

            # Route to appropriate search method based on search_type
            if search_type == "full-text":
                results = await _vector_store.search_fts(query, limit=limit)
            else:
                query_embedding, cache_hit = await _embedding_service.embed_query(
                    query, cache=_vector_store
                )
                if search_type == "hybrid":
                    results = await _vector_store.search_hybrid(
                        query_embedding, query, limit=limit
                    )
                else:  # semantic
                    results = await _vector_store.search(query_embedding, limit=limit)
            if _stats:
                _stats.record_search(
                    time.perf_counter() - start, cache_hit=cache_hit, query=query
                )

            if not results:
                return [TextContent(type="text", text="No results found")]

            output_lines = []
            for r in results:
                # Format: path:start-end for chunk location
                location = f"{r.path}:{r.start_line}-{r.end_line}"
                output_lines.append(
                    f"## {location} (score: {r.score:.2f})\n```\n{r.content}\n```\n"
                )

            return [TextContent(type="text", text="\n".join(output_lines))]

        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Search failed: {e}")],
                isError=True,
            )

    return server


def init_server(
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    stats: "StatsTracker | None" = None,
    progress: "IndexingProgress | None" = None,
    project_config: ProjectConfig | None = None,
    root_path: Path | None = None,
) -> None:
    """Initialize the server with dependencies."""
    global _embedding_service, _vector_store, _stats, _progress
    global _project_config, _root_path
    _embedding_service = embedding_service
    _vector_store = vector_store
    _stats = stats
    _progress = progress
    _project_config = project_config
    _root_path = root_path


async def run_mcp_server(
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    stats: "StatsTracker | None" = None,
    progress: "IndexingProgress | None" = None,
    project_config: ProjectConfig | None = None,
    root_path: Path | None = None,
) -> None:
    """Run the MCP server over stdio."""
    init_server(embedding_service, vector_store, stats, progress, project_config, root_path)
    server = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
