"""MCP server for semantic codebase search."""

import time
from typing import TYPE_CHECKING

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from .embeddings import EmbeddingService
from .vectorstore import VectorStore

if TYPE_CHECKING:
    from mcp_main import StatsTracker

# Will be initialized on startup
_embedding_service: EmbeddingService | None = None
_vector_store: VectorStore | None = None
_stats: "StatsTracker | None" = None


def create_server() -> Server:
    """Create the MCP server with search tool."""
    server = Server("giddyanne")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
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
) -> None:
    """Initialize the server with dependencies."""
    global _embedding_service, _vector_store, _stats
    _embedding_service = embedding_service
    _vector_store = vector_store
    _stats = stats


async def run_mcp_server(
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    stats: "StatsTracker | None" = None,
) -> None:
    """Run the MCP server over stdio."""
    init_server(embedding_service, vector_store, stats)
    server = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
