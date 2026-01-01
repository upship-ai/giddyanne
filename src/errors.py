"""Giddyanne error types.

Errors are defined close to where they're raised:
- ConfigError: src/project_config.py
- StorageError: src/vectorstore.py
- SearchError: here (used by both servers)
"""


class SearchError(Exception):
    """Raised when search operations fail.

    Used by both HTTP and MCP servers to signal search-time failures
    like embedding service errors or vector store query failures.
    """

    pass
