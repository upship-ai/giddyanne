# User Guide

This guide covers how to get the most out of giddyanne's semantic search.

## How Semantic Search Works

Giddyanne uses embedding models to understand the *meaning* of code, not just match keywords. This means:

- **"auth logic"** finds `login.py`, `session_manager.py`, `permissions.go`
- **"error handling"** finds try/catch blocks, error types, recovery code
- **"database queries"** finds SQL, ORM calls, connection pooling

The search combines two approaches:
- **Semantic search**: Finds code with similar meaning to your query
- **Full-text search**: Finds exact matches for specific terms

By default, results are ranked using both (hybrid search). Use `--semantic` or `--full-text` flags to use only one.

## Writing Good Queries

### What Works Well

**Conceptual queries** - describe what the code does:
```bash
giddy find "user authentication"
giddy find "parse command line arguments"
giddy find "send email notifications"
```

**Problem-oriented queries** - describe what you're trying to solve:
```bash
giddy find "handle network timeouts"
giddy find "validate user input"
giddy find "cache expensive computations"
```

**Behavior descriptions** - describe what happens:
```bash
giddy find "retry failed requests"
giddy find "log errors to file"
giddy find "convert dates to timestamps"
```

### What Works Less Well

**Variable names** - use grep instead:
```bash
# Less effective:
giddy find "userAuthToken"

# Better - use grep for exact matches:
grep -r "userAuthToken" src/
```

**Very short queries** - add context:
```bash
# Too vague:
giddy find "config"

# Better:
giddy find "load configuration from yaml file"
```

### Tips

1. **Use natural language** - write queries like you'd describe the code to someone
2. **Be specific** - "database connection pooling" beats "database stuff"
3. **Include verbs** - "validate", "parse", "send", "handle" help semantic matching
4. **Add context** - "http error handling" is better than just "errors"

## Config Reference

The `.giddyanne.yaml` file controls what gets indexed and how. Here's every option:

### Paths (required)

```yaml
paths:
  - path: src/
    description: Main application code
  - path: tests/
    description: Test suite
```

| Field | Required | Description |
|-------|----------|-------------|
| `path` | Yes | File or directory to index (relative to config file) |
| `description` | No | Human-readable description, embedded alongside content for better search |

**Tip:** More specific descriptions improve search quality. Instead of "code", use "authentication and session management".

### Maximal Example

A complete config showing all available options:

```yaml
paths:
  # Index the entire project with a general description
  - path: .
    description: Web application with REST API

  # Override with more specific descriptions for key directories
  - path: src/api/
    description: REST endpoints, request handlers, middleware

  - path: src/auth/
    description: Authentication, JWT tokens, session management, permissions

  - path: src/db/
    description: Database models, migrations, query builders

  - path: src/utils/
    description: Shared utilities, helpers, validation functions

  - path: tests/
    description: Test suite, fixtures, mocks

settings:
  # File handling
  max_file_size: 1000000           # Skip files > 1MB
  ignore_patterns:                  # Beyond .gitignore
    - "*.generated.*"
    - "*.min.js"
    - "vendor/"

  # Server
  host: "0.0.0.0"                  # Bind address
  port: 8000                        # HTTP port

  # Chunking (defaults work well for most codebases)
  min_chunk_lines: 10
  max_chunk_lines: 50
  overlap_lines: 5

  # Embedding model
  local_model: "all-MiniLM-L6-v2"  # Or: all-mpnet-base-v2, paraphrase-MiniLM-L3-v2
```

### Settings Reference

#### File Handling

| Setting | Default | Description |
|---------|---------|-------------|
| `max_file_size` | 1000000 | Skip files larger than this (bytes). 1MB default. |
| `ignore_patterns` | [] | Additional patterns to ignore (beyond .gitignore). Uses gitignore syntax. |

#### Server

| Setting | Default | Description |
|---------|---------|-------------|
| `host` | "0.0.0.0" | Network interface to bind to |
| `port` | 8000 | HTTP port for the server |

#### Chunking

Code is split into chunks for indexing. These settings control chunk boundaries:

| Setting | Default | Description |
|---------|---------|-------------|
| `min_chunk_lines` | 10 | Minimum lines per chunk |
| `max_chunk_lines` | 50 | Maximum lines per chunk |
| `overlap_lines` | 5 | Lines of overlap between chunks |

Defaults work well for most codebases. Smaller chunks give more precise results but may lose context.

#### Embedding

| Setting | Default | Description |
|---------|---------|-------------|
| `local_model` | "all-MiniLM-L6-v2" | Sentence-transformers model name |

The database is stored at `.giddyanne/<model-name>/vectors.lance`. This allows switching models without re-indexing (each model gets its own index).

The default model is a good balance of speed and quality. Other options:
- `all-mpnet-base-v2`: Higher quality, slower
- `paraphrase-MiniLM-L3-v2`: Faster, lower quality

## Troubleshooting

### "No results found"

1. **Check server status**: Run `giddy status` - is indexing complete?
2. **Rephrase your query**: Try describing what the code does differently
3. **Check paths**: Does your config include the directory you're searching?
4. **Try hybrid search**: If using `--semantic` or `--full-text`, try without flags

### "Config file not found"

You need a `.giddyanne.yaml` in your project root:

```bash
giddy init          # Shows a prompt to help you create one
```

Or create a minimal config:

```yaml
paths:
  - path: .
    description: My project
```

### Server won't start

**Port already in use:**
```bash
giddy down          # Stop any existing server
giddy up --port 9000  # Use a different port
```

**Check logs:**
```bash
giddy log           # Stream server logs
```

### Slow indexing

First index takes time (~45s for 750 files on M1 Pro). Subsequent startups are faster because:
- Unchanged files use cached embeddings
- Only modified files get re-indexed

If re-indexing is slow:
1. Check `giddy status` for progress
2. Consider excluding large directories via `ignore_patterns`
3. Check if files are being repeatedly modified

### High memory usage

The embedding model uses ~500MB RAM. If this is a problem:
- Use a smaller model: `local_model: "paraphrase-MiniLM-L3-v2"`
- Stop the server when not in use: `giddy down`

### Search returns irrelevant results

1. **Add descriptions**: Better path descriptions improve ranking
2. **Be more specific**: Longer, more descriptive queries help
3. **Check file types**: Only supported languages are indexed (see README)

### Emacs/VSCode not connecting

1. Ensure the server is running: `giddy status`
2. Check you're in a directory with `.giddyanne.yaml`
3. For MCP: verify `giddy mcp` works from command line

## Advanced: Internal Files

All runtime files live in `.giddyanne/` in your project root:

```
.giddyanne/
├── all-MiniLM-L6-v2/     # Named after your embedding model
│   ├── vectors.lance/    # Vector database (LanceDB)
│   └── mcp.log           # MCP server log
├── -8000.log             # HTTP server log (host-port.log)
└── -8000.pid             # Server PID file (host-port.pid)
```

### Log Files

**HTTP server log** (`.giddyanne/<host>-<port>.log`):
```bash
# View directly
cat .giddyanne/-8000.log

# Or stream live
giddy log
```

Contains indexing progress, file watch events, and search requests.

**MCP server log** (`.giddyanne/<model>/mcp.log`):
```bash
cat .giddyanne/all-MiniLM-L6-v2/mcp.log
```

Contains MCP protocol messages and search requests from Claude Code.

### PID Files

The `.pid` file tracks the running server process:

```bash
# Check if server is actually running
cat .giddyanne/-8000.pid  # Shows PID
ps -p $(cat .giddyanne/-8000.pid)  # Verify process exists
```

If `giddy status` says the server is running but it's not responding, the PID file may be stale. Remove it and restart:

```bash
rm .giddyanne/-8000.pid
giddy up
```

### Database Files

The `vectors.lance/` directory is a LanceDB database containing your indexed embeddings.

**To force a re-index** (keeps logs):
```bash
giddy drop
```

**To reset everything** (removes all .giddyanne data):
```bash
giddy clean        # prompts for confirmation
giddy clean --force  # no confirmation
```
