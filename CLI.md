# CLI Reference

Run `giddy help` for built-in help.

## Commands

| Command | Description |
|---------|-------------|
| `giddy find <query>` | Search the codebase (auto-starts server if needed) |
| `giddy up` | Start the server |
| `giddy down` | Stop the server |
| `giddy bounce` | Restart the server |
| `giddy status` | Show server status and indexing progress |
| `giddy health` | Show index statistics (files, chunks, size) |
| `giddy sitemap` | List all indexed files |
| `giddy log` | Stream server logs in real-time |
| `giddy drop` | Remove search index (keeps logs) |
| `giddy clean` | Remove all giddyanne data (with confirmation) |
| `giddy init` | Generate a config file prompt |
| `giddy mcp` | Run MCP server (for Claude Code) |

## Abbreviations

Commands can be shortened to their prefix:

| Short | Full |
|-------|------|
| `giddy f` | `giddy find` |
| `giddy st` | `giddy status` |
| `giddy h` | `giddy health` |
| `giddy si` | `giddy sitemap` |
| `giddy l` | `giddy log` |
| `giddy b` | `giddy bounce` |

## Flags

### Server flags

```bash
giddy up --port 9000      # Use specific port
giddy up --host localhost # Bind to specific host
giddy up --verbose        # Enable debug logging
```

### Search flags

```bash
giddy find --limit 20     # Return more results (default: 10)
giddy find --json         # Output as JSON
giddy find --files-only   # Only show file paths
giddy find --verbose      # Show full content (no truncation)
giddy find --semantic     # Semantic search only
giddy find --full-text    # Full-text search only
```

### Other flags

```bash
giddy --version           # Print version and exit
giddy -V                  # Short form
giddy health --verbose    # List all files with chunk counts
giddy sitemap             # List all indexed file paths
giddy sitemap --verbose   # Include chunk counts and modification times
giddy sitemap --json      # Output as JSON
giddy clean --force       # Skip confirmation prompt
```

## Shell Completions

Enable tab completion for commands and flags.

**Bash** (add to `~/.bashrc`):
```bash
eval "$(giddy completion bash)"
```

**Zsh** (add to `~/.zshrc`):
```zsh
eval "$(giddy completion zsh)"
```

**Fish** (add to `~/.config/fish/config.fish`):
```fish
giddy completion fish | source
```

![Shell completions](img/autocomplete.png)

[← Back to README.md](README.md)
