# Changelog

All notable changes to giddyanne will be documented in this file.

## [Unreleased]

## [1.2.1] - 2026-02-12

### Added

- **`giddy install` command**: Downloads the embedding model (~90MB) upfront so `giddy up` starts immediately without a silent download pause

## [1.2.0] - 2026-02-12

### Added

- **`--version` / `-V` flag** for CLI (`giddy --version` prints `giddy 1.2.0`)
- **MCP `version` tool** returns the installed giddyanne version
- **Single version source of truth**: `pyproject.toml` drives all version strings — Go CLI reads it at build time via ldflags, Python reads via `importlib.metadata`
- **Zero-config git repo support**: `giddy` now works in any git repo without a `.giddyanne.yaml` config file. Auto-detects common source directories (`src/`, `lib/`, `app/`, `tests/`, `docs/`) and stores indexes in the system temp directory. Config file still takes priority when present.

## [1.1.1] - 2025-12-31

Installation improvements.

### Added

- Pre-built binaries for macOS (arm64, amd64) and Linux (amd64) attached to GitHub releases
- Flexible install location: `make install BIN_DIR=/usr/local/bin`

## [1.1.0] - 2025-12-31

Search quality improvements, Emacs integration, and CLI polish.

### Added

**Search Quality**
- Language-aware chunking: split code at function/class boundaries using language-specific separators
- Hybrid search: BM25 + semantic with `--semantic`, `--full-text`, `--hybrid` (default) modes
- Model-specific vector stores: database path includes embedding model name
- Faster initial import: parallelized file I/O + batched encode() calls

**CLI**
- Shell completions for bash, zsh, and fish
- Health metrics: index size, query latency via `/health` endpoint and `giddy health`
- Prefix shortcuts: `giddy f` for `find`, `giddy st` for `status`, etc.
- Better startup experience: friendlier first-run, clearer server feedback

**Emacs**
- Full integration package with Vertico support and nerd-icons
- Commands: `giddyanne-find`, `giddyanne-up/down`, `giddyanne-status`, `giddyanne-health`, `giddyanne-log`
- Results grouped by file with pulse highlight on jump
- Installation docs for Doom Emacs

**VSCode**
- Extension with QuickPick search UI
- Commands: Find, Start/Stop Server, Status, Health, Stream Logs
- Results grouped by file with pulse highlight on jump
- Keybinding: `Cmd+Shift+G` / `Ctrl+Shift+G`
- Build with `make vscode`, install via `.vsix`

**Languages**
- Language support expanded from 6 to 28 languages:
  - **Programming:** C, C++, C#, Dart, Elixir, Java, Kotlin, Lua, Objective-C, PHP, R, Ruby, Scala, Shell, Swift, Zig
  - **Markup & config:** CSS, HTML, JSON, Markdown, TOML, YAML

### Fixed
- INFO logs no longer appear without `--verbose` flag
- `giddy health` shows current state when server is down (like `giddy status`)
- `giddy clean` finds model-specific index paths
- `--files-only --json` returns only path and lines, not content

### Breaking Changes
- **Database path includes model name.** Migration: `mkdir -p .giddyanne/all-MiniLM-L6-v2 && mv .giddyanne/vectors.lance .giddyanne/all-MiniLM-L6-v2/`
- **Removed OpenRouter support.** Remove `GIDDY_OPENROUTER_API_KEY` env var and `openrouter_model` from config

## [1.0.0] - 2025-12-30

First stable release. Semantic code search for codebases via CLI and MCP server.

### Features

**Core**
- Semantic code search using local sentence-transformers embeddings
- LanceDB vector store with chunk-based schema
- Hybrid chunking: split on blank lines, merge small chunks, split large with overlap
- File watcher for automatic re-indexing on changes
- Search query caching with hit statistics
- File timestamp tracking to skip unchanged files on startup
- Index reconciliation removes stale files on startup
- Glob patterns in ignore settings

**CLI (`giddy`)**
- `giddy search <query>` - search with auto-start if server not running
- `giddy start` - start the server daemon
- `giddy stop` - stop the server
- `giddy status` - show server status and indexing progress
- `giddy monitor` - stream server logs
- `giddy clean` - clear the search index
- `giddy init` - generate config file prompt
- `giddy mcp` - run MCP server for Claude Code
- `--verbose` flag for debug logging
- `--port`, `--host` flags for server configuration

**HTTP API**
- `POST /search` - semantic search
- `GET /status` - indexing progress
- `GET /health` - liveness check
- `GET /stats` - index statistics

**MCP Server**
- Single `search` tool for Claude Code integration
- Description embeddings for better context matching

**Configuration**
- `.giddyanne.yaml` project config with paths and descriptions
- Configurable chunking parameters
