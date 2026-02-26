# Changelog

All notable changes to giddyanne will be documented in this file.

## [1.6.2] - 2026-02-25

### Fixed

- **Lint fixes**: Cleaned up linter warnings in engine and benchmark code

## [1.6.1] - 2026-02-25

### Fixed

- **Watcher ignores node_modules, .giddyanne, .git, etc.**: Hardcoded `ALWAYS_IGNORE_DIRS` set that is always excluded from watching and indexing regardless of config. Previously, watchdog registered inotify watches on every subdirectory including `node_modules` (10k+ watches on a typical JS project, now ~140).
- **Non-recursive watcher with selective directory walks**: Replaced `recursive=True` Observer with manual directory walking that prunes ignored dirs. New directories are auto-watched via `DirCreatedEvent`. Eliminates thousands of wasted kernel inotify watches.
- **Batch upserts in watcher-triggered indexing**: `index_file` (the single-file reindex path) now uses `embed_chunks_batch` + `upsert_batch` instead of per-chunk embed/upsert loops. A 22-chunk file previously created 22 LanceDB fragments triggering expensive compaction; now creates 1.
- **Diagnostic logging for memory leaks**: RSS tracking via `/proc/self/status` at every indexing phase, per-file size logging, stats-log output for WATCH/INDEX/PREP/EMBED events visible via `giddy log`. Watcher `fire()` callback now catches and logs async exceptions.

### Added

- **`make qa` target**: Runs linter (`ruff check`) and tests (`pytest`) in one command.
- **GitHub Actions CI**: `make qa` runs automatically on pushes and PRs to `main`, with venv caching.

## [1.6.0] - 2026-02-23

### Added

- **Emacs Lisp language support**: `.el` files are now indexed with tree-sitter AST chunking, splitting on `defun`, `defmacro`, and special forms (`defvar`, `defcustom`, etc.)

### Changed

- **Index all project files by default**: All files with supported extensions are now indexed automatically, regardless of whether they appear in `paths` config. The `paths` config is now description-only — it adds annotation context that improves search relevance, but doesn't control what gets indexed. `.gitignore` and `ignore_patterns` still apply. Watcher covers the entire project root instead of only configured directories.

## [1.5.0] - 2026-02-22

### Added

- **Shared embedding server**: A global singleton service (`embed_server.py`) that loads the embedding model once and serves all project servers over a Unix socket. Project servers that previously loaded their own ~900MB model copy now proxy through the shared server at ~180MB each. Auto-starts on `giddy up` if not already running. Manage directly with `giddy embed start|stop|status`. Configurable via `~/.config/giddyanne/config.yaml` (socket path, model, auto_start). Falls back to in-process embedding if the shared server is unavailable.
- **`giddy embed` CLI command**: `giddy embed start`, `giddy embed stop`, `giddy embed status` for managing the shared embedding server independently of any project.
- **Global config** (`~/.config/giddyanne/config.yaml`): Optional config for the shared embedding service — socket path, PID/log paths, model name, and auto_start toggle.
- **Cross-encoder reranker** (C4): Optional second-stage reranker using `cross-encoder/ms-marco-MiniLM-L6-v2` (22.7M params). Over-fetches candidates (5x instead of 3x), runs existing pipeline (RRF, category bias, dedup), then re-scores with the cross-encoder before returning. Opt-in via `reranker_model` in `.giddyanne.yaml`. Uses `CrossEncoder` from sentence-transformers (no new dependency). Benchmarks: giddyanne recall 35%→80% (hybrid), MRR 0.34→0.90. test_2 FTS recall 60%→70%, MRR 0.40→0.70. test_1 MRR improved (+0.05-0.15) but semantic recall regressed 87%→67% on one query where the right file wasn't in the candidate pool. Disabled by default — best for repos where relevant files are retrieved but poorly ranked.
- **`--rerank` CLI flag**: Enable the cross-encoder reranker without a config file. `giddy up --rerank` and `giddy bounce --rerank` apply the default reranker model. For MCP, set `GIDDY_RERANK=1` environment variable.
- **Token-aware chunk splitting**: After language-aware chunking, oversized chunks are recursively halved until they fit the model's token budget (256 tokens for MiniLM). Eliminates silent embedding truncation — content coverage went from 56% to 100%. Chunk count roughly doubles (313→740 for giddyanne repo). Benchmarks: test_2 recall 70%→77%, MRR 0.70→0.85. Semantic-only search now matches old hybrid baseline. Index times ~2x from more chunks. Ollama backend now impractical without batching (timed out at 600s on 32 files due to per-request HTTP overhead × ~2200 embed calls).
- **Ollama embedding backend**: GPU-accelerated embeddings via a local Ollama server. Opt-in via `--ollama` CLI flag, `GIDDY_OLLAMA=1` env var, or `ollama: true` in config. Configurable URL and model (`ollama_url`, `ollama_model`). Default model: `all-minilm:l6-v2`.
- **Truncation stats**: `/stats` endpoint and `giddy stats` now report embedding coverage — how many chunks were truncated and what percentage of content is fully covered.

### Changed

- **BM25 field boosting** (C3): Added `fts_content` column that repeats file path and symbol name 3x before raw content, simulating BM25F field-level boosting. FTS now indexes this enriched column instead of raw content. Benchmarks: test_1 full-text recall 63%→87%, "grid" query ranks `grid.tsx` #1. test_2 Stripe query ranks `stripe.ts` #1. giddyanne hybrid MRR holds at 1.00. Requires `giddy clean --force && giddy up` to rebuild index.
- **Chunk context enrichment** (C1): Prepend file path and function/class name to chunk content before embedding. Stored content unchanged (search results still show raw code). Benchmarks: giddyanne 75% recall (steady, "configuration and settings" query improved 0%→50%), test_2 80%→87%, test_1 77%→70% (one already-weak query regressed).
- **Custom RRF with tunable weights** (C2): Replaced LanceDB's black-box `RRFReranker` with our own weighted Reciprocal Rank Fusion. Runs semantic and FTS searches independently, merges ranked lists with configurable weights (semantic 1.2, FTS 0.5). Removes `lancedb.rerankers` dependency from hybrid search. Benchmarks: recall matches semantic-only (giddyanne 75%, test_2 70%, test_1 83%), MRR improved (giddyanne 0.65→1.00, test_2 0.65→0.70).

### Added

- **Quality benchmark script** (`benchmarks/quality.py`): Reusable benchmark for measuring search recall, precision, and MRR across projects and search modes. Supports `--mode all` for side-by-side comparison of semantic, full-text, and hybrid search.

## [1.4.1] - 2026-02-21

### Fixed

- **Version references**: Updated all version strings across docs, Emacs package, and VSCode extension

## [1.4.0] - 2026-02-21

### Added

- **Search result deduplication**: Results are now deduplicated by file — only the best-scoring chunk per file is kept, freeing result slots for diverse files. Applied to all three search methods (semantic, full-text, hybrid). Over-fetches from LanceDB to compensate for collapsed results.
- **File category scoring bias**: Search scores are now weighted by file type — source code files rank at full weight (1.0×), test files at 0.8×, and docs at 0.6×. When a test file and its source have similar relevance, the source file surfaces first.

### Changed

- **Tree-sitter AST chunking**: Replaced string-based separator splitting with tree-sitter AST parsing for code chunking. Functions, classes, and other definitions are now split at real node boundaries instead of pattern-matching on strings like `"\ndef "`. Falls back to blank-line splitting for languages without a tree-sitter grammar. Adds `tree-sitter-language-pack` dependency.
- **Configurable embedding model**: `local_model` in `.giddyanne.yaml` lets you swap embedding models. Each model gets its own index directory. Default remains `all-MiniLM-L6-v2` — benchmarking showed `nomic-ai/CodeRankEmbed` (768d, code-specialized) indexes 5-6x slower and can't finish indexing a modest ~300-file repo on a MacBook Pro, with no quality gain to justify it.
- **Cosine distance for vector search**: Switched from L2 to cosine distance. Scores now display as meaningful 0-1 values instead of near-zero floats.
- **Query prefix support**: Models that require instruction prefixes on search queries (like CodeRankEmbed) are handled automatically. Document embeddings are unaffected.
- **`giddy install` reads model from config**: No longer hardcodes the model name — picks up `local_model` from `.giddyanne.yaml` or uses the default.

## [1.3.2] - 2026-02-20

### Added

- **MCP status/health/stats tools**: AI assistants can now check indexing progress, server health, and index statistics directly via MCP

### Fixed

- **`giddy find` waits for indexing on cold start**: Previously returned a 503 error and exited if the server was still indexing. Now silently waits until ready, then returns results. Use `--verbose` to see progress.

## [1.3.1] - 2026-02-20

### Added

- **Sitemap command in Emacs and VSCode**: `giddyanne-sitemap` (Emacs) and `Giddyanne: Sitemap` (VSCode) show indexed files as a tree

### Fixed

- Fix empty tree node for standalone file path groups in MCP sitemap output
- Fix unused variable warning in MCP server entrypoint
- Fix `fmt.Println` redundant newline warning in Go CLI

## [1.3.0] - 2026-02-20

### Added

- **Sitemap command**: new `/sitemap` endpoint, MCP `sitemap` tool, and `giddy sitemap` CLI command. Shows all indexed files, optionally with chunk counts and modification times (`--verbose`). API response includes a `paths` field with `path` and `description` for each configured group.
- **File-tree sitemap output**: sitemap displays files under their path groups with `├──`/`└──` tree connectors, intermediate directories reconstructed. Verbose mode appends chunk count and mtime inline.
- **Sitemap API tests**: test coverage for `/sitemap` endpoint with and without config, including verbose mode and empty index

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
