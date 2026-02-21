# Installation

**Platforms:** macOS, Linux

## Prerequisites

- Python 3.11+
- Go 1.21+ (or use pre-built binary)

## Build from Source

```bash
git clone --depth 1 --branch v1.3.2 https://github.com/mazziv/giddyanne.git
cd giddyanne
make install
```

This builds the CLI, creates a Python virtualenv, and symlinks `giddy` to `~/bin`.

To install elsewhere:

```bash
make install BIN_DIR=/usr/local/bin  # or any directory in your PATH
```

## Pre-built Binary

If you don't have Go installed:

```bash
# Clone the repo (needed for Python server code)
git clone --depth 1 --branch v1.3.2 https://github.com/mazziv/giddyanne.git
cd giddyanne

# Download binary for your platform from Releases:
# https://github.com/mazziv/giddyanne/releases
# Move it into the repo root (same directory as http_main.py)

# On macOS, clear the quarantine flag
xattr -d com.apple.quarantine giddy

# Set up Python environment
make python

# Symlink to PATH (preserves path to Python code)
ln -sf "$(pwd)/giddy" ~/bin/giddy
```

## First Run

First install downloads dependencies (~2GB for PyTorch). This is a one-time download.

Then download the embedding model:

```bash
giddy install      # Download embedding model (~90MB, one-time)
```

If you skip this step, the model downloads automatically on first `giddy up` — but the
explicit install step makes the wait visible instead of silently blocking startup.

Initial indexing takes time depending on codebase size:
- ~750 files: ~45 seconds on M1 Pro
- Runs once at startup, then watches for changes
- Subsequent starts are faster (unchanged files use cached embeddings)

## Verify Installation

```bash
giddy help
# Shows available commands
```

## Uninstall

```bash
# Remove symlink
rm ~/bin/giddy  # or wherever you installed it

# Remove the repo
rm -rf /path/to/giddyanne

# Optional: remove cached models
rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
```

[← Back to README.md](README.md)
