# Installation

The `giddy` binary needs the Python server code to run. Two options:

## Option 1: Pre-built binary (no Go required)

```bash
# Clone the repo
git clone https://github.com/mazziv/giddyanne.git
cd giddyanne

# Download the binary for your platform from Releases
# and move it into the repo root (same directory as main.py)

# On macOS, clear the quarantine flag
xattr -d com.apple.quarantine giddy

# Set up Python environment
make python

# Add to PATH (symlink preserves the path to Python code)
ln -sf "$(pwd)/giddy" ~/bin/giddy
```

Requires Python 3.10+.

## Option 2: Build from source

```bash
git clone https://github.com/mazziv/giddyanne.git
cd giddyanne
make install   # Builds Go + Python, symlinks to ~/bin
```

Requires Go 1.21+ and Python 3.10+.
