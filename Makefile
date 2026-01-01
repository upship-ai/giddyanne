.PHONY: all go python vscode install clean

BIN_DIR := $(HOME)/bin
GO_SOURCES := $(shell find cmd -name '*.go') go.mod
VSCODE_SOURCES := $(shell find vscode/src -name '*.ts') vscode/package.json

all: giddy .venv/.installed

go: giddy

giddy: $(GO_SOURCES)
	go build -o giddy ./cmd/giddy

python: .venv/.installed

.venv:
	python3 -m venv .venv

.venv/.installed: .venv pyproject.toml
	.venv/bin/pip install -q -e .
	@touch $@

install: all
	mkdir -p $(BIN_DIR)
	ln -sf $(CURDIR)/giddy $(BIN_DIR)/giddy
	@echo ""
	@echo "Installed to $(BIN_DIR)/giddy"
	@echo ""
	@echo "Make sure $(BIN_DIR) is in your PATH, then run 'giddy init' in any project."

vscode: vscode/giddyanne-1.1.1.vsix

vscode/node_modules: vscode/package.json
	cd vscode && npm install --silent
	@touch $@

vscode/giddyanne-1.1.1.vsix: vscode/node_modules $(VSCODE_SOURCES)
	cd vscode && npm run compile --silent && npx @vscode/vsce package -o giddyanne-1.1.1.vsix
	@echo ""
	@echo "Built vscode/giddyanne-1.1.1.vsix"
	@echo "Install with: code --install-extension vscode/giddyanne-1.1.1.vsix"

clean:
	rm -f giddy .venv/.installed vscode/giddyanne-*.vsix
	rm -rf vscode/out vscode/node_modules
