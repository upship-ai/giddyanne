# Giddyanne

**Find code by meaning, not keywords. Completely offline.**

Search your codebase the way you think about it: "authentication logic", "date formatting", "API error handling". Giddyanne understands what you mean and finds the right code—even if it uses different words.

```bash
giddy find "authentication logic"
> src/auth/login.py:45-67 (0.89)
>   def verify_user_credentials(username, password):
```

## Why Giddyanne?

**Private** – Your code never leaves your machine. No cloud APIs, no data collection, no privacy concerns.

**Fast** – Search happens in milliseconds after initial indexing. Pre-computed embeddings mean instant results.

**Local** – Runs entirely on your hardware using sentence-transformers. No internet required, no API keys, no rate limits.

**Smart** – Semantic search understands intent. "Where do I format dates?" finds `formatTimestamp()` even though the words don't match.

**Integrated** - Works where you work. VSCode, Emacs, CLI, MCP for AI assistants, or HTTP API for custom workflows.

**v1.3.0** · [Changelog](CHANGELOG.md)

---

## How It Works

1. **Index once** – Giddyanne scans your project and creates embeddings for every code chunk
2. **Search semantically** – Query in natural language to find relevant code by meaning
3. **Stay in sync** – File watcher automatically re-indexes when code changes

Unlike GitHub Copilot or cloud-based code search, everything runs locally. Unlike grep or ripgrep, it understands semantic meaning. You get the best of both worlds: powerful semantic search with complete privacy.

## Quickstart

```bash
git clone --depth 1 --branch v1.3.0 https://github.com/mazziv/giddyanne.git
cd giddyanne && make install
```

Then in any git repo:

```bash
# Search semantically (auto-starts server, no config needed)
giddy find "error handling for API calls"
> src/api/client.py:123-145 (0.76)
>   def handle_request_error(response):

# Check what's indexed
giddy health
> Indexed files: 28
> Total chunks:  288
> Index size:    3.85 MB

# Stop the server
giddy down
```

Works out of the box in any git repo. For better results, add a `.giddyanne.yaml` config with path descriptions — see [CONFIG.md](CONFIG.md) or run `giddy init`.

## Real-World Examples

| You search for... | It finds... |
|-------------------|-------------|
| "authentication logic" | `verify_credentials()`, `check_auth_token()`, `login_handler()` |
| "format dates" | `formatTimestamp()`, `parseDateTime()`, `dateToString()` |
| "database queries" | SQL functions, ORM calls, query builders |
| "error handling" | try/catch blocks, error handlers, exception classes |

The search understands synonyms, related concepts, and programming patterns—not just exact text matches.

## Use Cases

- **Onboarding**: New to a codebase? Search "how does authentication work" to find all related code
- **Refactoring**: Find all similar implementations across your project
- **Code review**: Quickly locate relevant context for a change
- **Documentation**: Search by concept, not by memorizing function names
- **Learning**: Explore unfamiliar codebases by searching for what they do, not what they're named

## Integrations

| Integration | Use it for... |
|-------------|---------------|
| **CLI** | Terminal-based search and server management |
| **Emacs** | Search directly from your editor |
| **VSCode** | Extension with inline search |
| **MCP** | AI assistants can search your codebase |
| **HTTP API** | Build your own integrations |

See [INTEGRATIONS.md](INTEGRATIONS.md) for setup instructions.

## Supported Languages

Only files with supported extensions are indexed. Code is split at natural boundaries (functions, classes) for better search results.

**Programming:** C, C++, C#, Dart, Elixir, Go, Java, JavaScript, Kotlin, Lua, Objective-C, PHP, Python, R, Ruby, Rust, Scala, Shell, SQL, Swift, TypeScript, Zig

**Markup & config:** CSS, HTML, JSON, Markdown, TOML, YAML

Files matching `.gitignore` patterns are automatically excluded.

## Documentation

| Doc | Description |
|-----|-------------|
| [INSTALL.md](INSTALL.md) | Installation options and prerequisites |
| [CONFIG.md](CONFIG.md) | Configuration file reference |
| [CLI.md](CLI.md) | Command and flag reference |
| [SEARCHING.md](SEARCHING.md) | How to write effective queries |
| [INTEGRATIONS.md](INTEGRATIONS.md) | MCP, Emacs, VSCode, HTTP API |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the internals work |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and help wanted |

## FAQ

**Q: How does this compare to GitHub Copilot?**  
A: Copilot generates code using cloud APIs. Giddyanne searches existing code completely offline. They solve different problems.

**Q: Why not just use grep?**  
A: Grep matches exact text. If you search for "auth" you won't find `verify_credentials()`. Semantic search finds code by meaning.

**Q: Does it work with large codebases?**  
A: Yes. Indexing is a one-time cost (with incremental updates). After that, searches are near-instant regardless of codebase size.

**Q: What about API keys?**  
A: None needed. Sentence-transformers runs locally on your CPU. No external services, no usage limits.

**Q: Will this slow down my machine?**  
A: Initial indexing uses CPU for embedding generation. After that, the search server is lightweight and only re-indexes changed files.

## License

MIT License - see [LICENSE.txt](LICENSE.txt)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and areas where we'd love help.
