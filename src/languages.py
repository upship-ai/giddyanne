"""Language definitions for syntax-aware code chunking.

To add a new language, simply add an entry to the LANGUAGES dict.
No other code changes required.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LanguageSpec:
    """Definition for a supported language.

    Attributes:
        name: Human-readable language name.
        extensions: File extensions including the dot (e.g., [".py", ".pyw"]).
        separators: Ordered list of separators, most specific first.
            The chunker tries each separator in order, falling back to
            the next if chunks are still too large.
    """

    name: str
    extensions: tuple[str, ...]
    separators: tuple[str, ...]


# Language registry - add new languages here
LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        name="Python",
        extensions=(".py", ".pyw"),
        separators=(
            "\nclass ",
            "\ndef ",
            "\n    def ",
            "\n        def ",
            "\n\n",
        ),
    ),
    "go": LanguageSpec(
        name="Go",
        extensions=(".go",),
        separators=(
            "\nfunc ",
            "\ntype ",
            "\nvar ",
            "\nconst ",
            "\n\n",
        ),
    ),
    "javascript": LanguageSpec(
        name="JavaScript",
        extensions=(".js", ".jsx", ".mjs"),
        separators=(
            "\nfunction ",
            "\nclass ",
            "\nconst ",
            "\nlet ",
            "\nexport ",
            "\n\n",
        ),
    ),
    "typescript": LanguageSpec(
        name="TypeScript",
        extensions=(".ts", ".tsx"),
        separators=(
            "\nfunction ",
            "\nclass ",
            "\ninterface ",
            "\ntype ",
            "\nenum ",
            "\nconst ",
            "\nlet ",
            "\nexport ",
            "\n\n",
        ),
    ),
    "rust": LanguageSpec(
        name="Rust",
        extensions=(".rs",),
        separators=(
            "\nfn ",
            "\nimpl ",
            "\nstruct ",
            "\nenum ",
            "\nmod ",
            "\npub fn ",
            "\npub struct ",
            "\npub enum ",
            "\npub mod ",
            "\n\n",
        ),
    ),
    "sql": LanguageSpec(
        name="SQL",
        extensions=(".sql",),
        separators=(
            "\nCREATE ",
            "\nALTER ",
            "\nDROP ",
            "\nSELECT ",
            "\nINSERT ",
            "\nUPDATE ",
            "\nDELETE ",
            "\nWITH ",
            "\nBEGIN",
            # Case-insensitive variants
            "\ncreate ",
            "\nalter ",
            "\ndrop ",
            "\nselect ",
            "\ninsert ",
            "\nupdate ",
            "\ndelete ",
            "\nwith ",
            "\nbegin",
            "\n\n",
        ),
    ),
    "java": LanguageSpec(
        name="Java",
        extensions=(".java",),
        separators=(
            "\npublic class ",
            "\nclass ",
            "\ninterface ",
            "\nenum ",
            "\n    public ",
            "\n    private ",
            "\n    protected ",
            "\n\n",
        ),
    ),
    "c": LanguageSpec(
        name="C",
        extensions=(".c", ".h"),
        separators=(
            "\nstatic ",
            "\nstruct ",
            "\ntypedef ",
            "\n#define ",
            "\n#include ",
            "\nvoid ",
            "\nint ",
            "\nchar ",
            "\n\n",
        ),
    ),
    "cpp": LanguageSpec(
        name="C++",
        extensions=(".cpp", ".hpp", ".cc", ".cxx", ".hxx", ".hh"),
        separators=(
            "\nclass ",
            "\nnamespace ",
            "\ntemplate",
            "\nstatic ",
            "\nvoid ",
            "\nint ",
            "\nstruct ",
            "\n#include ",
            "\n\n",
        ),
    ),
    "csharp": LanguageSpec(
        name="C#",
        extensions=(".cs",),
        separators=(
            "\nnamespace ",
            "\npublic class ",
            "\nclass ",
            "\ninterface ",
            "\nenum ",
            "\nstruct ",
            "\n    public ",
            "\n    private ",
            "\n    protected ",
            "\n\n",
        ),
    ),
    "ruby": LanguageSpec(
        name="Ruby",
        extensions=(".rb", ".rake"),
        separators=(
            "\nclass ",
            "\nmodule ",
            "\ndef ",
            "\n  def ",
            "\n    def ",
            "\n\n",
        ),
    ),
    "html": LanguageSpec(
        name="HTML",
        extensions=(".html", ".htm"),
        separators=(
            "\n<template",
            "\n<script",
            "\n<style",
            "\n<header",
            "\n<footer",
            "\n<main",
            "\n<section",
            "\n<article",
            "\n<nav",
            "\n<div",
            "\n\n",
        ),
    ),
    "css": LanguageSpec(
        name="CSS",
        extensions=(".css",),
        separators=(
            "\n@media ",
            "\n@keyframes ",
            "\n@import ",
            "\n.",
            "\n#",
            "\n\n",
        ),
    ),
    "php": LanguageSpec(
        name="PHP",
        extensions=(".php", ".phtml"),
        separators=(
            "\nclass ",
            "\ninterface ",
            "\ntrait ",
            "\nfunction ",
            "\n    public ",
            "\n    private ",
            "\n    protected ",
            "\n\n",
        ),
    ),
    "kotlin": LanguageSpec(
        name="Kotlin",
        extensions=(".kt", ".kts"),
        separators=(
            "\nclass ",
            "\ninterface ",
            "\nobject ",
            "\nfun ",
            "\nval ",
            "\nvar ",
            "\n    fun ",
            "\n\n",
        ),
    ),
    "swift": LanguageSpec(
        name="Swift",
        extensions=(".swift",),
        separators=(
            "\nclass ",
            "\nstruct ",
            "\nenum ",
            "\nprotocol ",
            "\nextension ",
            "\nfunc ",
            "\n    func ",
            "\n\n",
        ),
    ),
    "scala": LanguageSpec(
        name="Scala",
        extensions=(".scala", ".sc"),
        separators=(
            "\nclass ",
            "\nobject ",
            "\ntrait ",
            "\ndef ",
            "\nval ",
            "\nvar ",
            "\n  def ",
            "\n\n",
        ),
    ),
    "shell": LanguageSpec(
        name="Shell",
        extensions=(".sh", ".bash", ".zsh"),
        separators=(
            "\nfunction ",
            "\nif ",
            "\nfor ",
            "\nwhile ",
            "\ncase ",
            "\n\n",
        ),
    ),
    "yaml": LanguageSpec(
        name="YAML",
        extensions=(".yaml", ".yml"),
        separators=(
            "\n\n",
        ),
    ),
    "json": LanguageSpec(
        name="JSON",
        extensions=(".json",),
        separators=(
            "\n\n",
        ),
    ),
    "toml": LanguageSpec(
        name="TOML",
        extensions=(".toml",),
        separators=(
            "\n[",
            "\n\n",
        ),
    ),
    "markdown": LanguageSpec(
        name="Markdown",
        extensions=(".md", ".markdown"),
        separators=(
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
        ),
    ),
    "objectivec": LanguageSpec(
        name="Objective-C",
        extensions=(".m", ".mm"),
        separators=(
            "\n@interface ",
            "\n@implementation ",
            "\n@protocol ",
            "\n- (",
            "\n+ (",
            "\n\n",
        ),
    ),
    "lua": LanguageSpec(
        name="Lua",
        extensions=(".lua",),
        separators=(
            "\nfunction ",
            "\nlocal function ",
            "\nlocal ",
            "\n\n",
        ),
    ),
    "dart": LanguageSpec(
        name="Dart",
        extensions=(".dart",),
        separators=(
            "\nclass ",
            "\nenum ",
            "\nvoid ",
            "\nFuture",
            "\n  void ",
            "\n  Future",
            "\n\n",
        ),
    ),
    "r": LanguageSpec(
        name="R",
        extensions=(".r", ".R"),
        separators=(
            "\n\n",
        ),
    ),
    "elixir": LanguageSpec(
        name="Elixir",
        extensions=(".ex", ".exs"),
        separators=(
            "\ndefmodule ",
            "\ndef ",
            "\ndefp ",
            "\n  def ",
            "\n  defp ",
            "\n\n",
        ),
    ),
    "zig": LanguageSpec(
        name="Zig",
        extensions=(".zig",),
        separators=(
            "\npub fn ",
            "\nfn ",
            "\nconst ",
            "\nvar ",
            "\nstruct ",
            "\n\n",
        ),
    ),
}

# Build extension lookup for fast detection
_EXT_TO_LANGUAGE: dict[str, LanguageSpec] = {}
for spec in LANGUAGES.values():
    for ext in spec.extensions:
        _EXT_TO_LANGUAGE[ext] = spec


def detect_language(path: str) -> LanguageSpec | None:
    """Detect language from file path.

    Args:
        path: File path (absolute or relative).

    Returns:
        LanguageSpec if language is recognized, None otherwise.
    """
    ext = Path(path).suffix.lower()
    return _EXT_TO_LANGUAGE.get(ext)


def supported_languages() -> list[str]:
    """Return list of supported language names for documentation."""
    return sorted(spec.name for spec in LANGUAGES.values())


def supported_extensions() -> set[str]:
    """Return set of all supported file extensions."""
    return set(_EXT_TO_LANGUAGE.keys())


def is_supported_file(path: str) -> bool:
    """Check if a file has a supported extension."""
    ext = Path(path).suffix.lower()
    return ext in _EXT_TO_LANGUAGE
