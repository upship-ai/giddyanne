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
        extensions: File extensions including the dot (e.g., (".py", ".pyw")).
        ts_language: Tree-sitter grammar name (e.g. "python"). None if no grammar available.
        node_types: AST node types to split on (e.g. ("function_definition", "class_definition")).
    """

    name: str
    extensions: tuple[str, ...]
    ts_language: str | None
    node_types: tuple[str, ...] = ()


# Language registry - add new languages here
LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        name="Python",
        extensions=(".py", ".pyw"),
        ts_language="python",
        node_types=("class_definition", "function_definition", "decorated_definition"),
    ),
    "go": LanguageSpec(
        name="Go",
        extensions=(".go",),
        ts_language="go",
        node_types=("function_declaration", "method_declaration", "type_declaration"),
    ),
    "javascript": LanguageSpec(
        name="JavaScript",
        extensions=(".js", ".jsx", ".mjs"),
        ts_language="javascript",
        node_types=(
            "function_declaration", "class_declaration",
            "lexical_declaration", "export_statement",
        ),
    ),
    "typescript": LanguageSpec(
        name="TypeScript",
        extensions=(".ts", ".tsx"),
        ts_language="typescript",
        node_types=(
            "function_declaration",
            "class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
            "export_statement",
        ),
    ),
    "rust": LanguageSpec(
        name="Rust",
        extensions=(".rs",),
        ts_language="rust",
        node_types=(
            "function_item", "impl_item", "struct_item",
            "enum_item", "mod_item", "trait_item",
        ),
    ),
    "sql": LanguageSpec(
        name="SQL",
        extensions=(".sql",),
        ts_language="sql",
        node_types=("statement",),
    ),
    "java": LanguageSpec(
        name="Java",
        extensions=(".java",),
        ts_language="java",
        node_types=("class_declaration", "interface_declaration", "enum_declaration"),
    ),
    "c": LanguageSpec(
        name="C",
        extensions=(".c", ".h"),
        ts_language="c",
        node_types=(
            "function_definition", "struct_specifier",
            "type_definition", "preproc_include",
        ),
    ),
    "cpp": LanguageSpec(
        name="C++",
        extensions=(".cpp", ".hpp", ".cc", ".cxx", ".hxx", ".hh"),
        ts_language="cpp",
        node_types=(
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
        ),
    ),
    "csharp": LanguageSpec(
        name="C#",
        extensions=(".cs",),
        ts_language=None,  # tree-sitter-language-pack uses "c_sharp" but it's unreliable
        node_types=(),
    ),
    "ruby": LanguageSpec(
        name="Ruby",
        extensions=(".rb", ".rake"),
        ts_language="ruby",
        node_types=("class", "module", "method", "singleton_method"),
    ),
    "html": LanguageSpec(
        name="HTML",
        extensions=(".html", ".htm"),
        ts_language="html",
        node_types=("element", "script_element", "style_element"),
    ),
    "css": LanguageSpec(
        name="CSS",
        extensions=(".css",),
        ts_language="css",
        node_types=("rule_set", "media_statement", "keyframes_statement", "import_statement"),
    ),
    "php": LanguageSpec(
        name="PHP",
        extensions=(".php", ".phtml"),
        ts_language="php",
        node_types=(
            "class_declaration", "interface_declaration",
            "trait_declaration", "function_definition",
        ),
    ),
    "kotlin": LanguageSpec(
        name="Kotlin",
        extensions=(".kt", ".kts"),
        ts_language="kotlin",
        node_types=("class_declaration", "object_declaration", "function_declaration"),
    ),
    "swift": LanguageSpec(
        name="Swift",
        extensions=(".swift",),
        ts_language="swift",
        node_types=("class_declaration", "protocol_declaration", "function_declaration"),
    ),
    "scala": LanguageSpec(
        name="Scala",
        extensions=(".scala", ".sc"),
        ts_language="scala",
        node_types=(
            "class_definition", "object_definition",
            "trait_definition", "function_definition",
        ),
    ),
    "shell": LanguageSpec(
        name="Shell",
        extensions=(".sh", ".bash", ".zsh"),
        ts_language="bash",
        node_types=("function_definition",),
    ),
    "yaml": LanguageSpec(
        name="YAML",
        extensions=(".yaml", ".yml"),
        ts_language=None,
    ),
    "json": LanguageSpec(
        name="JSON",
        extensions=(".json",),
        ts_language=None,
    ),
    "toml": LanguageSpec(
        name="TOML",
        extensions=(".toml",),
        ts_language="toml",
        node_types=("table",),
    ),
    "markdown": LanguageSpec(
        name="Markdown",
        extensions=(".md", ".markdown"),
        ts_language="markdown",
        node_types=("section",),
    ),
    "objectivec": LanguageSpec(
        name="Objective-C",
        extensions=(".m", ".mm"),
        ts_language="objc",
        node_types=("class_interface", "class_implementation", "protocol_declaration"),
    ),
    "lua": LanguageSpec(
        name="Lua",
        extensions=(".lua",),
        ts_language="lua",
        node_types=("function_declaration",),
    ),
    "dart": LanguageSpec(
        name="Dart",
        extensions=(".dart",),
        ts_language="dart",
        node_types=("class_definition", "enum_declaration", "function_signature"),
    ),
    "r": LanguageSpec(
        name="R",
        extensions=(".r", ".R"),
        ts_language="r",
        node_types=(),
    ),
    "elixir": LanguageSpec(
        name="Elixir",
        extensions=(".ex", ".exs"),
        ts_language="elixir",
        node_types=("call",),
    ),
    "zig": LanguageSpec(
        name="Zig",
        extensions=(".zig",),
        ts_language="zig",
        node_types=("Decl",),
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
