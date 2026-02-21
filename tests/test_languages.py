"""Tests for language detection and registry."""

from src.languages import LANGUAGES, detect_language, supported_languages


class TestDetectLanguage:
    def test_detects_python(self):
        lang = detect_language("foo/bar.py")
        assert lang is not None
        assert lang.name == "Python"

    def test_detects_python_pyw(self):
        lang = detect_language("script.pyw")
        assert lang is not None
        assert lang.name == "Python"

    def test_detects_go(self):
        lang = detect_language("main.go")
        assert lang is not None
        assert lang.name == "Go"

    def test_detects_javascript(self):
        for ext in [".js", ".jsx", ".mjs"]:
            lang = detect_language(f"file{ext}")
            assert lang is not None
            assert lang.name == "JavaScript"

    def test_detects_typescript(self):
        for ext in [".ts", ".tsx"]:
            lang = detect_language(f"file{ext}")
            assert lang is not None
            assert lang.name == "TypeScript"

    def test_detects_rust(self):
        lang = detect_language("lib.rs")
        assert lang is not None
        assert lang.name == "Rust"

    def test_detects_sql(self):
        lang = detect_language("schema.sql")
        assert lang is not None
        assert lang.name == "SQL"

    def test_case_insensitive_extension(self):
        lang = detect_language("file.PY")
        assert lang is not None
        assert lang.name == "Python"

    def test_unknown_extension_returns_none(self):
        assert detect_language("file.xyz") is None
        assert detect_language("file.txt") is None
        assert detect_language("file.log") is None

    def test_no_extension_returns_none(self):
        assert detect_language("Makefile") is None
        assert detect_language("README") is None


class TestLanguageRegistry:
    def test_all_languages_have_extensions(self):
        for name, spec in LANGUAGES.items():
            assert len(spec.extensions) > 0, f"{name} has no extensions"

    def test_extensions_start_with_dot(self):
        for name, spec in LANGUAGES.items():
            for ext in spec.extensions:
                assert ext.startswith("."), f"{name} extension {ext} missing dot"

    def test_ts_languages_have_node_types(self):
        """Languages with tree-sitter grammars should have node_types defined."""
        for name, spec in LANGUAGES.items():
            if spec.ts_language and spec.node_types:
                assert len(spec.node_types) > 0, f"{name} has ts_language but no node_types"


class TestSupportedLanguages:
    def test_returns_sorted_names(self):
        names = supported_languages()
        assert names == sorted(names)

    def test_includes_expected_languages(self):
        names = supported_languages()
        assert "Python" in names
        assert "Go" in names
        assert "TypeScript" in names
        assert "SQL" in names


class TestSupportedExtensions:
    def test_returns_set(self):
        from src.languages import supported_extensions

        exts = supported_extensions()
        assert isinstance(exts, set)

    def test_includes_expected_extensions(self):
        from src.languages import supported_extensions

        exts = supported_extensions()
        assert ".py" in exts
        assert ".go" in exts
        assert ".ts" in exts
        assert ".sql" in exts


class TestIsSupportedFile:
    def test_supported_files(self):
        from src.languages import is_supported_file

        assert is_supported_file("main.py") is True
        assert is_supported_file("main.go") is True
        assert is_supported_file("app.tsx") is True
        assert is_supported_file("schema.sql") is True

    def test_unsupported_files(self):
        from src.languages import is_supported_file

        assert is_supported_file("readme.txt") is False
        assert is_supported_file("data.xml") is False
        assert is_supported_file("debug.log") is False
        assert is_supported_file("Makefile") is False

    def test_case_insensitive(self):
        from src.languages import is_supported_file

        assert is_supported_file("Main.PY") is True
        assert is_supported_file("app.GO") is True
