"""Tests for chunker module."""

from src.chunker import Chunk, chunk_content
from src.languages import detect_language


class TestChunk:
    def test_create_chunk(self):
        chunk = Chunk(start_line=1, end_line=10, content="hello\nworld")
        assert chunk.start_line == 1
        assert chunk.end_line == 10
        assert chunk.content == "hello\nworld"


class TestChunkContent:
    def test_empty_content_returns_empty_list(self):
        assert chunk_content("") == []
        assert chunk_content("   ") == []
        assert chunk_content("\n\n\n") == []

    def test_small_content_single_chunk(self):
        content = "line1\nline2\nline3"
        chunks = chunk_content(content, min_lines=1, max_lines=50)
        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 3

    def test_splits_on_blank_lines(self):
        content = """def foo():
    pass

def bar():
    pass"""
        chunks = chunk_content(content, min_lines=1, max_lines=50)
        # Should split into two chunks at the blank line
        assert len(chunks) == 2
        assert "foo" in chunks[0].content
        assert "bar" in chunks[1].content

    def test_merges_small_chunks(self):
        content = """a

b

c

d"""
        # With min_lines=5, small single-line segments should merge
        chunks = chunk_content(content, min_lines=5, max_lines=50)
        # All segments are tiny, should merge into fewer chunks
        assert len(chunks) < 4

    def test_splits_large_chunks(self):
        # Create content with 100 lines, no blank lines
        lines = [f"line {i}" for i in range(100)]
        content = "\n".join(lines)

        chunks = chunk_content(content, min_lines=10, max_lines=30, overlap=5)

        # Should be split into multiple chunks
        assert len(chunks) > 1

        # Each chunk should be at most 30 lines
        for chunk in chunks:
            chunk_lines = len(chunk.content.split("\n"))
            assert chunk_lines <= 30

    def test_overlap_between_large_chunks(self):
        # Create content with 60 lines
        lines = [f"line {i}" for i in range(60)]
        content = "\n".join(lines)

        chunks = chunk_content(content, min_lines=10, max_lines=30, overlap=10)

        # With 60 lines, max=30, overlap=10, we should get overlapping chunks
        assert len(chunks) >= 2

        # Check that chunks have correct line number progression
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Later chunks should start after earlier ones
                assert chunk.start_line > chunks[i - 1].start_line

    def test_preserves_line_numbers(self):
        content = """line 1
line 2

line 4
line 5"""
        chunks = chunk_content(content, min_lines=1, max_lines=50)

        # First chunk should start at line 1
        assert chunks[0].start_line == 1

        # Second chunk should start after the blank line
        if len(chunks) > 1:
            assert chunks[1].start_line > chunks[0].end_line

    def test_handles_content_without_blank_lines(self):
        content = "line1\nline2\nline3\nline4\nline5"
        chunks = chunk_content(content, min_lines=2, max_lines=10)

        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 5

    def test_real_code_example(self):
        content = '''"""Module docstring."""

import os
import sys

def main():
    """Main function."""
    print("hello")
    return 0

class Foo:
    """A class."""

    def method(self):
        pass'''

        chunks = chunk_content(content, min_lines=5, max_lines=20)

        # Should produce reasonable chunks
        assert len(chunks) >= 1

        # All content should be captured
        all_content = "\n\n".join(c.content for c in chunks)
        assert "Module docstring" in all_content
        assert "main" in all_content
        assert "Foo" in all_content


class TestTreeSitterChunking:
    """Tests for tree-sitter AST-based chunking."""

    def test_python_splits_on_def(self):
        content = '''def foo():
    return 1

def bar():
    return 2

def baz():
    return 3'''
        language = detect_language("test.py")
        chunks = chunk_content(content, language=language, min_lines=1, max_lines=50)

        # Should split on function definitions
        assert len(chunks) >= 2
        # Each function should be in its own chunk or merged appropriately
        all_content = " ".join(c.content for c in chunks)
        assert "foo" in all_content
        assert "bar" in all_content
        assert "baz" in all_content

    def test_python_splits_on_class(self):
        content = '''class Foo:
    pass

class Bar:
    pass'''
        language = detect_language("test.py")
        chunks = chunk_content(content, language=language, min_lines=1, max_lines=50)

        # Should split on class definitions
        assert len(chunks) >= 2
        assert "class Foo" in chunks[0].content
        assert "class Bar" in chunks[1].content

    def test_python_decorated_functions(self):
        content = '''@decorator
def foo():
    return 1

@other_decorator
def bar():
    return 2'''
        language = detect_language("test.py")
        chunks = chunk_content(content, language=language, min_lines=1, max_lines=50)

        assert len(chunks) >= 2
        all_content = " ".join(c.content for c in chunks)
        assert "@decorator" in all_content
        assert "foo" in all_content
        assert "bar" in all_content

    def test_python_imports_merge_with_first_def(self):
        content = '''import os
import sys

def main():
    print("hello")'''
        language = detect_language("test.py")
        chunks = chunk_content(content, language=language, min_lines=1, max_lines=50)

        # Imports should be merged into the first chunk (either with main or standalone)
        all_content = " ".join(c.content for c in chunks)
        assert "import os" in all_content
        assert "main" in all_content

    def test_go_splits_on_func(self):
        content = '''package main

func main() {
    fmt.Println("hello")
}

func helper() int {
    return 42
}'''
        language = detect_language("main.go")
        chunks = chunk_content(content, language=language, min_lines=1, max_lines=50)

        assert len(chunks) >= 2
        all_content = " ".join(c.content for c in chunks)
        assert "func main" in all_content
        assert "func helper" in all_content

    def test_javascript_splits_on_function(self):
        content = '''function greet(name) {
    return `Hello, ${name}!`;
}

function farewell(name) {
    return `Goodbye, ${name}!`;
}'''
        language = detect_language("app.js")
        chunks = chunk_content(content, language=language, min_lines=1, max_lines=50)

        assert len(chunks) >= 2
        assert "greet" in chunks[0].content
        assert "farewell" in chunks[1].content

    def test_typescript_splits_on_interface(self):
        content = '''interface User {
    name: string;
    age: number;
}

interface Product {
    id: number;
    title: string;
}'''
        language = detect_language("types.ts")
        chunks = chunk_content(content, language=language, min_lines=1, max_lines=50)

        assert len(chunks) >= 2
        assert "interface User" in chunks[0].content
        assert "interface Product" in chunks[1].content

    def test_rust_splits_on_items(self):
        content = '''fn main() {
    println!("hello");
}

struct Foo {
    x: i32,
}

impl Foo {
    fn new() -> Self {
        Foo { x: 0 }
    }
}'''
        language = detect_language("main.rs")
        chunks = chunk_content(content, language=language, min_lines=1, max_lines=50)

        assert len(chunks) >= 2
        all_content = " ".join(c.content for c in chunks)
        assert "fn main" in all_content
        assert "struct Foo" in all_content
        assert "impl Foo" in all_content

    def test_sql_splits_on_statements(self):
        content = '''CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE products (
    id INT PRIMARY KEY,
    title VARCHAR(200)
);'''
        language = detect_language("schema.sql")
        chunks = chunk_content(content, language=language, min_lines=1, max_lines=50)

        assert len(chunks) >= 2
        assert "users" in chunks[0].content
        assert "products" in chunks[1].content

    def test_unknown_language_falls_back_to_blank_lines(self):
        content = '''section one
line 1
line 2

section two
line 3
line 4'''
        # .xyz is unknown, should fall back to blank-line splitting
        chunks = chunk_content(content, language=None, min_lines=1, max_lines=50)

        assert len(chunks) >= 2
        assert "section one" in chunks[0].content
        assert "section two" in chunks[1].content

    def test_language_aware_preserves_line_numbers(self):
        content = '''def first():
    pass

def second():
    pass'''
        language = detect_language("test.py")
        chunks = chunk_content(content, language=language, min_lines=1, max_lines=50)

        # First chunk should start at line 1
        assert chunks[0].start_line == 1
        # Second chunk should start after the first
        assert chunks[1].start_line > chunks[0].start_line

    def test_treesitter_line_numbers_accurate(self):
        content = '''def foo():
    return 1

def bar():
    return 2'''
        language = detect_language("test.py")
        chunks = chunk_content(content, language=language, min_lines=1, max_lines=50)

        # foo is on lines 1-2
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 2

        # bar is on lines 4-5
        assert chunks[1].start_line == 4
        assert chunks[1].end_line == 5

    def test_no_ts_language_falls_back_to_blank_lines(self):
        """Languages with ts_language=None should fall back to blank-line splitting."""
        content = '''key: value

other: data'''
        language = detect_language("config.yaml")
        chunks = chunk_content(content, language=language, min_lines=1, max_lines=50)

        assert len(chunks) >= 2
        assert "key: value" in chunks[0].content
        assert "other: data" in chunks[1].content
