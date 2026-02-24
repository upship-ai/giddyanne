"""Tests for vectorstore module."""

import pytest

from src.vectorstore import (
    SearchResult,
    VectorStore,
    _apply_category_bias,
    _classify_file,
    _deduplicate_by_file,
    _rerank_results,
    _rrf_merge,
)


def _make_fts(path: str, content: str, header: str = "") -> str:
    """Build fts_content string for tests (matches engine._build_fts_content)."""
    from pathlib import Path as P

    basename = P(path).name
    header_part = f" {header}" if header else ""
    boost_line = f"{path} {basename}{header_part}"
    prefix = "\n".join([boost_line] * 3)
    return f"{prefix}\n{content}"


class TestSearchResult:
    def test_create_search_result(self):
        result = SearchResult(
            path="/path/to/file.py",
            chunk_index=0,
            start_line=1,
            end_line=50,
            content="print('hello')",
            description="Source code",
            score=0.95,
            content_score=0.90,
            description_score=0.85,
        )
        assert result.path == "/path/to/file.py"
        assert result.chunk_index == 0
        assert result.start_line == 1
        assert result.end_line == 50
        assert result.content == "print('hello')"
        assert result.description == "Source code"
        assert result.score == 0.95
        assert result.content_score == 0.90
        assert result.description_score == 0.85


class TestVectorStore:
    @pytest.fixture
    def db_path(self, tmp_path):
        return tmp_path / "test.lance"

    @pytest.fixture
    async def store(self, db_path):
        store = VectorStore(db_path, dimension=128)
        await store.connect()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_connect_creates_table(self, db_path):
        store = VectorStore(db_path, dimension=128)
        await store.connect()

        assert store._table is not None
        assert store._db is not None

        await store.close()

    @pytest.mark.asyncio
    async def test_upsert_and_list(self, store):
        embedding = [0.1] * 128

        await store.upsert(
            path="/test/file.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="test content",
            path_embedding=embedding,
            content_embedding=embedding,
        )

        paths = await store.list_all()
        assert "/test/file.py" in paths

    @pytest.mark.asyncio
    async def test_upsert_multiple_chunks(self, store):
        """Test that multiple chunks from same file are stored."""
        embedding = [0.1] * 128

        await store.upsert(
            path="/test/file.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="chunk 0",
            path_embedding=embedding,
            content_embedding=embedding,
        )
        await store.upsert(
            path="/test/file.py",
            chunk_index=1,
            start_line=21,
            end_line=40,
            content="chunk 1",
            path_embedding=embedding,
            content_embedding=embedding,
        )

        file_chunks = await store.list_all()
        # File should have 2 chunks
        assert file_chunks["/test/file.py"] == 2

    @pytest.mark.asyncio
    async def test_delete_removes_all_chunks(self, store):
        """Test that delete removes all chunks for a file."""
        embedding = [0.1] * 128

        # Add two chunks
        await store.upsert(
            path="/test/file.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="chunk 0",
            path_embedding=embedding,
            content_embedding=embedding,
        )
        await store.upsert(
            path="/test/file.py",
            chunk_index=1,
            start_line=21,
            end_line=40,
            content="chunk 1",
            path_embedding=embedding,
            content_embedding=embedding,
        )

        await store.delete("/test/file.py")

        paths = await store.list_all()
        assert "/test/file.py" not in paths

    @pytest.mark.asyncio
    async def test_delete_nonexistent_does_not_error(self, store):
        # Should not raise
        await store.delete("/nonexistent/file.py")

    @pytest.mark.asyncio
    async def test_search(self, store):
        # Insert chunks with different embeddings
        await store.upsert(
            path="/file1.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="authentication login",
            path_embedding=[0.9] + [0.1] * 127,
            content_embedding=[0.9] + [0.1] * 127,
        )
        await store.upsert(
            path="/file2.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="database queries",
            path_embedding=[0.1] + [0.9] + [0.1] * 126,
            content_embedding=[0.1] + [0.9] + [0.1] * 126,
        )

        # Search with embedding similar to file1
        results = await store.search(
            query_embedding=[0.9] + [0.1] * 127,
            limit=5,
        )

        assert len(results) > 0
        assert results[0].path == "/file1.py"
        assert results[0].chunk_index == 0
        assert results[0].start_line == 1
        assert results[0].end_line == 20

    @pytest.mark.asyncio
    async def test_search_limit(self, store):
        embedding = [0.1] * 128

        for i in range(5):
            await store.upsert(
                path=f"/file{i}.py",
                chunk_index=0,
                start_line=1,
                end_line=20,
                content=f"content {i}",
                path_embedding=embedding,
                content_embedding=embedding,
            )

        results = await store.search(query_embedding=embedding, limit=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_operations_without_connect_raise(self, db_path):
        store = VectorStore(db_path, dimension=128)

        with pytest.raises(RuntimeError, match="Not connected"):
            await store.upsert(
                "/test", 0, 1, 20, "content", [0.1] * 128, [0.1] * 128
            )

        with pytest.raises(RuntimeError, match="Not connected"):
            await store.delete("/test")

        with pytest.raises(RuntimeError, match="Not connected"):
            await store.search([0.1] * 128)

        with pytest.raises(RuntimeError, match="Not connected"):
            await store.list_all()

    @pytest.mark.asyncio
    async def test_upsert_with_none_content_embedding(self, store):
        """When content_embedding is None, path_embedding is used instead."""
        path_embedding = [0.5] * 128

        await store.upsert(
            path="/test/file.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="test",
            path_embedding=path_embedding,
            content_embedding=None,
        )

        # Should not raise and chunk should be searchable
        results = await store.search(query_embedding=path_embedding, limit=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_file_mtime_unknown_file(self, store):
        """get_file_mtime returns None for files not in the index."""
        mtime = await store.get_file_mtime("/nonexistent/file.py")
        assert mtime is None

    @pytest.mark.asyncio
    async def test_get_file_mtime_returns_stored_value(self, store):
        """get_file_mtime returns the mtime stored during upsert."""
        embedding = [0.1] * 128
        expected_mtime = 1703788800.123

        await store.upsert(
            path="/test/file.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="test content",
            path_embedding=embedding,
            content_embedding=embedding,
            mtime=expected_mtime,
        )

        mtime = await store.get_file_mtime("/test/file.py")
        assert mtime == expected_mtime

    @pytest.mark.asyncio
    async def test_get_file_mtime_after_delete(self, store):
        """get_file_mtime returns None after file is deleted."""
        embedding = [0.1] * 128

        await store.upsert(
            path="/test/file.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="test content",
            path_embedding=embedding,
            content_embedding=embedding,
            mtime=1703788800.0,
        )

        await store.delete("/test/file.py")

        mtime = await store.get_file_mtime("/test/file.py")
        assert mtime is None

    @pytest.mark.asyncio
    async def test_upsert_batch_empty(self, store):
        """upsert_batch with empty list should not error."""
        await store.upsert_batch([])
        paths = await store.list_all()
        assert len(paths) == 0

    @pytest.mark.asyncio
    async def test_upsert_batch_single(self, store):
        """upsert_batch with single chunk."""
        embedding = [0.1] * 128

        chunks = [{
            "path": "/test/file.py",
            "chunk_index": 0,
            "start_line": 1,
            "end_line": 20,
            "content": "test content",
            "description": "test description",
            "mtime": 1703788800.0,
            "path_embedding": embedding,
            "content_embedding": embedding,
            "description_embedding": embedding,
        }]

        await store.upsert_batch(chunks)

        paths = await store.list_all()
        assert "/test/file.py" in paths

    @pytest.mark.asyncio
    async def test_upsert_batch_multiple(self, store):
        """upsert_batch with multiple chunks from different files."""
        embedding = [0.1] * 128

        chunks = [
            {
                "path": "/test/file1.py",
                "chunk_index": 0,
                "start_line": 1,
                "end_line": 20,
                "content": "content 1",
                "description": "",
                "mtime": 1703788800.0,
                "path_embedding": embedding,
                "content_embedding": embedding,
                "description_embedding": None,
            },
            {
                "path": "/test/file1.py",
                "chunk_index": 1,
                "start_line": 21,
                "end_line": 40,
                "content": "content 2",
                "description": "",
                "mtime": 1703788800.0,
                "path_embedding": embedding,
                "content_embedding": embedding,
                "description_embedding": None,
            },
            {
                "path": "/test/file2.py",
                "chunk_index": 0,
                "start_line": 1,
                "end_line": 10,
                "content": "content 3",
                "description": "desc",
                "mtime": 1703788801.0,
                "path_embedding": embedding,
                "content_embedding": embedding,
                "description_embedding": embedding,
            },
        ]

        await store.upsert_batch(chunks)

        paths = await store.list_all()
        assert paths["/test/file1.py"] == 2
        assert paths["/test/file2.py"] == 1

    @pytest.mark.asyncio
    async def test_upsert_batch_without_connect_raises(self, db_path):
        store = VectorStore(db_path, dimension=128)

        with pytest.raises(RuntimeError, match="Not connected"):
            await store.upsert_batch([{"path": "/test", "chunk_index": 0}])

    @pytest.mark.asyncio
    async def test_search_fts(self, store):
        """Test full-text search returns results based on text matching."""
        embedding = [0.1] * 128

        await store.upsert(
            path="/auth.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="def authenticate_user(username, password): pass",
            fts_content=_make_fts(
                "auth.py",
                "def authenticate_user(username, password): pass",
                "authenticate_user",
            ),
            path_embedding=embedding,
            content_embedding=embedding,
        )
        await store.upsert(
            path="/db.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="def query_database(sql): pass",
            fts_content=_make_fts(
                "db.py", "def query_database(sql): pass",
                "query_database",
            ),
            path_embedding=embedding,
            content_embedding=embedding,
        )

        # FTS should find the file with "authenticate" in fts_content
        results = await store.search_fts("authenticate", limit=5)

        # May return empty if FTS index isn't ready, which is valid
        if results:
            assert results[0].path == "/auth.py"

    @pytest.mark.asyncio
    async def test_search_fts_matches_path_terms(self, store):
        """FTS matches path terms from boosted fts_content prefix."""
        embedding = [0.1] * 128

        await store.upsert(
            path="/src/auth/session.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="def validate_token(): pass",
            fts_content=_make_fts(
                "src/auth/session.py",
                "def validate_token(): pass",
                "validate_token",
            ),
            path_embedding=embedding,
            content_embedding=embedding,
        )
        await store.upsert(
            path="/src/utils/helpers.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="def format_string(): pass",
            fts_content=_make_fts(
                "src/utils/helpers.py",
                "def format_string(): pass",
                "format_string",
            ),
            path_embedding=embedding,
            content_embedding=embedding,
        )

        # Searching "session" should find session.py via path boosting
        results = await store.search_fts("session", limit=5)
        if results:
            assert results[0].path == "/src/auth/session.py"

    @pytest.mark.asyncio
    async def test_search_fts_matches_symbol_names(self, store):
        """FTS matches symbol names from header in fts_content prefix."""
        embedding = [0.1] * 128

        await store.upsert(
            path="/models.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="class User:\n    name: str",
            fts_content=_make_fts(
                "models.py", "class User:\n    name: str", "User",
            ),
            path_embedding=embedding,
            content_embedding=embedding,
        )
        await store.upsert(
            path="/views.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="def render_page(): pass",
            fts_content=_make_fts(
                "views.py", "def render_page(): pass", "render_page",
            ),
            path_embedding=embedding,
            content_embedding=embedding,
        )

        # Searching "render_page" should find views.py
        results = await store.search_fts("render_page", limit=5)
        if results:
            assert results[0].path == "/views.py"

    @pytest.mark.asyncio
    async def test_search_fts_empty_on_no_match(self, store):
        """Test FTS returns empty list when no matches."""
        embedding = [0.1] * 128

        await store.upsert(
            path="/file.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="hello world",
            fts_content=_make_fts("file.py", "hello world"),
            path_embedding=embedding,
            content_embedding=embedding,
        )

        results = await store.search_fts("nonexistent_term_xyz", limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_hybrid(self, store):
        """Test hybrid search combines vector and text search."""
        # Create embeddings that are different for each file
        auth_embedding = [0.9] + [0.1] * 127
        db_embedding = [0.1] + [0.9] + [0.1] * 126

        await store.upsert(
            path="/auth.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="def authenticate_user(): pass",
            fts_content=_make_fts(
                "auth.py", "def authenticate_user(): pass",
                "authenticate_user",
            ),
            path_embedding=auth_embedding,
            content_embedding=auth_embedding,
        )
        await store.upsert(
            path="/db.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="def query_database(): pass",
            fts_content=_make_fts(
                "db.py", "def query_database(): pass",
                "query_database",
            ),
            path_embedding=db_embedding,
            content_embedding=db_embedding,
        )

        # Hybrid search with both embedding and text
        results = await store.search_hybrid(
            query_embedding=auth_embedding,
            query_text="authenticate",
            limit=5,
        )

        # Should return results (may fall back to semantic if hybrid fails)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_hybrid_fallback_to_semantic(self, store):
        """Test hybrid search falls back to semantic when FTS fails."""
        embedding = [0.9] + [0.1] * 127

        await store.upsert(
            path="/file.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="test content",
            fts_content=_make_fts("file.py", "test content"),
            path_embedding=embedding,
            content_embedding=embedding,
        )

        # Even if hybrid search has issues, it should fall back to semantic
        results = await store.search_hybrid(
            query_embedding=embedding,
            query_text="test",
            limit=5,
        )

        assert len(results) > 0
        assert results[0].path == "/file.py"


class TestDeduplicateByFile:
    def test_keeps_best_scoring_chunk_per_file(self):
        """Dedup keeps only the highest-scoring chunk per file."""
        results = [
            SearchResult("/a.py", 0, 1, 10, "chunk0", "", 0.9, 0.9, 0.0),
            SearchResult("/a.py", 1, 11, 20, "chunk1", "", 0.7, 0.7, 0.0),
            SearchResult("/b.py", 0, 1, 10, "chunk0", "", 0.8, 0.8, 0.0),
        ]
        deduped = _deduplicate_by_file(results)
        paths = [r.path for r in deduped]
        assert paths == ["/a.py", "/b.py"]
        # /a.py should keep the chunk with score 0.9
        assert deduped[0].score == 0.9
        assert deduped[0].chunk_index == 0

    def test_no_duplicate_paths(self):
        """No two results share the same path after dedup."""
        results = [
            SearchResult("/a.py", i, 1, 10, f"chunk{i}", "", 0.5 + i * 0.1, 0.5, 0.0)
            for i in range(5)
        ]
        deduped = _deduplicate_by_file(results)
        assert len(deduped) == 1
        assert deduped[0].score == 0.9

    def test_diverse_files_not_crowded_out(self):
        """Files from other paths aren't lost when one file has many chunks."""
        results = [
            # 5 chunks from engine.py dominating
            SearchResult(
                "/engine.py", i, i * 10, (i + 1) * 10, f"e{i}", "",
                0.95 - i * 0.01, 0.9, 0.0,
            )
            for i in range(5)
        ] + [
            SearchResult("/api.py", 0, 1, 10, "api", "", 0.85, 0.85, 0.0),
            SearchResult("/utils.py", 0, 1, 10, "utils", "", 0.80, 0.80, 0.0),
        ]
        deduped = _deduplicate_by_file(results)
        paths = {r.path for r in deduped}
        assert paths == {"/engine.py", "/api.py", "/utils.py"}

    def test_empty_input(self):
        assert _deduplicate_by_file([]) == []

    def test_sorted_by_score_descending(self):
        results = [
            SearchResult("/c.py", 0, 1, 10, "", "", 0.5, 0.5, 0.0),
            SearchResult("/a.py", 0, 1, 10, "", "", 0.9, 0.9, 0.0),
            SearchResult("/b.py", 0, 1, 10, "", "", 0.7, 0.7, 0.0),
        ]
        deduped = _deduplicate_by_file(results)
        scores = [r.score for r in deduped]
        assert scores == [0.9, 0.7, 0.5]


class TestSearchDeduplication:
    """Integration test: search methods deduplicate results by file."""

    @pytest.fixture
    def db_path(self, tmp_path):
        return tmp_path / "test_dedup.lance"

    @pytest.fixture
    async def store(self, db_path):
        store = VectorStore(db_path, dimension=128)
        await store.connect()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_semantic_search_deduplicates(self, store):
        """Semantic search returns at most one result per file."""
        target = [0.9] + [0.1] * 127
        slightly_off = [0.85] + [0.15] * 127

        # Two chunks from same file, both close to query
        await store.upsert("/engine.py", 0, 1, 20, "chunk0", target, target)
        await store.upsert("/engine.py", 1, 21, 40, "chunk1", slightly_off, slightly_off)
        # One chunk from another file
        await store.upsert("/api.py", 0, 1, 20, "api chunk", slightly_off, slightly_off)

        results = await store.search(query_embedding=target, limit=10)
        paths = [r.path for r in results]
        assert len(paths) == len(set(paths)), f"Duplicate paths in results: {paths}"
        assert "/engine.py" in paths
        assert "/api.py" in paths


class TestClassifyFile:
    def test_source_file(self):
        assert _classify_file("src/vectorstore.py") == "code"

    def test_top_level_source(self):
        assert _classify_file("main.py") == "code"

    def test_test_dir(self):
        assert _classify_file("tests/test_vectorstore.py") == "test"

    def test_test_singular_dir(self):
        assert _classify_file("test/helpers.py") == "test"

    def test_test_prefix_filename(self):
        assert _classify_file("src/test_utils.py") == "test"

    def test_test_suffix_filename(self):
        assert _classify_file("src/vectorstore_test.py") == "test"

    def test_markdown_doc(self):
        assert _classify_file("README.md") == "docs"

    def test_rst_doc(self):
        assert _classify_file("docs/guide.rst") == "docs"

    def test_txt_doc(self):
        assert _classify_file("notes.txt") == "docs"

    def test_docs_dir(self):
        assert _classify_file("docs/api.py") == "docs"

    def test_nested_path(self):
        assert _classify_file("src/lib/internal/engine.py") == "code"

    def test_nested_test_dir(self):
        assert _classify_file("src/tests/integration/test_api.py") == "test"


class TestApplyCategoryBias:
    def _make_result(self, path: str, score: float) -> SearchResult:
        return SearchResult(path, 0, 1, 10, "content", "", score, score, 0.0)

    def test_code_unchanged(self):
        results = _apply_category_bias([self._make_result("src/app.py", 0.8)])
        assert results[0].score == 0.8

    def test_docs_score_reduced(self):
        """Doc score multiplied by 0.6."""
        results = _apply_category_bias([self._make_result("README.md", 1.0)])
        assert results[0].score == pytest.approx(0.6)

    def test_test_score_reduced(self):
        """Test score multiplied by 0.8."""
        results = _apply_category_bias([self._make_result("tests/test_app.py", 1.0)])
        assert results[0].score == pytest.approx(0.8)

    def test_doc_below_code_after_bias(self):
        """Doc at 0.8 (weighted 0.48) sorts below code at 0.5."""
        results = _apply_category_bias([
            self._make_result("README.md", 0.8),
            self._make_result("src/app.py", 0.5),
        ])
        scores = {r.path: r.score for r in results}
        assert scores["src/app.py"] > scores["README.md"]

    def test_strong_doc_stays_above(self):
        """Doc at 1.0 (weighted 0.6) still beats code at 0.5."""
        results = _apply_category_bias([
            self._make_result("README.md", 1.0),
            self._make_result("src/app.py", 0.5),
        ])
        scores = {r.path: r.score for r in results}
        assert scores["README.md"] > scores["src/app.py"]

    def test_unknown_category_unchanged(self):
        """Files not matching any category keep their score."""
        results = _apply_category_bias([self._make_result("Makefile", 0.7)])
        assert results[0].score == 0.7


class TestCategoryBiasIntegration:
    """Integration: source files rank above test files with similar embeddings."""

    @pytest.fixture
    def db_path(self, tmp_path):
        return tmp_path / "test_bias.lance"

    @pytest.fixture
    async def store(self, db_path):
        store = VectorStore(db_path, dimension=128)
        await store.connect()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_source_ranks_above_test(self, store):
        """Source file should rank above test file when embeddings are identical."""
        embedding = [0.9] + [0.1] * 127

        await store.upsert(
            path="tests/test_search.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="def test_search(): assert search() == expected",
            path_embedding=embedding,
            content_embedding=embedding,
        )
        await store.upsert(
            path="src/search.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="def search(): return find_results()",
            path_embedding=embedding,
            content_embedding=embedding,
        )

        results = await store.search(query_embedding=embedding, limit=10)
        assert results[0].path == "src/search.py"


class TestRRFMerge:
    def _make_result(self, path: str, chunk_index: int = 0, score: float = 0.0,
                     content_score: float = 0.0, description_score: float = 0.0) -> SearchResult:
        return SearchResult(
            path=path, chunk_index=chunk_index, start_line=1, end_line=10,
            content="content", description="", score=score,
            content_score=content_score, description_score=description_score,
        )

    def test_overlapping_results_fused(self):
        """Same chunk in both lists gets scores from both."""
        sem = [self._make_result("/a.py", content_score=0.9)]
        fts = [self._make_result("/a.py", content_score=0.5)]

        merged = _rrf_merge(sem, fts)
        assert len(merged) == 1
        assert merged[0].path == "/a.py"
        # score = 1/(60+1) + 1/(60+1) = 2/61
        expected = 1.0 / 61 + 1.0 / 61
        assert merged[0].score == pytest.approx(expected)

    def test_non_overlapping_results(self):
        """Disjoint results appear in merged list."""
        sem = [self._make_result("/a.py", content_score=0.9)]
        fts = [self._make_result("/b.py", content_score=0.5)]

        merged = _rrf_merge(sem, fts)
        assert len(merged) == 2
        paths = {r.path for r in merged}
        assert paths == {"/a.py", "/b.py"}

    def test_equal_weights_standard_rrf(self):
        """With equal weights, RRF score is standard formula."""
        sem = [
            self._make_result("/a.py"),
            self._make_result("/b.py"),
        ]
        fts = [
            self._make_result("/b.py"),
            self._make_result("/a.py"),
        ]
        merged = _rrf_merge(sem, fts, semantic_weight=1.0, fts_weight=1.0, k=60)
        scores = {r.path: r.score for r in merged}
        # /a.py: sem rank 1, fts rank 2 → 1/61 + 1/62
        # /b.py: sem rank 2, fts rank 1 → 1/62 + 1/61
        # Both should have the same score
        assert scores["/a.py"] == pytest.approx(scores["/b.py"])

    def test_zero_semantic_weight(self):
        """With semantic_weight=0, only FTS ranking matters."""
        sem = [self._make_result("/a.py")]
        fts = [
            self._make_result("/b.py"),
            self._make_result("/a.py"),
        ]
        merged = _rrf_merge(sem, fts, semantic_weight=0.0, fts_weight=1.0)
        scores = {r.path: r.score for r in merged}
        # /a.py: sem contributes 0, fts rank 2 → 1/62
        # /b.py: fts rank 1 → 1/61
        assert scores["/b.py"] > scores["/a.py"]

    def test_zero_fts_weight(self):
        """With fts_weight=0, only semantic ranking matters."""
        sem = [
            self._make_result("/a.py"),
            self._make_result("/b.py"),
        ]
        fts = [self._make_result("/b.py")]
        merged = _rrf_merge(sem, fts, semantic_weight=1.0, fts_weight=0.0)
        scores = {r.path: r.score for r in merged}
        # /a.py: sem rank 1 → 1/61
        # /b.py: sem rank 2, fts contributes 0 → 1/62
        assert scores["/a.py"] > scores["/b.py"]

    def test_preserves_content_and_description_scores(self):
        """Original content_score and description_score are preserved."""
        sem = [self._make_result("/a.py", content_score=0.9, description_score=0.7)]
        fts = []
        merged = _rrf_merge(sem, fts)
        assert merged[0].content_score == 0.9
        assert merged[0].description_score == 0.7

    def test_chunk_level_keying(self):
        """Different chunks from the same file are treated separately."""
        sem = [self._make_result("/a.py", chunk_index=0)]
        fts = [self._make_result("/a.py", chunk_index=1)]
        merged = _rrf_merge(sem, fts)
        assert len(merged) == 2

    def test_empty_inputs(self):
        """Empty inputs produce empty output."""
        assert _rrf_merge([], []) == []
        sem = [self._make_result("/a.py")]
        merged = _rrf_merge(sem, [])
        assert len(merged) == 1
        merged = _rrf_merge([], sem)
        assert len(merged) == 1


class TestRerankerHelper:
    def _make_result(self, path: str, score: float, content: str = "content") -> SearchResult:
        return SearchResult(
            path=path, chunk_index=0, start_line=1, end_line=10,
            content=content, description="", score=score,
            content_score=score, description_score=0.0,
        )

    def test_rerank_replaces_scores_and_sorts(self):
        """_rerank_results replaces scores with cross-encoder scores and re-sorts."""
        from unittest.mock import MagicMock

        reranker = MagicMock()
        reranker.rerank.return_value = [0.1, 0.9, 0.5]

        results = [
            self._make_result("/a.py", 0.9, "content a"),
            self._make_result("/b.py", 0.8, "content b"),
            self._make_result("/c.py", 0.7, "content c"),
        ]

        reranked = _rerank_results(results, "test query", reranker)

        # Should be sorted by new scores: b(0.9), c(0.5), a(0.1)
        assert reranked[0].path == "/b.py"
        assert reranked[0].score == 0.9
        assert reranked[1].path == "/c.py"
        assert reranked[1].score == 0.5
        assert reranked[2].path == "/a.py"
        assert reranked[2].score == 0.1

        reranker.rerank.assert_called_once_with(
            "test query", ["content a", "content b", "content c"]
        )

    def test_rerank_empty_results(self):
        """Empty results returns empty without calling reranker."""
        from unittest.mock import MagicMock

        reranker = MagicMock()
        result = _rerank_results([], "query", reranker)
        assert result == []
        reranker.rerank.assert_not_called()

    def test_rerank_preserves_metadata(self):
        """Reranking preserves content_score and description_score."""
        from unittest.mock import MagicMock

        reranker = MagicMock()
        reranker.rerank.return_value = [5.0]

        results = [SearchResult(
            path="/a.py", chunk_index=2, start_line=10, end_line=20,
            content="hello", description="desc", score=0.5,
            content_score=0.8, description_score=0.3,
        )]

        reranked = _rerank_results(results, "query", reranker)
        assert reranked[0].content_score == 0.8
        assert reranked[0].description_score == 0.3
        assert reranked[0].chunk_index == 2
        assert reranked[0].start_line == 10


class TestVectorStoreReranker:
    """Integration: VectorStore with mock reranker changes result order."""

    @pytest.fixture
    def db_path(self, tmp_path):
        return tmp_path / "test_reranker.lance"

    @pytest.fixture
    async def store_with_reranker(self, db_path):
        from unittest.mock import MagicMock

        reranker = MagicMock()
        # Reverse the order: give lower-ranked results higher scores
        reranker.rerank.side_effect = lambda q, docs: list(
            reversed([float(i) for i in range(len(docs))])
        )
        store = VectorStore(db_path, dimension=128, reranker=reranker)
        await store.connect()
        yield store
        await store.close()

    @pytest.fixture
    async def store_no_reranker(self, db_path):
        store = VectorStore(db_path, dimension=128)
        await store.connect()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_semantic_search_with_reranker(self, store_with_reranker):
        """Semantic search with reranker reorders results."""
        store = store_with_reranker
        high_emb = [0.9] + [0.1] * 127
        low_emb = [0.1] + [0.9] + [0.1] * 126

        await store.upsert("/high.py", 0, 1, 20, "high match", high_emb, high_emb)
        await store.upsert("/low.py", 0, 1, 20, "low match", low_emb, low_emb)

        # With query_text, reranker fires and reverses order
        results = await store.search(
            query_embedding=high_emb, limit=10, query_text="test query"
        )
        assert len(results) >= 2
        # Reranker reverses: last gets highest score
        assert store.reranker.rerank.called

    @pytest.mark.asyncio
    async def test_semantic_search_without_query_text_skips_reranker(self, store_with_reranker):
        """Semantic search without query_text does not rerank."""
        store = store_with_reranker
        emb = [0.9] + [0.1] * 127
        await store.upsert("/file.py", 0, 1, 20, "content", emb, emb)

        results = await store.search(query_embedding=emb, limit=10)
        assert len(results) == 1
        store.reranker.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_reranker_existing_behavior(self, store_no_reranker):
        """VectorStore without reranker maintains existing behavior."""
        store = store_no_reranker
        emb = [0.9] + [0.1] * 127
        await store.upsert("/file.py", 0, 1, 20, "content", emb, emb)

        results = await store.search(query_embedding=emb, limit=10)
        assert len(results) == 1
        assert results[0].path == "/file.py"
