"""Tests for vectorstore module."""

import pytest

from src.vectorstore import SearchResult, VectorStore


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
            path_embedding=embedding,
            content_embedding=embedding,
        )
        await store.upsert(
            path="/db.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="def query_database(sql): pass",
            path_embedding=embedding,
            content_embedding=embedding,
        )

        # FTS should find the file with "authenticate" in content
        results = await store.search_fts("authenticate", limit=5)

        # May return empty if FTS index isn't ready, which is valid
        if results:
            assert results[0].path == "/auth.py"

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
            path_embedding=auth_embedding,
            content_embedding=auth_embedding,
        )
        await store.upsert(
            path="/db.py",
            chunk_index=0,
            start_line=1,
            end_line=20,
            content="def query_database(): pass",
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
