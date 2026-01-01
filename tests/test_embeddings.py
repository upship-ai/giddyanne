"""Tests for embeddings module."""

import pytest

from src.embeddings import (
    ChunkEmbedding,
    EmbeddingProvider,
    EmbeddingService,
)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock provider for testing without loading real models."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return "mock-model"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Return fake embeddings of the right dimension
        return [[0.1] * self._dimension for _ in texts]

    def dimension(self) -> int:
        return self._dimension


class TestMockProvider:
    @pytest.mark.asyncio
    async def test_embed_returns_correct_shape(self):
        provider = MockEmbeddingProvider(dimension=128)
        texts = ["hello", "world"]
        embeddings = await provider.embed(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 128
        assert len(embeddings[1]) == 128

    def test_dimension(self):
        provider = MockEmbeddingProvider(dimension=256)
        assert provider.dimension() == 256


class TestEmbeddingService:
    @pytest.mark.asyncio
    async def test_embed_file_with_content(self):
        provider = MockEmbeddingProvider(dimension=128)
        service = EmbeddingService(provider)

        result = await service.embed_file("/path/to/file.py", "print('hello')")

        assert result["path"] == "/path/to/file.py"
        assert result["content"] == "print('hello')"
        assert result["description"] == ""
        assert len(result["path_embedding"]) == 128
        assert len(result["content_embedding"]) == 128
        assert result["description_embedding"] is None

    @pytest.mark.asyncio
    async def test_embed_file_empty_content(self):
        provider = MockEmbeddingProvider(dimension=128)
        service = EmbeddingService(provider)

        result = await service.embed_file("/path/to/file.py", "")

        assert result["path"] == "/path/to/file.py"
        assert result["content"] == ""
        assert len(result["path_embedding"]) == 128
        assert result["content_embedding"] is None
        assert result["description_embedding"] is None

    @pytest.mark.asyncio
    async def test_embed_file_with_description(self):
        provider = MockEmbeddingProvider(dimension=128)
        service = EmbeddingService(provider)

        result = await service.embed_file(
            "/src/admin.py", "class AdminPanel:", "Admin control panel code"
        )

        assert result["path"] == "/src/admin.py"
        assert result["content"] == "class AdminPanel:"
        assert result["description"] == "Admin control panel code"
        assert len(result["path_embedding"]) == 128
        assert len(result["content_embedding"]) == 128
        assert len(result["description_embedding"]) == 128

    @pytest.mark.asyncio
    async def test_embed_query(self):
        provider = MockEmbeddingProvider(dimension=128)
        service = EmbeddingService(provider)

        embedding, cache_hit = await service.embed_query("find authentication code")

        assert len(embedding) == 128
        assert cache_hit is False

    @pytest.mark.asyncio
    async def test_embed_chunks_batch_empty(self):
        provider = MockEmbeddingProvider(dimension=128)
        service = EmbeddingService(provider)

        results = await service.embed_chunks_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_embed_chunks_batch_single(self):
        provider = MockEmbeddingProvider(dimension=128)
        service = EmbeddingService(provider)

        chunks = [
            ("/path/file.py", 0, 1, 20, "content here", "description", 1234567890.0)
        ]
        results = await service.embed_chunks_batch(chunks)

        assert len(results) == 1
        assert isinstance(results[0], ChunkEmbedding)
        assert results[0].path == "/path/file.py"
        assert results[0].chunk_index == 0
        assert results[0].start_line == 1
        assert results[0].end_line == 20
        assert results[0].content == "content here"
        assert results[0].description == "description"
        assert results[0].mtime == 1234567890.0
        assert len(results[0].path_embedding) == 128
        assert len(results[0].content_embedding) == 128
        assert len(results[0].description_embedding) == 128

    @pytest.mark.asyncio
    async def test_embed_chunks_batch_multiple(self):
        provider = MockEmbeddingProvider(dimension=128)
        service = EmbeddingService(provider)

        chunks = [
            ("/path/file1.py", 0, 1, 20, "content 1", "desc 1", 1234567890.0),
            ("/path/file1.py", 1, 21, 40, "content 2", "desc 2", 1234567890.0),
            ("/path/file2.py", 0, 1, 10, "content 3", "", 1234567891.0),
        ]
        results = await service.embed_chunks_batch(chunks)

        assert len(results) == 3
        assert results[0].path == "/path/file1.py"
        assert results[0].chunk_index == 0
        assert results[1].path == "/path/file1.py"
        assert results[1].chunk_index == 1
        assert results[2].path == "/path/file2.py"
        assert results[2].chunk_index == 0
        # Third chunk has no description
        assert results[2].description_embedding is None

    @pytest.mark.asyncio
    async def test_embed_chunks_batch_empty_content(self):
        """Empty content should result in None content_embedding."""
        provider = MockEmbeddingProvider(dimension=128)
        service = EmbeddingService(provider)

        chunks = [
            ("/path/empty.py", 0, 1, 1, "", "has description", 1234567890.0),
        ]
        results = await service.embed_chunks_batch(chunks)

        assert len(results) == 1
        assert results[0].content_embedding is None
        assert results[0].description_embedding is not None
