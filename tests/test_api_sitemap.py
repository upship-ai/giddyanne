"""Tests for the /sitemap API endpoint."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.api import create_app
from src.project_config import PathConfig, ProjectConfig, ProjectSettings


@pytest.fixture
def mock_vector_store():
    store = AsyncMock()
    store.list_all.return_value = {
        "src/api.py": 3,
        "src/vectorstore.py": 5,
        "cmd/giddy/main.go": 8,
    }
    store.get_all_mtimes.return_value = {
        "src/api.py": 1700000000.0,
        "src/vectorstore.py": 1700000100.0,
        "cmd/giddy/main.go": 1700000200.0,
    }
    return store


@pytest.fixture
def mock_embedding_service():
    return AsyncMock()


@pytest.fixture
def project_config():
    return ProjectConfig(
        paths=[
            PathConfig(path="src/", description="Core source code"),
            PathConfig(path="cmd/", description="Go CLI client"),
        ],
        settings=ProjectSettings(),
    )


@pytest.fixture
def app_with_config(mock_vector_store, mock_embedding_service, project_config):
    return create_app(
        mock_vector_store, mock_embedding_service,
        project_config=project_config,
    )


@pytest.fixture
def app_without_config(mock_vector_store, mock_embedding_service):
    return create_app(mock_vector_store, mock_embedding_service)


class TestSitemapWithConfig:
    @pytest.mark.asyncio
    async def test_default_includes_paths(self, app_with_config):
        async with AsyncClient(
            transport=ASGITransport(app=app_with_config),
            base_url="http://test",
        ) as client:
            resp = await client.get("/sitemap")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert sorted(data["files"]) == [
            "cmd/giddy/main.go",
            "src/api.py",
            "src/vectorstore.py",
        ]
        assert data["paths"] == [
            {"path": "src/", "description": "Core source code"},
            {"path": "cmd/", "description": "Go CLI client"},
        ]

    @pytest.mark.asyncio
    async def test_verbose_includes_paths(self, app_with_config):
        async with AsyncClient(
            transport=ASGITransport(app=app_with_config),
            base_url="http://test",
        ) as client:
            resp = await client.get("/sitemap?verbose=true")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert data["paths"] == [
            {"path": "src/", "description": "Core source code"},
            {"path": "cmd/", "description": "Go CLI client"},
        ]
        # Verbose files have chunks and mtime
        files_by_path = {f["path"]: f for f in data["files"]}
        assert files_by_path["src/api.py"]["chunks"] == 3
        assert files_by_path["src/api.py"]["mtime"] == 1700000000.0
        assert files_by_path["cmd/giddy/main.go"]["chunks"] == 8


class TestSitemapWithoutConfig:
    @pytest.mark.asyncio
    async def test_default_no_paths_field(self, app_without_config):
        async with AsyncClient(
            transport=ASGITransport(app=app_without_config),
            base_url="http://test",
        ) as client:
            resp = await client.get("/sitemap")

        data = resp.json()
        assert "paths" not in data
        assert data["count"] == 3
        assert len(data["files"]) == 3

    @pytest.mark.asyncio
    async def test_verbose_no_paths_field(self, app_without_config):
        async with AsyncClient(
            transport=ASGITransport(app=app_without_config),
            base_url="http://test",
        ) as client:
            resp = await client.get("/sitemap?verbose=true")

        data = resp.json()
        assert "paths" not in data
        assert data["count"] == 3


class TestSitemapEmptyIndex:
    @pytest.mark.asyncio
    async def test_empty_index(self, mock_embedding_service, project_config):
        store = AsyncMock()
        store.list_all.return_value = {}
        app = create_app(
            store, mock_embedding_service,
            project_config=project_config,
        )
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/sitemap")

        data = resp.json()
        assert data["count"] == 0
        assert data["files"] == []
        # Paths still present even when no files indexed
        assert len(data["paths"]) == 2
