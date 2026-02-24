"""Tests for watcher module."""

import asyncio
from pathlib import Path

import pytest

from src.project_config import FileFilter, PathConfig, ProjectConfig, ProjectSettings
from src.watcher import AsyncEventHandler, EventType, FileEvent, FileWatcher


def create_file_filter(root_path: Path, paths: list[str] | None = None) -> FileFilter:
    """Create a FileFilter for testing."""
    if paths is None:
        paths = ["."]  # Watch everything by default
    path_configs = [PathConfig(path=p, description="") for p in paths]
    config = ProjectConfig(paths=path_configs, settings=ProjectSettings())
    return FileFilter(root_path, config)


class TestEventType:
    def test_event_types_exist(self):
        assert EventType.CREATED.value == "created"
        assert EventType.MODIFIED.value == "modified"
        assert EventType.DELETED.value == "deleted"
        assert EventType.MOVED.value == "moved"


class TestFileEvent:
    def test_create_file_event(self):
        event = FileEvent(
            event_type=EventType.CREATED,
            path=Path("/test/file.py"),
            is_directory=False,
        )
        assert event.event_type == EventType.CREATED
        assert event.path == Path("/test/file.py")
        assert event.is_directory is False
        assert event.dest_path is None

    def test_move_event_with_dest(self):
        event = FileEvent(
            event_type=EventType.MOVED,
            path=Path("/test/old.py"),
            is_directory=False,
            dest_path=Path("/test/new.py"),
        )
        assert event.dest_path == Path("/test/new.py")


class TestAsyncEventHandler:
    def test_should_process_files_in_paths(self, tmp_path):
        # Create test structure
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        test_file = src_dir / "main.py"
        test_file.write_text("test")

        file_filter = create_file_filter(tmp_path, ["src/"])
        handler = AsyncEventHandler(
            callback=lambda e: None,
            loop=asyncio.new_event_loop(),
            file_filter=file_filter,
        )

        assert handler._should_process(str(test_file))

    def test_processes_files_outside_configured_paths(self, tmp_path):
        """Files outside configured paths are still processed (paths are for descriptions only)."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        other_file = other_dir / "file.py"
        other_file.write_text("test")

        file_filter = create_file_filter(tmp_path, ["src/"])
        handler = AsyncEventHandler(
            callback=lambda e: None,
            loop=asyncio.new_event_loop(),
            file_filter=file_filter,
        )

        assert handler._should_process(str(other_file))

    def test_respects_ignore_patterns(self, tmp_path):
        # Create test structure
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        cache_dir = src_dir / "__pycache__"
        cache_dir.mkdir()
        cache_file = cache_dir / "module.pyc"
        cache_file.write_text("test")

        path_configs = [PathConfig(path="src/", description="")]
        settings = ProjectSettings(ignore_patterns=["__pycache__"])
        config = ProjectConfig(paths=path_configs, settings=settings)
        file_filter = FileFilter(tmp_path, config)

        handler = AsyncEventHandler(
            callback=lambda e: None,
            loop=asyncio.new_event_loop(),
            file_filter=file_filter,
        )

        assert not handler._should_process(str(cache_file))


class TestFileWatcher:
    @pytest.fixture
    def callback(self):
        async def cb(event: FileEvent) -> None:
            pass
        return cb

    def test_create_watcher(self, tmp_path, callback):
        file_filter = create_file_filter(tmp_path)
        watcher = FileWatcher(file_filter, callback)
        assert watcher.root_path == tmp_path
        assert not watcher.is_running

    @pytest.mark.asyncio
    async def test_start_stop(self, tmp_path, callback):
        file_filter = create_file_filter(tmp_path)
        watcher = FileWatcher(file_filter, callback)

        await watcher.start()
        assert watcher.is_running

        await watcher.stop()
        assert not watcher.is_running

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self, tmp_path, callback):
        file_filter = create_file_filter(tmp_path)
        watcher = FileWatcher(file_filter, callback)
        # Should not raise
        await watcher.stop()

    @pytest.mark.asyncio
    async def test_detects_file_creation(self, tmp_path):
        events = []

        async def collect_events(event: FileEvent) -> None:
            events.append(event)

        # Create src directory first (watching src/)
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        file_filter = create_file_filter(tmp_path, ["src/"])
        watcher = FileWatcher(file_filter, collect_events)
        await watcher.start()

        # Give watcher time to initialize
        await asyncio.sleep(0.1)

        # Create a file in the watched directory (must have supported extension)
        test_file = src_dir / "new_file.py"
        test_file.write_text("# hello")

        # Wait for event to be processed
        await asyncio.sleep(0.5)

        await watcher.stop()

        # Should have at least one CREATED event
        created_events = [e for e in events if e.event_type == EventType.CREATED]
        assert len(created_events) >= 1

    @pytest.mark.asyncio
    async def test_ignores_files_outside_paths(self, tmp_path):
        events = []

        async def collect_events(event: FileEvent) -> None:
            events.append(event)

        # Create directories
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        # Only watch src/
        file_filter = create_file_filter(tmp_path, ["src/"])
        watcher = FileWatcher(file_filter, collect_events)
        await watcher.start()

        await asyncio.sleep(0.1)

        # Create file in other/ (not watched)
        (other_dir / "ignored.txt").write_text("test")

        await asyncio.sleep(0.3)

        await watcher.stop()

        # Should not have events for other/ files
        other_events = [e for e in events if "other" in str(e.path)]
        assert len(other_events) == 0
