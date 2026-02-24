"""File system watcher module."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

if TYPE_CHECKING:
    from src.project_config import FileFilter

logger = logging.getLogger(__name__)

# Debounce delay in seconds - wait this long after last event before processing
DEBOUNCE_DELAY = 0.1


class EventType(Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileEvent:
    event_type: EventType
    path: Path
    is_directory: bool
    dest_path: Path | None = None  # For move events


class FileEventCallback(Protocol):
    async def __call__(self, event: FileEvent) -> None: ...


class AsyncEventHandler(FileSystemEventHandler):
    """Bridges watchdog's sync events to async callbacks with debouncing.

    Debouncing prevents multiple rapid events on the same file from triggering
    multiple indexing operations. When multiple events arrive for the same path,
    only the last event is processed after a short delay.
    """

    def __init__(
        self,
        callback: FileEventCallback,
        loop: asyncio.AbstractEventLoop,
        file_filter: FileFilter,
    ):
        self.callback = callback
        self.loop = loop
        self.file_filter = file_filter
        # Debounce state: path -> (timer, event)
        self._pending: dict[str, tuple[threading.Timer, FileEvent]] = {}
        self._lock = threading.Lock()

    def _should_process(self, path: str) -> bool:
        """Check if file event should be processed."""
        return self.file_filter.matches_path(Path(path))

    def _schedule_callback(self, event: FileEvent) -> None:
        """Schedule event with debouncing - cancels previous pending event for same path.

        Event type precedence when coalescing:
        - DELETED always wins (file is gone)
        - CREATED preserved over MODIFIED (file is new)
        - MODIFIED is default
        """
        path_key = str(event.path)

        with self._lock:
            # Cancel any pending timer for this path
            if path_key in self._pending:
                old_timer, old_event = self._pending[path_key]
                old_timer.cancel()

                # Determine which event type to keep
                # DELETED always wins
                if event.event_type == EventType.DELETED:
                    pass  # Use new DELETED event
                elif old_event.event_type == EventType.DELETED:
                    event = old_event  # Keep old DELETED
                # CREATED over MODIFIED (preserve "file is new" semantic)
                elif old_event.event_type == EventType.CREATED:
                    event = FileEvent(
                        event_type=EventType.CREATED,
                        path=event.path,
                        is_directory=event.is_directory,
                        dest_path=event.dest_path,
                    )

            # Create new timer that fires callback after delay
            def fire():
                with self._lock:
                    if path_key in self._pending:
                        del self._pending[path_key]
                asyncio.run_coroutine_threadsafe(self.callback(event), self.loop)

            timer = threading.Timer(DEBOUNCE_DELAY, fire)
            self._pending[path_key] = (timer, event)
            timer.start()

    def on_created(self, event: FileSystemEvent) -> None:
        if not self._should_process(event.src_path):
            return
        self._schedule_callback(
            FileEvent(
                event_type=EventType.CREATED,
                path=Path(event.src_path),
                is_directory=event.is_directory,
            )
        )

    def on_modified(self, event: FileSystemEvent) -> None:
        if not self._should_process(event.src_path):
            return
        self._schedule_callback(
            FileEvent(
                event_type=EventType.MODIFIED,
                path=Path(event.src_path),
                is_directory=event.is_directory,
            )
        )

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not self._should_process(event.src_path):
            return
        self._schedule_callback(
            FileEvent(
                event_type=EventType.DELETED,
                path=Path(event.src_path),
                is_directory=event.is_directory,
            )
        )

    def on_moved(self, event: FileSystemEvent) -> None:
        if not self._should_process(event.src_path):
            return
        self._schedule_callback(
            FileEvent(
                event_type=EventType.MOVED,
                path=Path(event.src_path),
                is_directory=event.is_directory,
                dest_path=Path(event.dest_path) if hasattr(event, "dest_path") else None,
            )
        )


class FileWatcher:
    """Watches configured paths for file system changes."""

    def __init__(
        self,
        file_filter: FileFilter,
        callback: FileEventCallback,
    ):
        self.file_filter = file_filter
        self.callback = callback
        self._observer: Observer | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        handler = AsyncEventHandler(self.callback, loop, self.file_filter)

        self._observer = Observer()

        self._observer.schedule(handler, str(self.file_filter.root_path), recursive=True)

        self._observer.start()

    async def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    @property
    def is_running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()

    @property
    def root_path(self) -> Path:
        return self.file_filter.root_path
