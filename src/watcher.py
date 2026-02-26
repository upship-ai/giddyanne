"""File system watcher module."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

if TYPE_CHECKING:
    from src.project_config import FileFilter

from src.project_config import ALWAYS_IGNORE_DIRS

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


def _is_ignored_dir(name: str) -> bool:
    """Check if a directory name should never be watched."""
    return name in ALWAYS_IGNORE_DIRS


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
        observer: Observer | None = None,
    ):
        self.callback = callback
        self.loop = loop
        self.file_filter = file_filter
        self._observer = observer
        # Debounce state: path -> (timer, event)
        self._pending: dict[str, tuple[threading.Timer, FileEvent]] = {}
        self._lock = threading.Lock()

    def _should_process(self, path: str) -> bool:
        """Check if file event should be processed."""
        # Fast reject: skip events in always-ignored directories
        # (avoids Path construction and pathspec matching overhead)
        parts = path.split("/")
        for part in parts:
            if part in ALWAYS_IGNORE_DIRS:
                return False
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
                    pending_count = len(self._pending)
                try:
                    if pending_count > 50:
                        logger.warning(
                            f"Watcher _pending backlog: {pending_count} entries"
                        )
                    future = asyncio.run_coroutine_threadsafe(
                        self.callback(event), self.loop
                    )
                    # Check for exceptions in the callback (non-blocking)
                    future.add_done_callback(
                        lambda f: (
                            logger.error(
                                f"Watcher callback failed for {path_key}: {f.exception()}"
                            )
                            if f.exception()
                            else None
                        )
                    )
                except Exception:
                    logger.exception(
                        f"Watcher failed to dispatch event for {path_key}"
                    )

            timer = threading.Timer(DEBOUNCE_DELAY, fire)
            self._pending[path_key] = (timer, event)
            timer.start()

    def on_created(self, event: FileSystemEvent) -> None:
        # Auto-watch new directories (unless ignored)
        if isinstance(event, DirCreatedEvent):
            dirname = os.path.basename(event.src_path)
            if not _is_ignored_dir(dirname) and self._observer:
                try:
                    self._observer.schedule(
                        self, event.src_path, recursive=False,
                    )
                except Exception:
                    pass  # Directory may have been deleted already
            return

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
        if event.is_directory:
            return  # Directory modifications are not useful
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
        if isinstance(event, DirDeletedEvent):
            return  # Directory deletions handled by watchdog internally
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
        # Auto-watch moved-to directories
        if isinstance(event, DirMovedEvent):
            if hasattr(event, "dest_path"):
                dirname = os.path.basename(event.dest_path)
                if not _is_ignored_dir(dirname) and self._observer:
                    try:
                        self._observer.schedule(
                            self, event.dest_path, recursive=False,
                        )
                    except Exception:
                        pass
            return

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
    """Watches configured paths for file system changes.

    Uses non-recursive watches to avoid registering inotify watches on
    always-ignored directories (node_modules, .git, etc.). Walks the
    directory tree at startup, skipping ignored dirs, and dynamically
    adds watches for new directories via DirCreatedEvent.
    """

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
        self._observer = Observer()
        handler = AsyncEventHandler(
            self.callback, loop, self.file_filter, self._observer,
        )

        root = str(self.file_filter.root_path)
        watch_count = self._schedule_filtered_watches(handler, root)
        logger.info(f"Watcher: registered {watch_count} directory watches")

        self._observer.start()

    def _schedule_filtered_watches(
        self, handler: AsyncEventHandler, root: str,
    ) -> int:
        """Walk tree, schedule non-recursive watches, skip ignored dirs.

        Returns the number of watches registered.
        """
        count = 0
        for dirpath, dirnames, _ in os.walk(root):
            # Prune ignored dirs in-place so os.walk doesn't descend
            dirnames[:] = [
                d for d in dirnames
                if not _is_ignored_dir(d)
            ]
            self._observer.schedule(handler, dirpath, recursive=False)
            count += 1
        return count

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
