"""Project configuration loader for .giddyanne.yaml files."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pathspec
import yaml

from src.languages import is_supported_file

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when project configuration is invalid."""

    pass


class FileFilter:
    """Determines which files should be indexed and watched.

    Files are included if they:
    1. Have a supported language extension (from languages.py)
    2. Are under a configured path
    3. Are not ignored by .gitignore or additional ignore_patterns

    Explicitly listed files bypass ignore patterns but still need supported extensions.
    """

    def __init__(self, root_path: Path, config: "ProjectConfig"):
        self.root_path = root_path
        self.config = config
        self._explicit_files: set[str] = set()
        self._path_prefixes: list[str] = []

        # Combine .gitignore patterns with config ignore_patterns
        ignore_patterns = self._load_gitignore_patterns()
        ignore_patterns.extend(config.settings.ignore_patterns)
        self._ignore_matcher = pathspec.PathSpec.from_lines(
            "gitwildmatch", ignore_patterns
        )
        self._build_path_info()

    def _load_gitignore_patterns(self) -> list[str]:
        """Load patterns from .gitignore file if it exists."""
        gitignore_path = self.root_path / ".gitignore"
        if not gitignore_path.exists():
            return []

        try:
            patterns = []
            with open(gitignore_path) as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        patterns.append(line)
            return patterns
        except OSError as e:
            logger.warning(f"Failed to read .gitignore: {e}")
            return []

    def _build_path_info(self) -> None:
        """Build lookup structures for explicit files and path prefixes."""
        for path_config in self.config.paths:
            target = self.root_path / path_config.path
            if target.is_file():
                self._explicit_files.add(path_config.path)
            self._path_prefixes.append(path_config.path)

    def _matches_ignore(self, rel_path: str) -> bool:
        """Check if path matches any ignore pattern (.gitignore + config patterns)."""
        return self._ignore_matcher.match_file(rel_path)

    def should_include(self, path: Path) -> bool:
        """Check if a file should be indexed/watched."""
        if path.is_dir():
            return False
        if not path.exists():
            return False

        # Only index files with supported language extensions
        if not is_supported_file(str(path)):
            return False

        try:
            rel_path = str(path.relative_to(self.root_path))
        except ValueError:
            return False  # Outside project root

        # Check file size
        try:
            if path.stat().st_size > self.config.settings.max_file_size:
                return False
        except OSError:
            return False

        # Check ignore patterns (.gitignore + config patterns)
        if self._matches_ignore(rel_path):
            return False

        # Check if under a configured path (or explicitly listed)
        if rel_path in self._explicit_files:
            return True

        for prefix in self._path_prefixes:
            if rel_path.startswith(prefix) or prefix == ".":
                return True

        return False

    def matches_path(self, path: Path) -> bool:
        """Check if path matches configured paths (for watcher, ignores existence checks).

        This is used by the watcher for deleted events where the file no longer exists.
        """
        if path.is_dir():
            return False

        # Only track files with supported language extensions
        if not is_supported_file(str(path)):
            return False

        try:
            rel_path = str(path.relative_to(self.root_path))
        except ValueError:
            return False  # Outside project root

        # Check ignore patterns (.gitignore + config patterns)
        if self._matches_ignore(rel_path):
            return False

        # Check if under a configured path (or explicitly listed)
        if rel_path in self._explicit_files:
            return True

        for prefix in self._path_prefixes:
            if rel_path.startswith(prefix) or prefix == ".":
                return True

        return False

    def get_description(self, path: Path) -> str:
        """Get cumulative description from all matching parent paths."""
        try:
            rel_path = str(path.relative_to(self.root_path))
        except ValueError:
            return ""

        # Collect all matching descriptions
        matches = []
        for path_config in self.config.paths:
            if rel_path.startswith(path_config.path) and path_config.description:
                matches.append((len(path_config.path), path_config.description))

        if not matches:
            return ""

        # Sort by path length (general → specific) and join
        matches.sort(key=lambda x: x[0])
        return ", ".join(desc for _, desc in matches)


@dataclass
class PathConfig:
    path: str
    description: str


DEFAULT_IGNORE_PATTERNS: list[str] = []
DEFAULT_MAX_FILE_SIZE = 1_000_000

# Chunking defaults
DEFAULT_MIN_CHUNK_LINES = 10
DEFAULT_MAX_CHUNK_LINES = 50
DEFAULT_OVERLAP_LINES = 5

# Server defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_DB_PATH = ".giddyanne/vectors.lance"

# Embedding defaults
DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"


@dataclass
class ProjectSettings:
    # File handling
    max_file_size: int = DEFAULT_MAX_FILE_SIZE
    ignore_patterns: list[str] = field(default_factory=lambda: DEFAULT_IGNORE_PATTERNS.copy())

    # Chunking
    min_chunk_lines: int = DEFAULT_MIN_CHUNK_LINES
    max_chunk_lines: int = DEFAULT_MAX_CHUNK_LINES
    overlap_lines: int = DEFAULT_OVERLAP_LINES

    # Server
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    db_path: str = DEFAULT_DB_PATH

    # Embedding model
    local_model: str = DEFAULT_LOCAL_MODEL


@dataclass
class ProjectConfig:
    paths: list[PathConfig]
    settings: ProjectSettings

    @classmethod
    def load(cls, config_path: Path) -> "ProjectConfig":
        """Load config from .giddyanne.yaml file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {config_path}: {e}") from e

        if not isinstance(data, dict):
            raise ConfigError(
                f"Config must be a YAML mapping, got {type(data).__name__}"
            )

        try:
            paths = [
                PathConfig(path=p["path"], description=p.get("description", ""))
                for p in data.get("paths", [])
            ]
        except (TypeError, KeyError) as e:
            raise ConfigError(f"Invalid paths in config: {e}") from e

        settings_data = data.get("settings", {})
        if not isinstance(settings_data, dict):
            raise ConfigError("settings must be a mapping")

        settings = ProjectSettings(
            # File handling
            max_file_size=settings_data.get("max_file_size", DEFAULT_MAX_FILE_SIZE),
            ignore_patterns=settings_data.get("ignore_patterns", DEFAULT_IGNORE_PATTERNS),
            # Chunking
            min_chunk_lines=settings_data.get("min_chunk_lines", DEFAULT_MIN_CHUNK_LINES),
            max_chunk_lines=settings_data.get("max_chunk_lines", DEFAULT_MAX_CHUNK_LINES),
            overlap_lines=settings_data.get("overlap_lines", DEFAULT_OVERLAP_LINES),
            # Server
            host=settings_data.get("host", DEFAULT_HOST),
            port=settings_data.get("port", DEFAULT_PORT),
            db_path=settings_data.get("db_path", DEFAULT_DB_PATH),
            # Embedding model
            local_model=settings_data.get("local_model", DEFAULT_LOCAL_MODEL),
        )

        return cls(paths=paths, settings=settings)

    @classmethod
    def default(cls, root_path: Path) -> "ProjectConfig":
        """Create a default config that indexes common source directories."""
        common_dirs = ["src", "lib", "app", "tests", "docs"]
        paths = []

        for dir_name in common_dirs:
            dir_path = root_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                paths.append(PathConfig(path=dir_name, description=f"{dir_name} directory"))

        # If no common dirs found, index the root
        if not paths:
            paths.append(PathConfig(path=".", description="Project root"))

        return cls(paths=paths, settings=ProjectSettings())
