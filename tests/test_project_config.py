"""Tests for project_config module."""

import pytest

from src.project_config import (
    DEFAULT_IGNORE_PATTERNS,
    DEFAULT_MAX_FILE_SIZE,
    ConfigError,
    FileFilter,
    PathConfig,
    ProjectConfig,
    ProjectSettings,
)


class TestPathConfig:
    def test_create_path_config(self):
        pc = PathConfig(path="src/", description="Source code")
        assert pc.path == "src/"
        assert pc.description == "Source code"


class TestProjectSettings:
    def test_defaults(self):
        settings = ProjectSettings()
        assert settings.max_file_size == DEFAULT_MAX_FILE_SIZE
        assert settings.ignore_patterns == DEFAULT_IGNORE_PATTERNS

    def test_custom_settings(self):
        settings = ProjectSettings(
            max_file_size=50000,
            ignore_patterns=[".git"],
        )
        assert settings.max_file_size == 50000
        assert settings.ignore_patterns == [".git"]


class TestProjectConfig:
    def test_load_from_yaml(self, sample_yaml_config):
        config = ProjectConfig.load(sample_yaml_config)

        assert len(config.paths) == 2
        assert config.paths[0].path == "src/"
        assert config.paths[0].description == "Source code"
        assert config.paths[1].path == "main.py"

        assert config.settings.max_file_size == 100000
        assert ".venv" in config.settings.ignore_patterns

    def test_load_missing_file_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            ProjectConfig.load(missing)

    def test_load_minimal_yaml(self, tmp_path):
        """Test loading YAML with only paths, no settings."""
        config_path = tmp_path / ".giddyanne.yaml"
        config_path.write_text("""
paths:
  - path: src/
    description: Code
""")
        config = ProjectConfig.load(config_path)

        assert len(config.paths) == 1
        assert config.settings.ignore_patterns == DEFAULT_IGNORE_PATTERNS
        assert config.settings.max_file_size == DEFAULT_MAX_FILE_SIZE

    def test_default_finds_common_dirs(self, sample_project_dir):
        config = ProjectConfig.default(sample_project_dir)

        path_names = [p.path for p in config.paths]
        assert "src" in path_names
        assert "tests" in path_names
        assert ".git" not in path_names

    def test_default_empty_project_uses_root(self, tmp_path):
        """When no common directories exist, default to root."""
        config = ProjectConfig.default(tmp_path)

        assert len(config.paths) == 1
        assert config.paths[0].path == "."

    def test_path_without_description(self, tmp_path):
        """Test loading paths without descriptions."""
        config_path = tmp_path / ".giddyanne.yaml"
        config_path.write_text("""
paths:
  - path: src/
""")
        config = ProjectConfig.load(config_path)

        assert config.paths[0].path == "src/"
        assert config.paths[0].description == ""

    def test_load_invalid_yaml_syntax(self, tmp_path):
        """Test loading malformed YAML raises ConfigError."""
        config_path = tmp_path / ".giddyanne.yaml"
        config_path.write_text("""
paths:
  - path: src/
    description: "unclosed string
""")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            ProjectConfig.load(config_path)

    def test_load_yaml_not_mapping(self, tmp_path):
        """Test loading YAML that's not a mapping raises ConfigError."""
        config_path = tmp_path / ".giddyanne.yaml"
        config_path.write_text("- just\n- a\n- list\n")
        with pytest.raises(ConfigError, match="must be a YAML mapping"):
            ProjectConfig.load(config_path)

    def test_load_paths_missing_path_key(self, tmp_path):
        """Test paths without 'path' key raises ConfigError."""
        config_path = tmp_path / ".giddyanne.yaml"
        config_path.write_text("""
paths:
  - description: "no path key"
""")
        with pytest.raises(ConfigError, match="Invalid paths"):
            ProjectConfig.load(config_path)

    def test_load_settings_not_mapping(self, tmp_path):
        """Test settings as non-mapping raises ConfigError."""
        config_path = tmp_path / ".giddyanne.yaml"
        config_path.write_text("""
paths:
  - path: src/
settings: "not a mapping"
""")
        with pytest.raises(ConfigError, match="settings must be a mapping"):
            ProjectConfig.load(config_path)


class TestFileFilter:
    """Tests for FileFilter class."""

    def test_only_includes_supported_extensions(self, tmp_path):
        """Test that only files with supported language extensions are included."""
        # Create test files
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("# python")
        (src_dir / "app.go").write_text("// go")
        (src_dir / "readme.txt").write_text("readme")
        (src_dir / "data.xml").write_text("<data/>")

        config = ProjectConfig(
            paths=[PathConfig(path="src/", description="")],
            settings=ProjectSettings(),
        )
        file_filter = FileFilter(tmp_path, config)

        # Supported extensions should be included
        assert file_filter.should_include(src_dir / "main.py") is True
        assert file_filter.should_include(src_dir / "app.go") is True

        # Unsupported extensions should be excluded
        assert file_filter.should_include(src_dir / "readme.txt") is False
        assert file_filter.should_include(src_dir / "data.xml") is False

    def test_respects_gitignore(self, tmp_path):
        """Test that .gitignore patterns are respected."""
        # Create .gitignore
        (tmp_path / ".gitignore").write_text("*.generated.py\nbuild/\n")

        # Create test files
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("# python")
        (src_dir / "code.generated.py").write_text("# generated")

        build_dir = tmp_path / "build"
        build_dir.mkdir()
        (build_dir / "output.py").write_text("# build output")

        config = ProjectConfig(
            paths=[
                PathConfig(path="src/", description=""),
                PathConfig(path="build/", description=""),
            ],
            settings=ProjectSettings(),
        )
        file_filter = FileFilter(tmp_path, config)

        # Normal file should be included
        assert file_filter.should_include(src_dir / "main.py") is True

        # Generated file should be excluded by gitignore pattern
        assert file_filter.should_include(src_dir / "code.generated.py") is False

        # Build directory files should be excluded
        assert file_filter.should_include(build_dir / "output.py") is False

    def test_combines_gitignore_and_config_patterns(self, tmp_path):
        """Test that both .gitignore and config ignore_patterns are applied."""
        # Create .gitignore
        (tmp_path / ".gitignore").write_text("*.log\n")

        # Create test files
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("# python")
        (src_dir / "test_foo.py").write_text("# test")

        config = ProjectConfig(
            paths=[PathConfig(path="src/", description="")],
            settings=ProjectSettings(ignore_patterns=["test_*.py"]),
        )
        file_filter = FileFilter(tmp_path, config)

        # Normal file should be included
        assert file_filter.should_include(src_dir / "main.py") is True

        # File matching config ignore pattern should be excluded
        assert file_filter.should_include(src_dir / "test_foo.py") is False

    def test_get_description_single_match(self, tmp_path):
        """Test get_description returns description for single matching path."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("# python")

        config = ProjectConfig(
            paths=[PathConfig(path="src", description="source code")],
            settings=ProjectSettings(),
        )
        file_filter = FileFilter(tmp_path, config)

        assert file_filter.get_description(src_dir / "main.py") == "source code"

    def test_get_description_cumulative(self, tmp_path):
        """Test get_description accumulates descriptions from parent paths."""
        admin_user_dir = tmp_path / "admin" / "user"
        admin_user_dir.mkdir(parents=True)
        (admin_user_dir / "profile.py").write_text("# python")

        config = ProjectConfig(
            paths=[
                PathConfig(path="admin", description="control panel"),
                PathConfig(path="admin/user", description="user management"),
            ],
            settings=ProjectSettings(),
        )
        file_filter = FileFilter(tmp_path, config)

        desc = file_filter.get_description(admin_user_dir / "profile.py")
        assert desc == "control panel, user management"

    def test_get_description_cumulative_three_levels(self, tmp_path):
        """Test cumulative descriptions work with deeply nested paths."""
        deep_dir = tmp_path / "admin" / "user" / "permissions"
        deep_dir.mkdir(parents=True)
        (deep_dir / "roles.py").write_text("# python")

        config = ProjectConfig(
            paths=[
                PathConfig(path="admin", description="control panel"),
                PathConfig(path="admin/user", description="user management"),
                PathConfig(path="admin/user/permissions", description="role-based access"),
            ],
            settings=ProjectSettings(),
        )
        file_filter = FileFilter(tmp_path, config)

        desc = file_filter.get_description(deep_dir / "roles.py")
        assert desc == "control panel, user management, role-based access"

    def test_get_description_skips_empty(self, tmp_path):
        """Test cumulative descriptions skip paths with empty descriptions."""
        admin_user_dir = tmp_path / "admin" / "user"
        admin_user_dir.mkdir(parents=True)
        (admin_user_dir / "profile.py").write_text("# python")

        config = ProjectConfig(
            paths=[
                PathConfig(path="admin", description=""),  # empty
                PathConfig(path="admin/user", description="user management"),
            ],
            settings=ProjectSettings(),
        )
        file_filter = FileFilter(tmp_path, config)

        desc = file_filter.get_description(admin_user_dir / "profile.py")
        assert desc == "user management"

    def test_get_description_no_match(self, tmp_path):
        """Test get_description returns empty string when no paths match."""
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        (other_dir / "file.py").write_text("# python")

        config = ProjectConfig(
            paths=[PathConfig(path="src", description="source code")],
            settings=ProjectSettings(),
        )
        file_filter = FileFilter(tmp_path, config)

        assert file_filter.get_description(other_dir / "file.py") == ""
