"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def tmp_path_factory_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def sample_yaml_config(tmp_path):
    """Create a sample .giddyanne.yaml config file."""
    config_content = """
paths:
  - path: src/
    description: Source code
  - path: main.py
    description: Entry point

settings:
  max_file_size: 100000
  ignore_patterns: [.git, __pycache__, .venv]
"""
    config_path = tmp_path / ".giddyanne.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_project_dir(tmp_path):
    """Create a sample project directory structure."""
    # Create directories
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "__pycache__").mkdir()

    # Create files
    (tmp_path / "main.py").write_text("print('hello')")
    (tmp_path / "src" / "app.py").write_text("def run(): pass")
    (tmp_path / "src" / "utils.py").write_text("def helper(): pass")
    (tmp_path / "README.md").write_text("# Project")

    return tmp_path
