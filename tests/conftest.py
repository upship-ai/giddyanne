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
