"""Download the embedding model for giddyanne.

Run this once after `make install` to pre-download the model
so that `giddy up` starts immediately without a download pause.
"""

import sys
from pathlib import Path

from src.embeddings import _needs_trust_remote_code
from src.project_config import DEFAULT_LOCAL_MODEL, ProjectConfig


def _resolve_model() -> str:
    """Get model name from project config if available, else use default."""
    cwd = Path.cwd()
    config_path = cwd / ".giddyanne.yaml"
    if config_path.exists():
        try:
            config = ProjectConfig.load(config_path)
            return config.settings.local_model
        except Exception:
            pass
    return DEFAULT_LOCAL_MODEL


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. Run: make python", file=sys.stderr)
        sys.exit(1)

    model_name = _resolve_model()

    print(f"Loading model: {model_name}")
    kwargs = {}
    if _needs_trust_remote_code(model_name):
        kwargs["trust_remote_code"] = True
    model = SentenceTransformer(model_name, **kwargs)

    from huggingface_hub.constants import HF_HUB_CACHE
    safe_name = model_name.replace("/", "--")
    cache_path = Path(HF_HUB_CACHE) / f"models--{safe_name}"

    dim = model.get_sentence_embedding_dimension()
    print(f"Model: {model_name} (dim={dim})")
    print(f"Cache: {cache_path}")
    print("Ready.")


if __name__ == "__main__":
    main()
