"""Download the embedding model for giddyanne.

Run this once after `make install` to pre-download the model (~90MB)
so that `giddy up` starts immediately without a download pause.
"""

import sys
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. Run: make python", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    from huggingface_hub.constants import HF_HUB_CACHE
    cache_path = Path(HF_HUB_CACHE) / f"models--sentence-transformers--{MODEL_NAME}"

    dim = model.get_sentence_embedding_dimension()
    print(f"Model: {MODEL_NAME} (dim={dim})")
    print(f"Cache: {cache_path}")
    print("Ready.")


if __name__ == "__main__":
    main()
