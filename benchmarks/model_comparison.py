"""Benchmark: CodeRankEmbed vs all-MiniLM-L6-v2

Runs both models end-to-end on a codebase and writes results
to data/model-comparison-<project>-<datetime>.json.

Usage:
    .venv/bin/python benchmarks/model_comparison.py [config.yaml]

Config defaults to benchmarks/configs/giddyanne.yaml if no arg given.
"""

import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Giddyanne repo root (where this script lives), used for sys.path and data output
GIDDYANNE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GIDDYANNE_ROOT))

MODELS = [
    "all-MiniLM-L6-v2",
    "nomic-ai/CodeRankEmbed",
]


def load_config(config_path: Path) -> dict:
    """Load benchmark config from YAML. Returns dict with project_root, queries, expected_files."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Resolve project_root relative to config file's directory
    project_root = Path(raw["project_root"])
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()

    queries = [entry["query"] for entry in raw["queries"]]
    expected_files = {entry["query"]: entry["expected"] for entry in raw["queries"]}

    return {
        "project_root": project_root,
        "queries": queries,
        "expected_files": expected_files,
    }


# Load config from CLI arg or default
_default_config = GIDDYANNE_ROOT / "benchmarks" / "configs" / "giddyanne.yaml"
_config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_config
_config = load_config(_config_path)

PROJECT_ROOT = _config["project_root"]
QUERIES = _config["queries"]
EXPECTED_FILES = _config["expected_files"]
PROJECT_NAME = _config_path.stem

os.chdir(PROJECT_ROOT)

CONFIG_PATH = PROJECT_ROOT / ".giddyanne.yaml"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
INDEX_RUNS = 3
QUERY_EMBED_RUNS = 5
WARM_QUERY_RUNS = 3


def run_cmd(args: list[str], timeout: int = 120) -> tuple[str, float]:
    """Run a command, return (stdout, elapsed_seconds)."""
    start = time.perf_counter()
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, cwd=PROJECT_ROOT
    )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        print(f"  WARN: {' '.join(args)} exited {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(f"  stderr: {result.stderr.strip()}", file=sys.stderr)
    return result.stdout.strip(), elapsed


def giddy(*args: str, timeout: int = 120) -> tuple[str, float]:
    return run_cmd(["giddy", *args], timeout=timeout)


def wait_for_ready(timeout: int = 120):
    """Poll giddy status until state is 'ready' or timeout."""
    deadline = time.time() + timeout
    out = ""
    while time.time() < deadline:
        try:
            out, _ = giddy("status", timeout=5)
        except subprocess.TimeoutExpired:
            time.sleep(1)
            continue
        lower = out.lower()
        if "ready" in lower or "running" in lower:
            return
        # Still starting or indexing — keep polling
        time.sleep(1)
    raise TimeoutError(f"Server not ready after {timeout}s. Last status: {out}")


def clear_hf_cache_for_model(model_name: str):
    """Delete HuggingFace cache entries for a specific model."""
    # HF cache uses models--<org>--<name> directory format
    cache_dir_name = "models--" + model_name.replace("/", "--")
    cache_path = HF_CACHE / cache_dir_name
    if cache_path.exists():
        print(f"  Clearing HF cache: {cache_path}")
        shutil.rmtree(cache_path)
    else:
        print(f"  No HF cache found at {cache_path}")


def write_config(model_name: str):
    """Patch the project's .giddyanne.yaml to use the given model."""
    import re
    text = CONFIG_PATH.read_text()
    line = f"  local_model: {model_name}"
    if re.search(r"^\s*local_model:", text, re.MULTILINE):
        text = re.sub(r"^\s*local_model:.*$", line, text, flags=re.MULTILINE)
    elif re.search(r"^settings:", text, re.MULTILINE):
        text = re.sub(r"^(settings:.*)", rf"\1\n{line}", text, flags=re.MULTILINE)
    else:
        text += f"\nsettings:\n{line}\n"
    CONFIG_PATH.write_text(text)


def time_model_load(model_name: str) -> dict:
    """Time SentenceTransformer load (cached only)."""
    from src.embeddings import _needs_trust_remote_code
    from sentence_transformers import SentenceTransformer

    kwargs = {}
    if _needs_trust_remote_code(model_name):
        kwargs["trust_remote_code"] = True

    print("  Timing cached load...")
    start = time.perf_counter()
    model = SentenceTransformer(model_name, **kwargs)
    cached_load = time.perf_counter() - start

    results = {
        "fresh_load_seconds": None,
        "cached_load_seconds": cached_load,
        "dimension": model.get_sentence_embedding_dimension(),
    }
    del model
    return results


def time_indexing(model_name: str) -> dict:
    """Time full index (giddy clean + up, wait for ready). Returns median of INDEX_RUNS."""
    times = []
    for i in range(INDEX_RUNS):
        print(f"  Index run {i + 1}/{INDEX_RUNS}...")
        giddy("down")
        giddy("clean", "--force")
        start = time.perf_counter()
        giddy("up")
        wait_for_ready(timeout=600)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        giddy("down")
        print(f"    {elapsed:.2f}s")

    return {
        "runs": times,
        "median_seconds": statistics.median(times),
    }


def run_searches(queries: list[str]) -> dict:
    """Run cold + warm searches, return timing and results."""
    cold_results = {}
    warm_results = {}

    for query in queries:
        # Cold search
        out, elapsed = giddy("find", query, "--json", "--limit", "5")
        try:
            results = json.loads(out)
        except json.JSONDecodeError:
            results = []
        cold_results[query] = {
            "time_seconds": elapsed,
            "results": [
                {"path": r["path"], "score": r["score"]} for r in results[:5]
            ],
        }

        # Warm searches
        warm_times = []
        for _ in range(WARM_QUERY_RUNS):
            _, elapsed = giddy("find", query, "--json", "--limit", "5")
            warm_times.append(elapsed)
        warm_results[query] = {
            "times": warm_times,
            "avg_seconds": statistics.mean(warm_times),
        }

    return {"cold": cold_results, "warm": warm_results}


def time_query_embedding(model_name: str, queries: list[str]) -> dict:
    """Time embed_query() directly (no HTTP overhead)."""
    from src.embeddings import EmbeddingService, LocalEmbedding

    provider = LocalEmbedding(model_name)
    service = EmbeddingService(provider)

    results = {}
    for query in queries:
        times = []
        for _ in range(QUERY_EMBED_RUNS):
            start = time.perf_counter()
            asyncio.run(service.embed_query(query))
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        results[query] = {
            "times": times,
            "median_seconds": statistics.median(times),
        }

    return results


def score_quality(search_results: dict) -> dict:
    """Score search results against expected files."""
    scores = {}
    for query, expected in EXPECTED_FILES.items():
        cold = search_results["cold"].get(query, {})
        result_paths = [r["path"] for r in cold.get("results", [])]
        # Match by checking if expected file appears as suffix of any result path
        found = 0
        for exp in expected:
            if any(rp.endswith(exp) for rp in result_paths):
                found += 1
        category = "broad" if len(expected) > 1 else "precise"
        scores[query] = {
            "category": category,
            "expected": expected,
            "found": found,
            "total_expected": len(expected),
            "score": found / len(expected) if expected else 0,
            "top_5_paths": [
                os.path.relpath(p, PROJECT_ROOT) for p in result_paths
            ],
        }

    avg_score = statistics.mean(s["score"] for s in scores.values())
    broad_scores = [s["score"] for s in scores.values() if s["category"] == "broad"]
    precise_scores = [s["score"] for s in scores.values() if s["category"] == "precise"]
    return {
        "per_query": scores,
        "average_score": avg_score,
        "broad_score": statistics.mean(broad_scores) if broad_scores else 0,
        "precise_score": statistics.mean(precise_scores) if precise_scores else 0,
    }


def print_summary(report: dict):
    """Print a readable comparison table."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)

    models = list(report["models"].keys())
    col_width = 30

    def row(label, *values):
        print(f"  {label:<28}", end="")
        for v in values:
            print(f"  {str(v):>{col_width}}", end="")
        print()

    def separator():
        print("  " + "-" * (28 + (col_width + 2) * len(models)))

    print()
    row("", *models)
    separator()

    # Model info
    row(
        "Dimension",
        *[str(report["models"][m]["model_load"]["dimension"]) for m in models],
    )
    separator()

    # Load times
    row(
        "Fresh download + load",
        *[f"{v:.2f}s" if (v := report['models'][m]['model_load']['fresh_load_seconds']) else "skipped" for m in models],
    )
    row(
        "Cached load",
        *[f"{report['models'][m]['model_load']['cached_load_seconds']:.2f}s" for m in models],
    )
    separator()

    # Indexing
    row(
        "Index time (median, 3 runs)",
        *[f"{report['models'][m]['indexing']['median_seconds']:.2f}s" for m in models],
    )
    separator()

    # Search times
    for m in models:
        cold_times = [
            report["models"][m]["searches"]["cold"][q]["time_seconds"]
            for q in QUERIES
        ]
        warm_times = [
            report["models"][m]["searches"]["warm"][q]["avg_seconds"]
            for q in QUERIES
        ]
        report["models"][m]["_cold_avg"] = statistics.mean(cold_times)
        report["models"][m]["_warm_avg"] = statistics.mean(warm_times)

    row(
        "Cold search (avg 5 queries)",
        *[f"{report['models'][m]['_cold_avg']:.3f}s" for m in models],
    )
    row(
        "Warm search (avg 5 queries)",
        *[f"{report['models'][m]['_warm_avg']:.3f}s" for m in models],
    )
    separator()

    # Query embed times
    for m in models:
        embed_times = [
            report["models"][m]["query_embedding"][q]["median_seconds"]
            for q in QUERIES
        ]
        report["models"][m]["_embed_avg"] = statistics.mean(embed_times)

    row(
        "Query embed (median, avg 5q)",
        *[f"{report['models'][m]['_embed_avg']:.4f}s" for m in models],
    )
    separator()

    # Quality
    row(
        "Quality (overall)",
        *[f"{report['models'][m]['quality']['average_score']:.0%}" for m in models],
    )
    row(
        "Quality (broad, 3q)",
        *[f"{report['models'][m]['quality']['broad_score']:.0%}" for m in models],
    )
    row(
        "Quality (precise, 2q)",
        *[f"{report['models'][m]['quality']['precise_score']:.0%}" for m in models],
    )
    separator()

    # Per-query quality detail
    broad_queries = [q for q in QUERIES if len(EXPECTED_FILES[q]) > 1]
    precise_queries = [q for q in QUERIES if len(EXPECTED_FILES[q]) == 1]

    for label, query_group in [("BROAD", broad_queries), ("PRECISE", precise_queries)]:
        print(f"\n  {label} QUERIES (top-5 results)")
        print("  " + "-" * 68)
        for query in query_group:
            print(f"\n  Query: \"{query}\"")
            for m in models:
                q_data = report["models"][m]["quality"]["per_query"][query]
                score = q_data["score"]
                paths = q_data["top_5_paths"]
                expected = q_data["expected"]
                print(f"    {m}:")
                print(f"      Score: {score:.0%} (expected: {', '.join(expected)})")
                for i, p in enumerate(paths, 1):
                    marker = " *" if any(p.endswith(e) for e in expected) else ""
                    print(f"      {i}. {p}{marker}")

    # Cleanup temp keys
    for m in models:
        report["models"][m].pop("_cold_avg", None)
        report["models"][m].pop("_warm_avg", None)
        report["models"][m].pop("_embed_avg", None)


def main():
    print("Model Comparison Benchmark")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Queries: {len(QUERIES)}")
    print()

    # Stop any existing server so we own the port
    giddy("down")

    # Save original config
    original_config = CONFIG_PATH.read_text()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": str(PROJECT_ROOT),
        "models": {},
        "queries": QUERIES,
        "expected_files": EXPECTED_FILES,
    }

    try:
        for model_name in MODELS:
            print(f"\n{'='*50}")
            print(f"MODEL: {model_name}")
            print(f"{'='*50}")

            model_report = {}

            # 1. Model download + load
            print("\n[1/5] Model load timing...")
            model_report["model_load"] = time_model_load(model_name)
            load = model_report["model_load"]
            fresh = f"{load['fresh_load_seconds']:.2f}s" if load["fresh_load_seconds"] else "skipped"
            print(f"  Fresh: {fresh}, Cached: {load['cached_load_seconds']:.2f}s, Dim: {load['dimension']}")

            # 2. Write config and index
            print("\n[2/5] Indexing...")
            write_config(model_name)
            model_report["indexing"] = time_indexing(model_name)
            print(f"  Median: {model_report['indexing']['median_seconds']:.2f}s")

            # 3-4. Cold + warm searches (need server running)
            print("\n[3/5] Searches (cold + warm)...")
            giddy("clean", "--force")
            giddy("up")
            wait_for_ready(timeout=600)
            model_report["searches"] = run_searches(QUERIES)
            giddy("down")

            # 5. Query embedding isolation
            print("\n[4/5] Query embedding timing...")
            model_report["query_embedding"] = time_query_embedding(
                model_name, QUERIES
            )

            # 6. Quality scoring
            print("\n[5/5] Quality scoring...")
            model_report["quality"] = score_quality(model_report["searches"])
            print(f"  Average quality: {model_report['quality']['average_score']:.0%}")

            report["models"][model_name] = model_report

            # Clean up
            giddy("down")
            giddy("clean", "--force")

    finally:
        # Restore original config
        CONFIG_PATH.write_text(original_config)
        print("\nRestored original .giddyanne.yaml")

    # Print summary
    print_summary(report)

    # Write JSON report (always to giddyanne's data/ folder)
    data_dir = GIDDYANNE_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = data_dir / f"model-comparison-{PROJECT_NAME}-{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
