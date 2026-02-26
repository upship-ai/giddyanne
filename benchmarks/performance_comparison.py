"""Benchmark: giddyanne vs grepai — performance and search quality

Measures indexing speed, memory, index size, search latency, and search
quality (recall/precision/MRR) for all tool variants across multiple repos.

Usage:
    .venv/bin/python benchmarks/performance_comparison.py [config.yaml ...]

Defaults to all configs in benchmarks/configs/ if no args given.
Requires: ollama running, giddy CLI available, grepai CLI available.
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

GIDDYANNE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GIDDYANNE_ROOT))
from src.project_config import FileFilter, ProjectConfig  # noqa: E402

INDEX_RUNS = 1
WARM_SEARCH_RUNS = 5

TOOLS = ["giddyanne"]
# Also available: "giddyanne-ollama", "giddyanne-fulltext", "giddyanne-semantic",
# "giddyanne-nomic", "giddyanne-nomic-ollama", "grepai"
# Note: ollama variants need batching support to be practical
# (timeout at 600s due to per-request HTTP overhead)

# Tool config: (ollama, local_model override, ollama_model override, search_flags)
TOOL_CONFIG = {
    "giddyanne":              (False, None, None, []),
    "giddyanne-fulltext":     (False, None, None, ["--full-text"]),
    "giddyanne-semantic":     (False, None, None, ["--semantic"]),
    "giddyanne-ollama":       (True,  None, None, []),
    "giddyanne-nomic":        (False, "nomic-ai/nomic-embed-text-v1.5", None, []),
    "giddyanne-nomic-ollama": (True,  None, "nomic-embed-text", []),
    "grepai":                 (None,  None, None, []),
}

# Tools that share an index — only the first one indexes, the rest reuse it.
GIDDY_INDEX_GROUP = {"giddyanne", "giddyanne-fulltext", "giddyanne-semantic"}


# ── config loading ────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    project_root = Path(raw["project_root"])
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()

    queries = [entry["query"] for entry in raw["queries"]]
    expected_files = {entry["query"]: entry["expected"] for entry in raw["queries"]}

    return {
        "project_root": project_root,
        "queries": queries,
        "expected_files": expected_files,
        "config_name": config_path.stem,
    }


def count_indexable_files(project_root: Path) -> int:
    """Count files that would be indexed, using the real FileFilter."""
    config_path = project_root / ".giddyanne.yaml"
    if config_path.exists():
        config = ProjectConfig.load(config_path)
    else:
        config = ProjectConfig.default(project_root)

    file_filter = FileFilter(project_root, config)

    # Deduplicate overlapping paths (e.g. app/ and app/(protected)/)
    sorted_paths = sorted(config.paths, key=lambda pc: pc.path)
    walk_targets: list[Path] = []
    for pc in sorted_paths:
        target = project_root / pc.path
        if not target.exists():
            continue
        if any(target == t or target.is_relative_to(t) for t in walk_targets):
            continue
        walk_targets.append(target)

    seen: set[Path] = set()
    for target in walk_targets:
        if target.is_file():
            if file_filter.should_include(target):
                seen.add(target)
            continue
        for dirpath, _, filenames in os.walk(target):
            for f in filenames:
                fp = Path(dirpath) / f
                if fp not in seen and file_filter.should_include(fp):
                    seen.add(fp)
    return len(seen)


# ── utilities ─────────────────────────────────────────────────────

def run_cmd(args: list[str], cwd: Path, timeout: int = 600) -> tuple[str, str, float, int]:
    """Run a command, return (stdout, stderr, elapsed, returncode)."""
    start = time.perf_counter()
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, cwd=cwd
    )
    elapsed = time.perf_counter() - start
    return result.stdout.strip(), result.stderr.strip(), elapsed, result.returncode


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / (1024 * 1024):.1f} MB"


def get_process_rss(pattern: str) -> int | None:
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if pattern in line and "grep" not in line and "ps aux" not in line:
                parts = line.split()
                if len(parts) >= 6:
                    rss_kb = int(parts[5])
                    return rss_kb * 1024
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return None


# ── search quality scoring ────────────────────────────────────────

def score_results(results: list[dict], expected: list[str], project_root: Path) -> dict:
    """Score search results against expected files."""
    result_paths = [r["path"] for r in results]

    found_files = []
    missed_files = []
    for exp in expected:
        if any(rp.endswith(exp) for rp in result_paths):
            found_files.append(exp)
        else:
            missed_files.append(exp)

    # Reciprocal rank: rank of first expected file found
    rr = 0.0
    for i, rp in enumerate(result_paths):
        if any(rp.endswith(exp) for exp in expected):
            rr = 1.0 / (i + 1)
            break

    recall = len(found_files) / len(expected) if expected else 0
    relevant_in_results = sum(
        1 for rp in result_paths if any(rp.endswith(exp) for exp in expected)
    )
    precision = relevant_in_results / len(result_paths) if result_paths else 0

    return {
        "recall": recall,
        "precision": precision,
        "reciprocal_rank": rr,
        "found": found_files,
        "missed": missed_files,
        "top_5": [
            os.path.relpath(rp, project_root)
            if rp.startswith(str(project_root)) else rp
            for rp in result_paths
        ],
    }


# ── giddyanne helpers ─────────────────────────────────────────────

def giddy_clean(cwd: Path):
    subprocess.run(["giddy", "down"], capture_output=True, cwd=cwd, timeout=30)
    subprocess.run(["pkill", "-f", "http_main.py"], capture_output=True, timeout=5)
    time.sleep(1)
    subprocess.run(["giddy", "clean", "--force"], capture_output=True, cwd=cwd, timeout=30)


def giddy_wait_ready(cwd: Path, timeout: int = 600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out, _, _, rc = run_cmd(["giddy", "status"], cwd, timeout=5)
        except subprocess.TimeoutExpired:
            time.sleep(1)
            continue
        lower = out.lower()
        if "running" in lower or "pid" in lower:
            try:
                status_out, _, _, src = run_cmd(
                    ["curl", "-s", "http://127.0.0.1:8000/status"], cwd, timeout=5
                )
                status = json.loads(status_out)
                if status.get("state") == "ready":
                    return
            except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
                pass
        time.sleep(1)
    raise TimeoutError(f"giddy not ready after {timeout}s")


def giddy_patch_config(cwd: Path, local_model: str | None, ollama_model: str | None) -> str | None:
    """Temporarily patch model settings in .giddyanne.yaml. Returns original file content."""
    import re
    config_path = cwd / ".giddyanne.yaml"
    if not config_path.exists():
        return None
    original = config_path.read_text()
    text = original

    for key, new_val in [("local_model", local_model), ("ollama_model", ollama_model)]:
        if not new_val:
            continue
        if re.search(rf"{key}:\s*\S+", text):
            text = re.sub(rf"{key}:\s*\S+", f"{key}: {new_val}", text)
        else:
            text = re.sub(r"(settings:)", f"\\1\n  {key}: {new_val}", text)

    if text != original:
        config_path.write_text(text)
        return original
    return None


def giddy_restore_config(cwd: Path, original_content: str | None):
    if original_content is None:
        return
    (cwd / ".giddyanne.yaml").write_text(original_content)


def giddy_index(cwd: Path, ollama: bool = False) -> float:
    """Clean, start, wait for ready. Returns elapsed seconds."""
    giddy_clean(cwd)
    cmd = ["giddy", "up"]
    if ollama:
        cmd.append("--ollama")
    start = time.perf_counter()
    subprocess.run(cmd, capture_output=True, cwd=cwd, timeout=600)
    giddy_wait_ready(cwd)
    return time.perf_counter() - start


def giddy_stop(cwd: Path):
    subprocess.run(["giddy", "down"], capture_output=True, cwd=cwd, timeout=30)


def giddy_index_size(cwd: Path) -> int:
    return dir_size(cwd / ".giddyanne")


def giddy_memory(cwd: Path) -> int | None:
    return get_process_rss("http_main.py")


def giddy_search(
    query: str, cwd: Path, extra_flags: list[str] | None = None,
) -> tuple[list[dict], float]:
    cmd = ["giddy", "find", query, "--json", "--limit", "5"]
    if extra_flags:
        cmd.extend(extra_flags)
    out, _, elapsed, rc = run_cmd(cmd, cwd)
    if rc != 0:
        return [], elapsed
    try:
        data = json.loads(out)
        return [{"path": r.get("path", ""), "score": r.get("score", 0)} for r in data[:5]], elapsed
    except json.JSONDecodeError:
        return [], elapsed


# ── grepai helpers ────────────────────────────────────────────────

def grepai_clean(cwd: Path):
    subprocess.run(
        ["grepai", "watch", "--stop"], capture_output=True, cwd=cwd, timeout=10
    )
    time.sleep(1)
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"grepai.*{cwd}"],
            capture_output=True, text=True, timeout=5
        )
        for pid in result.stdout.strip().split():
            subprocess.run(["kill", pid], capture_output=True, timeout=5)
    except (subprocess.TimeoutExpired, ValueError):
        pass
    time.sleep(1)
    grepai_dir = cwd / ".grepai"
    if grepai_dir.exists():
        shutil.rmtree(grepai_dir)


def grepai_wait_indexed(cwd: Path, timeout: int = 600) -> subprocess.Popen:
    log_path = cwd / ".grepai" / "bench-watch.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        ["grepai", "watch"],
        stdout=log_file, stderr=subprocess.STDOUT,
        cwd=cwd,
    )

    deadline = time.time() + timeout
    time.sleep(3)

    while time.time() < deadline:
        if proc.poll() is not None:
            log_file.close()
            raise TimeoutError(f"grepai watch exited with code {proc.returncode}")
        try:
            content = log_path.read_text(errors="replace")
            if "Watching for changes" in content:
                return proc
        except OSError:
            pass
        time.sleep(3)

    proc.terminate()
    log_file.close()
    raise TimeoutError(f"grepai indexing timed out after {timeout}s")


def grepai_index(cwd: Path) -> tuple[float, subprocess.Popen]:
    """Clean, init, watch, wait for index. Returns (elapsed, proc)."""
    grepai_clean(cwd)
    subprocess.run(
        ["grepai", "init", "--yes"],
        capture_output=True, cwd=cwd, timeout=30
    )
    start = time.perf_counter()
    proc = grepai_wait_indexed(cwd)
    return time.perf_counter() - start, proc


def grepai_stop(cwd: Path, proc: subprocess.Popen | None = None):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    subprocess.run(
        ["grepai", "watch", "--stop"], capture_output=True, cwd=cwd, timeout=10
    )
    time.sleep(1)


def grepai_index_size(cwd: Path) -> int:
    return dir_size(cwd / ".grepai")


def grepai_memory(cwd: Path) -> int | None:
    return get_process_rss("grepai")


def grepai_search(query: str, cwd: Path) -> tuple[list[dict], float]:
    out, _, elapsed, rc = run_cmd(
        ["grepai", "search", query, "--json", "--limit", "5"], cwd
    )
    if rc != 0:
        return [], elapsed
    try:
        data = json.loads(out)
        results = [
            {"path": r.get("file_path", ""), "score": r.get("score", 0)}
            for r in data[:5]
        ]
        return results, elapsed
    except json.JSONDecodeError:
        return [], elapsed


# ── benchmark runner ──────────────────────────────────────────────

def benchmark_tool(
    tool: str, config: dict, skip_indexing: bool = False, keep_running: bool = False,
    shared_index_times: list[float] | None = None,
) -> dict:
    cwd = config["project_root"]
    queries = config["queries"]
    expected_files = config["expected_files"]
    print(f"\n  {tool}")
    print(f"  {'=' * len(tool)}")

    result = {
        "index_times": [],
        "index_size_bytes": 0,
        "memory_rss_bytes": None,
        "search_cold": [],
        "search_warm_avg": [],
        "quality": {},
        "file_count": count_indexable_files(cwd),
    }
    grepai_proc = None

    is_giddy = tool.startswith("giddyanne")
    use_ollama, local_model, ollama_model, search_flags = TOOL_CONFIG[tool]
    saved_config = None
    if is_giddy and (local_model or ollama_model):
        saved_config = giddy_patch_config(cwd, local_model, ollama_model)

    # ── indexing ──
    if skip_indexing:
        print("    Index: reusing running server")
        result["index_times"] = shared_index_times or []
        result["index_size_bytes"] = giddy_index_size(cwd)
        result["memory_rss_bytes"] = giddy_memory(cwd)
    else:
        for i in range(INDEX_RUNS):
            print(f"    Index run {i + 1}/{INDEX_RUNS}...", end=" ", flush=True)
            try:
                if is_giddy:
                    elapsed = giddy_index(cwd, ollama=use_ollama)
                    result["index_times"].append(elapsed)
                    print(f"{elapsed:.2f}s")

                    if i == INDEX_RUNS - 1:
                        result["index_size_bytes"] = giddy_index_size(cwd)
                        result["memory_rss_bytes"] = giddy_memory(cwd)

                else:
                    elapsed, grepai_proc = grepai_index(cwd)
                    result["index_times"].append(elapsed)
                    print(f"{elapsed:.2f}s")

                    if i == INDEX_RUNS - 1:
                        result["index_size_bytes"] = grepai_index_size(cwd)
                        result["memory_rss_bytes"] = grepai_memory(cwd)
                    else:
                        grepai_stop(cwd, grepai_proc)
                        grepai_proc = None

            except (TimeoutError, subprocess.TimeoutExpired) as e:
                print(f"TIMEOUT: {e}")
                result["index_times"].append(None)

    # ── make sure tool is running for search ──
    try:
        if is_giddy:
            giddy_wait_ready(cwd, timeout=10)
        else:
            pass  # already running from last grepai_index call
    except TimeoutError:
        print("    WARN: tool not ready for search tests")
        if is_giddy:
            giddy_restore_config(cwd, saved_config)
        return result

    # ── search latency + quality ──
    if is_giddy:
        def search_fn(q, d):
            return giddy_search(q, d, extra_flags=search_flags)
    else:
        search_fn = grepai_search
    print(f"    Search ({len(queries)} queries)...")
    for query in queries:
        results, cold_time = search_fn(query, cwd)
        result["search_cold"].append(cold_time)

        # Quality scoring
        quality = score_results(results, expected_files[query], cwd)
        result["quality"][query] = quality
        status = "HIT" if quality["recall"] > 0 else "MISS"
        print(f"      {status} ({quality['recall']:.0%}) {query[:50]}  [{cold_time:.3f}s]")

        # Warm searches
        warm_times = []
        for _ in range(WARM_SEARCH_RUNS):
            _, t = search_fn(query, cwd)
            warm_times.append(t)
        result["search_warm_avg"].append(statistics.mean(warm_times))

    # ── cleanup ──
    if is_giddy:
        if not keep_running:
            giddy_stop(cwd)
        giddy_restore_config(cwd, saved_config)
    else:
        grepai_stop(cwd, grepai_proc)

    # ── summarize ──
    valid_times = [t for t in result["index_times"] if t is not None]
    median_index = statistics.median(valid_times) if valid_times else None
    avg_cold = statistics.mean(result["search_cold"]) if result["search_cold"] else None
    avg_warm = statistics.mean(result["search_warm_avg"]) if result["search_warm_avg"] else None

    recalls = [q["recall"] for q in result["quality"].values()]
    precisions = [q["precision"] for q in result["quality"].values()]
    rrs = [q["reciprocal_rank"] for q in result["quality"].values()]

    result["summary"] = {
        "median_index_seconds": median_index,
        "index_size_human": format_bytes(result["index_size_bytes"]),
        "memory_rss_human": (
            format_bytes(result["memory_rss_bytes"])
            if result["memory_rss_bytes"] else None
        ),
        "avg_cold_search": avg_cold,
        "avg_warm_search": avg_warm,
        "files_per_second": result["file_count"] / median_index if median_index else None,
        "avg_recall": statistics.mean(recalls) if recalls else None,
        "avg_precision": statistics.mean(precisions) if precisions else None,
        "mrr": statistics.mean(rrs) if rrs else None,
    }

    print("    ---")
    if median_index:
        fps = result['summary']['files_per_second']
        n = result['file_count']
        print(f"    Index: {median_index:.2f}s median ({n} files, {fps:.1f} files/s)")
    else:
        print("    Index: FAILED")
    print(f"    Size: {result['summary']['index_size_human']}")
    if result["memory_rss_bytes"]:
        print(f"    Memory: {result['summary']['memory_rss_human']}")
    if avg_cold:
        print(f"    Search: {avg_cold:.3f}s cold, {avg_warm:.3f}s warm")
    if recalls:
        r = result['summary']['avg_recall']
        p = result['summary']['avg_precision']
        mrr = result['summary']['mrr']
        print(f"    Quality: {r:.0%} recall, {p:.0%} precision, {mrr:.2f} MRR")

    return result


def benchmark_repo(config: dict) -> dict:
    cwd = config["project_root"]
    name = config["config_name"]
    file_count = count_indexable_files(cwd)

    print(f"\n{'=' * 60}")
    print(f"REPO: {name} ({cwd})")
    print(f"Files: {file_count}")
    print(f"{'=' * 60}")

    repo_result = {
        "project": str(cwd),
        "project_name": name,
        "file_count": file_count,
        "expected_files": config["expected_files"],
        "tools": {},
    }

    # Identify which tools are in the shared giddy index group
    giddy_group_tools = [t for t in TOOLS if t in GIDDY_INDEX_GROUP]
    giddy_group_last = giddy_group_tools[-1] if giddy_group_tools else None
    giddy_group_indexed = False
    giddy_group_first_result = None

    for tool in TOOLS:
        in_group = tool in GIDDY_INDEX_GROUP
        skip = in_group and giddy_group_indexed
        keep = in_group and tool != giddy_group_last
        shared_times = (
            giddy_group_first_result["index_times"]
            if skip and giddy_group_first_result else None
        )

        repo_result["tools"][tool] = benchmark_tool(
            tool, config, skip_indexing=skip, keep_running=keep,
            shared_index_times=shared_times,
        )

        if in_group and not giddy_group_indexed:
            giddy_group_indexed = True
            giddy_group_first_result = repo_result["tools"][tool]

    return repo_result


# ── reporting ─────────────────────────────────────────────────────

def print_report(report: dict):
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    col = 22

    def row(label, *values):
        print(f"  {label:<28}", end="")
        for v in values:
            print(f"  {str(v):>{col}}", end="")
        print()

    def fmt_time(t):
        return f"{t:.2f}s" if t else "N/A"

    def fmt_rate(r):
        return f"{r:.1f} files/s" if r else "N/A"

    def fmt_pct(v):
        return f"{v:.0%}" if v is not None else "N/A"

    def fmt_score(v):
        return f"{v:.2f}" if v is not None else "N/A"

    for repo in report["repos"]:
        name = repo["project_name"]
        files = repo["file_count"]
        tools = list(repo["tools"].keys())
        summaries = [repo["tools"][t]["summary"] for t in tools]

        def sep():
            print("  " + "-" * (28 + (col + 2) * len(tools)))

        print(f"\n  {name} ({files} files)")
        row("", *tools)
        sep()

        row("Index time (median)", *[fmt_time(s["median_index_seconds"]) for s in summaries])
        row("Index rate", *[fmt_rate(s["files_per_second"]) for s in summaries])
        row("Index size", *[s["index_size_human"] for s in summaries])
        row("Memory (RSS)", *[s.get("memory_rss_human") or "N/A" for s in summaries])
        sep()
        row("Search cold (avg)", *[fmt_time(s["avg_cold_search"]) for s in summaries])
        row("Search warm (avg)", *[fmt_time(s["avg_warm_search"]) for s in summaries])
        sep()
        row("Recall (avg)", *[fmt_pct(s["avg_recall"]) for s in summaries])
        row("Precision (avg)", *[fmt_pct(s["avg_precision"]) for s in summaries])
        row("MRR", *[fmt_score(s["mrr"]) for s in summaries])
        sep()

        # Per-query detail
        print("\n  Per-query detail:")
        expected_files = repo["expected_files"]
        queries = list(expected_files.keys())
        for query in queries:
            print(f"\n    \"{query}\"")
            for t in tools:
                q = repo["tools"][t].get("quality", {}).get(query)
                if not q:
                    print(f"      {t}: N/A")
                    continue
                r = q['recall']
                rr = q['reciprocal_rank']
                top = ", ".join(
                    os.path.basename(p)
                    + (" *" if any(p.endswith(e) for e in expected_files[query]) else "")
                    for p in q["top_5"][:3]
                )
                print(f"      {t}: {r:.0%} recall, RR={rr:.2f}  [{top}]")
                if q["missed"]:
                    print(f"        missed: {', '.join(os.path.basename(m) for m in q['missed'])}")


def main():
    config_dir = GIDDYANNE_ROOT / "benchmarks" / "configs"

    args = sys.argv[1:]
    if args:
        config_paths = [Path(p) for p in args]
    else:
        config_paths = sorted(config_dir.glob("*.yaml"))

    configs = [load_config(p) for p in config_paths]

    print("Benchmark")
    print(f"Tools: {', '.join(TOOLS)}")
    print(f"Repos: {len(configs)}")
    print(f"Index runs: {INDEX_RUNS}")
    print(f"Search warm runs: {WARM_SEARCH_RUNS}")

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "tools": TOOLS,
        "index_runs": INDEX_RUNS,
        "search_warm_runs": WARM_SEARCH_RUNS,
        "repos": [],
    }

    for config in configs:
        repo_result = benchmark_repo(config)
        report["repos"].append(repo_result)

    print_report(report)

    # Write JSON
    data_dir = GIDDYANNE_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = data_dir / f"benchmark-{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
