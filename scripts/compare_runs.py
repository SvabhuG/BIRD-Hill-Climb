"""Side-by-side comparison of multiple BIRD eval result JSONs.

Loads N saved result files, aligns them on question_id, and prints:
  1. Per-run headline EX + by-status + by-difficulty breakdown
  2. Per-DB EX matrix (rows = db_id, cols = run) with the worst DB per run highlighted
  3. Pairwise win/loss diff (where A is correct and B is wrong, and vice versa)

Use after a multi-model sweep to pick the strongest base, see *which questions*
each model gets uniquely right, and identify databases that are wedge-driving
weak spots.

Examples:
    # Compare local files
    python -m scripts.compare_runs results/baseline-7b.json results/baseline-32b.json

    # Pull straight from the Modal volume
    python -m scripts.compare_runs \\
        bird-results:/baseline-qwen2.5-coder-7b-instruct-dev-full.json \\
        bird-results:/baseline-qwen2.5-coder-32b-instruct-dev-full.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


# Fallback if the user doesn't supply --names: derive a short label from the filename.
def _short_name(path: Path) -> str:
    stem = path.stem
    for prefix in ("baseline-", "linked-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    for suffix in ("-dev-full", "-dev"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _localize(path_or_uri: str) -> Path:
    if not path_or_uri.startswith("bird-results:"):
        return Path(path_or_uri)
    if shutil.which("modal") is None:
        sys.exit("`modal` CLI not on PATH; install or pass a local path")
    remote = path_or_uri.split(":", 1)[1].lstrip("/")
    local_dir = Path(tempfile.mkdtemp(prefix="bird-results-"))
    print(f"[compare] modal volume get bird-results /{remote} -> {local_dir}/", file=sys.stderr)
    subprocess.run(
        ["modal", "volume", "get", "bird-results", f"/{remote}", str(local_dir)],
        check=True,
    )
    return local_dir / Path(remote).name


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _print_headline(runs: list[tuple[str, dict]]) -> None:
    cols = ["EX", "n", "correct", "exec_err", "timeout", "wrong", "empty"]
    name_w = max(len(n) for n, _ in runs)
    name_w = max(name_w, len("model"))
    header = f"{'model':<{name_w}}  " + "  ".join(f"{c:>9s}" for c in cols)
    print(f"\n== headline ==\n{header}")
    for name, payload in runs:
        bs = payload.get("by_status", {})
        cells = {
            "EX": f"{payload.get('ex', 0):.4f}",
            "n": f"{payload.get('n', 0)}",
            "correct": f"{bs.get('correct', 0)}",
            "exec_err": f"{bs.get('exec_error', 0)}",
            "timeout": f"{bs.get('timeout', 0)}",
            "wrong": f"{bs.get('wrong', 0)}",
            "empty": f"{bs.get('empty', 0)}",
        }
        print(f"{name:<{name_w}}  " + "  ".join(f"{cells[c]:>9s}" for c in cols))


def _print_difficulty(runs: list[tuple[str, dict]]) -> None:
    """Per-difficulty EX, model-major. simple→moderate→challenging order."""
    diffs = ["simple", "moderate", "challenging"]
    name_w = max(len(n) for n, _ in runs)
    name_w = max(name_w, len("model"))
    header = f"\n== by difficulty ==\n{'model':<{name_w}}  " + "  ".join(f"{d:>14s}" for d in diffs)
    print(header)
    for name, payload in runs:
        bd = payload.get("by_difficulty", {})
        cells = []
        for d in diffs:
            b = bd.get(d, {"n": 0, "correct": 0})
            ex = b["correct"] / b["n"] if b["n"] else 0.0
            cells.append(f"{b['correct']:>3d}/{b['n']:<3d} {ex:>5.1%}")
        print(f"{name:<{name_w}}  " + "  ".join(f"{c:>14s}" for c in cells))


def _print_per_db(runs: list[tuple[str, dict]], topn: int = 10) -> None:
    """Per-DB EX matrix; sort DBs by *worst* EX across all runs (most painful first)."""
    db_runs: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"n": 0, "correct": 0}))
    for name, payload in runs:
        for r in payload.get("results", []):
            b = db_runs[r["db_id"]][name]
            b["n"] += 1
            if r["status"] == "correct":
                b["correct"] += 1

    # rank DBs by min EX across runs (lowest-min first = most dragging)
    def _min_ex(db: str) -> float:
        return min(
            (db_runs[db][n]["correct"] / max(db_runs[db][n]["n"], 1) for n, _ in runs),
            default=1.0,
        )
    ranked = sorted(db_runs.keys(), key=_min_ex)[:topn]

    name_w = max(len(n) for n, _ in runs)
    db_w = max(len(d) for d in ranked) if ranked else 4
    db_w = max(db_w, len("db_id"))
    header = f"\n== by db_id (worst-min first, top {topn}) ==\n{'db_id':<{db_w}}  " + \
             "  ".join(f"{n:>{max(name_w, 12)}s}" for n, _ in runs)
    print(header)
    for db in ranked:
        cells = []
        for n, _ in runs:
            b = db_runs[db][n]
            ex = b["correct"] / b["n"] if b["n"] else 0.0
            cells.append(f"{b['correct']:>3d}/{b['n']:<3d} {ex:>5.1%}")
        print(f"{db:<{db_w}}  " + "  ".join(f"{c:>{max(name_w, 12)}s}" for c in cells))


def _print_pairwise(runs: list[tuple[str, dict]]) -> None:
    """For each pair (A, B): count question_ids where A correct AND B wrong.

    Cell [A][B] reads as "A wins B" — A got it right, B didn't.
    Diagonal is blank. Useful for spotting complementary models (high cross-wins).
    """
    correct_sets: dict[str, set[int]] = {}
    for name, payload in runs:
        s = {r["question_id"] for r in payload.get("results", []) if r["status"] == "correct"}
        correct_sets[name] = s

    names = [n for n, _ in runs]
    name_w = max(len(n) for n in names)
    print(f"\n== pairwise wins (rows correct, cols wrong; diag = -) ==")
    header = f"{'':<{name_w}}  " + "  ".join(f"{n:>{max(name_w, 8)}s}" for n in names)
    print(header)
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append("-")
            else:
                wins = len(correct_sets[a] - correct_sets[b])
                cells.append(str(wins))
        print(f"{a:<{name_w}}  " + "  ".join(f"{c:>{max(name_w, 8)}s}" for c in cells))

    # Union and intersection of correct sets — shows ensemble ceiling.
    if len(names) > 1:
        union = set().union(*correct_sets.values())
        inter = set.intersection(*correct_sets.values())
        n = runs[0][1].get("n", 0)
        print(f"\nensemble ceiling (any-model-correct): {len(union)}/{n} = {len(union) / max(n, 1):.4f}")
        print(f"intersection (all-models-correct):  {len(inter)}/{n} = {len(inter) / max(n, 1):.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="Local JSON paths or `bird-results:/...` Modal-volume URIs")
    p.add_argument("--names", nargs="*", default=None,
                   help="Optional short labels per file (positional). Defaults derived from filenames.")
    p.add_argument("--topn-db", type=int, default=10, help="How many worst-DB rows to show.")
    args = p.parse_args()

    paths = [_localize(p) for p in args.paths]
    payloads = [_load(p) for p in paths]
    if args.names and len(args.names) != len(paths):
        sys.exit(f"--names count ({len(args.names)}) must match path count ({len(paths)})")
    names = args.names or [_short_name(p) for p in paths]
    runs = list(zip(names, payloads))

    _print_headline(runs)
    _print_difficulty(runs)
    _print_per_db(runs, topn=args.topn_db)
    if len(runs) >= 2:
        _print_pairwise(runs)


if __name__ == "__main__":
    main()
