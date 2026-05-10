"""Inspect a saved BIRD eval-result JSON: filter, sort, and pretty-print failures.

Use this after every Modal run to read 20-50 wrong examples in 10 minutes and
pattern-match what the model is getting wrong. Pure local — pulls the JSON from
the bird-results volume first if you give it a `bird-results:/path` URI.

Examples:
    # Just look at the worst category of failure
    python -m scripts.inspect_failures results/baseline.json --status wrong --limit 20

    # Drill into one database that's struggling
    python -m scripts.inspect_failures results/baseline.json --db_id formula_1

    # Pull from Modal volume directly (requires `modal` CLI authed)
    python -m scripts.inspect_failures bird-results:/baseline-qwen7b-dev50-first.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter, defaultdict
from pathlib import Path


# Order failures from "most worth inspecting" to least.
_INTEREST_ORDER = ["wrong", "exec_error", "timeout", "empty", "gold_error", "correct"]


def _localize(path_or_uri: str) -> Path:
    """If the arg is `bird-results:/foo.json`, fetch via `modal volume get` to a
    temp file and return that local path. Otherwise return the path as-is."""
    if not path_or_uri.startswith("bird-results:"):
        return Path(path_or_uri)
    if shutil.which("modal") is None:
        sys.exit("`modal` CLI not on PATH; install it or pass a local path")
    remote = path_or_uri.split(":", 1)[1].lstrip("/")
    local_dir = Path(tempfile.mkdtemp(prefix="bird-results-"))
    print(f"[inspect] modal volume get bird-results /{remote} -> {local_dir}/", file=sys.stderr)
    subprocess.run(
        ["modal", "volume", "get", "bird-results", f"/{remote}", str(local_dir)],
        check=True,
    )
    # `modal volume get` lays files out preserving the leaf filename
    return local_dir / Path(remote).name


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="Local JSON path or `bird-results:/...` Modal-volume URI")
    p.add_argument("--status", action="append", default=[],
                   help=f"Filter to one or more statuses ({', '.join(_INTEREST_ORDER)}). "
                        "Repeatable. Default: all wrong/error categories.")
    p.add_argument("--difficulty", action="append", default=[],
                   help="Filter to simple/moderate/challenging. Repeatable.")
    p.add_argument("--db_id", action="append", default=[], help="Filter to one or more db_ids. Repeatable.")
    p.add_argument("--limit", type=int, default=20, help="Max examples to print after filtering.")
    p.add_argument("--summary-only", action="store_true", help="Just print the headline summary, not examples.")
    p.add_argument("--width", type=int, default=120, help="Terminal width for SQL wrapping.")
    return p.parse_args()


def _short(s: str | None, width: int) -> str:
    if s is None:
        return ""
    s = s.strip()
    if len(s) <= width:
        return s
    return s[: width - 1] + "…"


def _wrap_sql(sql: str, width: int) -> str:
    """Conservative wrap for a single SQL string — keeps it readable but bounded."""
    if not sql:
        return "(empty)"
    sql = sql.replace("\r", "").strip()
    return "\n".join(textwrap.wrap(sql, width=width, break_long_words=False) or [""])


def _print_summary(payload: dict) -> None:
    print(f"== {payload.get('split', '?')} split — {payload['n']} examples ==")
    ex = payload.get("ex")
    if ex is not None:
        print(f"EX = {ex:.4f}  ({payload['n_correct']}/{payload['n']})\n")

    print("by status:")
    for k in _INTEREST_ORDER:
        v = payload.get("by_status", {}).get(k, 0)
        if v:
            pct = v / max(payload["n"], 1)
            print(f"  {k:<12s} {v:>5d}  ({pct:.2%})")

    print("\nby difficulty:")
    for d, b in sorted(payload.get("by_difficulty", {}).items()):
        ex = b["correct"] / b["n"] if b["n"] else 0.0
        print(f"  {d:<12s} {b['correct']:>4d}/{b['n']:<4d}  ({ex:.2%})")

    # Per-DB failure rates — surfaces the "this one DB is killing us" pattern.
    by_db: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in payload.get("results", []):
        b = by_db[r["db_id"]]
        b["n"] += 1
        if r["status"] == "correct":
            b["correct"] += 1
    if by_db:
        print("\nby db_id (worst first):")
        ranked = sorted(by_db.items(), key=lambda kv: kv[1]["correct"] / max(kv[1]["n"], 1))
        for db, b in ranked[:8]:
            ex = b["correct"] / b["n"] if b["n"] else 0.0
            print(f"  {db:<28s} {b['correct']:>3d}/{b['n']:<3d}  ({ex:.2%})")


def _print_failure(r: dict, width: int) -> None:
    diff = r.get("difficulty") or "?"
    print(f"\n— q{r['question_id']:>4d}  [{r['status']:<10s}]  [{diff}]  db={r['db_id']}")
    if r.get("error"):
        print(f"  error: {_short(r['error'], width - 9)}")
    print("  pred:")
    for line in _wrap_sql(r.get("predicted_sql", ""), width - 4).splitlines():
        print(f"    {line}")
    print("  gold:")
    for line in _wrap_sql(r.get("gold_sql", ""), width - 4).splitlines():
        print(f"    {line}")


def main() -> None:
    args = _parse_args()
    path = _localize(args.path)
    payload = json.loads(path.read_text())

    _print_summary(payload)
    if args.summary_only:
        return

    # Default filter: anything that isn't correct (the interesting part).
    statuses = set(args.status) if args.status else {s for s in _INTEREST_ORDER if s != "correct"}
    diffs = set(args.difficulty)
    dbs = set(args.db_id)

    rows = [r for r in payload.get("results", [])
            if r["status"] in statuses
            and (not diffs or r.get("difficulty") in diffs)
            and (not dbs or r["db_id"] in dbs)]

    rows.sort(key=lambda r: (_INTEREST_ORDER.index(r["status"]) if r["status"] in _INTEREST_ORDER else 99,
                             r.get("difficulty") or "z",
                             r["db_id"]))

    print(f"\n== showing {min(len(rows), args.limit)} of {len(rows)} matching examples ==")
    for r in rows[: args.limit]:
        _print_failure(r, args.width)


if __name__ == "__main__":
    main()
