"""Pairwise per-question diff between two BIRD result JSONs.

Where `compare_runs.py` aggregates N runs (headline EX, by-DB matrix, ensemble
ceiling), this tool gives you the *per-question* picture for any two runs:
  - Transition matrix: broke_it / fixed_it / stayed_correct / stayed_wrong
  - Net delta by difficulty and by db_id (where do gains and losses concentrate?)
  - Same-SQL vs different-SQL split for transitions (rules out eval flakes)
  - Heuristic categorization of broken/fixed transitions (which SQL features
    changed: aggregation / JOIN count / CAST / GROUP BY / ORDER BY / LIKE)
  - Side-by-side samples of broken and fixed cases for manual inspection

Use after any A/B experiment — same base, different strategy; same strategy,
different base; before/after a bug fix; etc.

Examples:
    # Local files
    python -m scripts.diff_runs results/correction-q3cm.json results/vc-q3cm.json

    # Pull from Modal volume
    python -m scripts.diff_runs \\
        bird-results:/correction-qwen3-coder-30b-a3b-instruct-dev-full.json \\
        bird-results:/voting-correction-qwen3-coder-30b-a3b-instruct-dev-full.json

    # Focus on a specific subset
    python -m scripts.diff_runs A.json B.json --difficulty moderate --db_id financial

    # More samples per direction
    python -m scripts.diff_runs A.json B.json --samples 8

    # Save the full structured diff
    python -m scripts.diff_runs A.json B.json --out diff.json
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


def _localize(path_or_uri: str) -> Path:
    if not path_or_uri.startswith("bird-results:"):
        return Path(path_or_uri)
    if shutil.which("modal") is None:
        sys.exit("`modal` CLI not on PATH; install it or pass a local path")
    remote = path_or_uri.split(":", 1)[1].lstrip("/")
    local_dir = Path(tempfile.mkdtemp(prefix="bird-results-"))
    print(f"[diff] modal volume get bird-results /{remote} -> {local_dir}/", file=sys.stderr)
    subprocess.run(
        ["modal", "volume", "get", "bird-results", f"/{remote}", str(local_dir)],
        check=True,
    )
    return local_dir / Path(remote).name


def _short_name(path: Path) -> str:
    stem = path.stem
    for prefix in ("baseline-", "linked-", "linking-", "voting-", "correction-",
                   "cot-", "fewshot-", "voting-correction-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    for suffix in ("-dev-full", "-dev"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


# ---------- categorization heuristics ----------

_AGG_TOKENS = ["count(", "sum(", "avg(", "min(", "max("]


def _join_count(sql: str) -> int:
    s = sql.lower()
    return s.count(" join ") + s.count("inner join") + s.count("left join")


def categorize_change(sql_a: str, sql_b: str) -> list[str]:
    """Return tags describing how sql_a differs from sql_b. Empty list = identical."""
    if sql_a.strip() == sql_b.strip():
        return []
    a, b = sql_a.lower(), sql_b.lower()
    tags: list[str] = []

    # Aggregation
    a_aggs = {tok.rstrip("(") for tok in _AGG_TOKENS if tok in a}
    b_aggs = {tok.rstrip("(") for tok in _AGG_TOKENS if tok in b}
    if a_aggs != b_aggs:
        tags.append(f"agg:{sorted(a_aggs)}≠{sorted(b_aggs)}")

    # JOIN count
    if _join_count(a) != _join_count(b):
        tags.append(f"join:{_join_count(a)}≠{_join_count(b)}")

    # CAST mismatch (BIRD's integer-division trap)
    if ("cast(" in a) != ("cast(" in b):
        tags.append("cast")

    # GROUP BY / ORDER BY / LIMIT presence
    for kw in ("group by", "order by", "limit "):
        if (kw in a) != (kw in b):
            tags.append(kw.strip().replace(" ", "_"))

    # LIKE vs equals
    if (" like " in a) != (" like " in b):
        tags.append("like_vs_eq")

    # CASE WHEN
    if ("case when" in a) != ("case when" in b):
        tags.append("case_when")

    # DISTINCT
    if ("select distinct" in a) != ("select distinct" in b):
        tags.append("distinct")

    return tags or ["other"]


# ---------- diff core ----------

def build_diff(rows_a: list[dict], rows_b: list[dict]) -> dict:
    a_by_qid = {r["question_id"]: r for r in rows_a}
    b_by_qid = {r["question_id"]: r for r in rows_b}
    common_qids = sorted(set(a_by_qid) & set(b_by_qid))

    if len(common_qids) != len(rows_a) or len(common_qids) != len(rows_b):
        print(f"[diff] WARNING: question_id sets differ "
              f"(a={len(rows_a)}, b={len(rows_b)}, common={len(common_qids)})", file=sys.stderr)

    transitions = {"broke_it": [], "fixed_it": [], "stayed_correct": [], "stayed_wrong": []}
    for qid in common_qids:
        a, b = a_by_qid[qid], b_by_qid[qid]
        ac, bc = a["status"] == "correct", b["status"] == "correct"
        if ac and not bc:
            transitions["broke_it"].append((a, b))
        elif not ac and bc:
            transitions["fixed_it"].append((a, b))
        elif ac and bc:
            transitions["stayed_correct"].append((a, b))
        else:
            transitions["stayed_wrong"].append((a, b))

    return {"common_qids": common_qids, "transitions": transitions,
            "a_correct": sum(1 for q in common_qids if a_by_qid[q]["status"] == "correct"),
            "b_correct": sum(1 for q in common_qids if b_by_qid[q]["status"] == "correct")}


# ---------- formatting ----------

def _wrap(s: str, width: int = 110) -> str:
    s = (s or "").replace("\r", "").strip()
    if len(s) <= width:
        return s
    return s[: width - 1] + "…"


def print_summary(diff: dict, name_a: str, name_b: str, args: argparse.Namespace) -> None:
    n = len(diff["common_qids"])
    a_ex = diff["a_correct"] / n if n else 0.0
    b_ex = diff["b_correct"] / n if n else 0.0
    print(f"\n=== {name_a}  vs  {name_b}  (n={n}) ===")
    print(f"  EX(A) = {a_ex:.4f}  ({diff['a_correct']})")
    print(f"  EX(B) = {b_ex:.4f}  ({diff['b_correct']})")
    print(f"  delta = {b_ex - a_ex:+.4f} ({diff['b_correct'] - diff['a_correct']:+d} questions)")

    t = diff["transitions"]
    print(f"\n  Transition matrix:")
    for k in ("broke_it", "fixed_it", "stayed_correct", "stayed_wrong"):
        print(f"    {k:<18s} {len(t[k]):>5d}")

    # Same-SQL vs different-SQL on flips
    def same_diff(rows):
        same = sum(1 for a, b in rows if (a.get("predicted_sql") or "").strip() == (b.get("predicted_sql") or "").strip())
        return same, len(rows) - same
    bs, bd = same_diff(t["broke_it"])
    fs, fd = same_diff(t["fixed_it"])
    print(f"\n  broke_it  same-SQL: {bs}   different-SQL: {bd}   "
          f"(same-SQL flips suggest eval flake or non-deterministic execution)")
    print(f"  fixed_it  same-SQL: {fs}   different-SQL: {fd}")


def print_breakdown(diff: dict, key: str, label: str) -> None:
    """Net (fixed − broke) by some attribute on the A row."""
    t = diff["transitions"]
    by_key = defaultdict(lambda: [0, 0])
    for a, _ in t["broke_it"]:
        by_key[a.get(key) or "?"][0] += 1
    for a, _ in t["fixed_it"]:
        by_key[a.get(key) or "?"][1] += 1
    if not by_key:
        return
    print(f"\n  By {label}:")
    print(f"    {label:<28s}  broke_it  fixed_it     net")
    for k in sorted(by_key, key=lambda kk: by_key[kk][0] - by_key[kk][1]):
        b, f = by_key[k]
        if b == 0 and f == 0:
            continue
        print(f"    {str(k):<28s} {b:>8d}  {f:>8d}  {f - b:>+6d}")


def print_categorization(diff: dict, top_n: int = 10) -> None:
    """Heuristic tags on (A's predicted_sql, B's predicted_sql) for each transition."""
    t = diff["transitions"]
    print(f"\n  Categorization of broke_it transitions (which SQL features changed):")
    counter = Counter()
    for a, b in t["broke_it"]:
        for tag in categorize_change(a.get("predicted_sql") or "", b.get("predicted_sql") or ""):
            counter[tag] += 1
    for tag, n in counter.most_common(top_n):
        print(f"    {tag:<40s} {n:>4d}")
    print(f"\n  Categorization of fixed_it transitions:")
    counter = Counter()
    for a, b in t["fixed_it"]:
        for tag in categorize_change(a.get("predicted_sql") or "", b.get("predicted_sql") or ""):
            counter[tag] += 1
    for tag, n in counter.most_common(top_n):
        print(f"    {tag:<40s} {n:>4d}")


def print_samples(diff: dict, n_per_direction: int, width: int) -> None:
    t = diff["transitions"]
    for direction, rows in (("broke_it (A correct → B not)", t["broke_it"]),
                            ("fixed_it (A not correct → B correct)", t["fixed_it"])):
        if not rows:
            continue
        print(f"\n--- {direction}: showing {min(n_per_direction, len(rows))} of {len(rows)} ---")
        for a, b in rows[:n_per_direction]:
            print(f"\n  q{a['question_id']:<5d} db={a['db_id']} difficulty={a.get('difficulty') or '?'}")
            print(f"    A status: {a['status']}   B status: {b['status']}")
            tags = categorize_change(a.get("predicted_sql") or "", b.get("predicted_sql") or "")
            if tags:
                print(f"    tags: {', '.join(tags)}")
            if b.get("error"):
                print(f"    B error: {_wrap(b['error'], width - 14)}")
            print(f"    A pred: {_wrap(a.get('predicted_sql') or '', width - 12)}")
            print(f"    B pred: {_wrap(b.get('predicted_sql') or '', width - 12)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path_a", help="Local JSON or `bird-results:/...` URI for the A run")
    p.add_argument("path_b", help="Same, for the B run")
    p.add_argument("--names", nargs=2, default=None,
                   help="Override A and B labels in the report (default: derived from filenames)")
    p.add_argument("--difficulty", choices=["simple", "moderate", "challenging"],
                   default=None, help="Restrict the diff to one difficulty bucket")
    p.add_argument("--db_id", default=None, help="Restrict the diff to one db_id")
    p.add_argument("--samples", type=int, default=4,
                   help="Number of broken_it / fixed_it samples to print (each direction)")
    p.add_argument("--no-cat", action="store_true",
                   help="Skip the heuristic categorization section")
    p.add_argument("--out", default=None,
                   help="Save the structured diff (transition lists, per-key counts) to a JSON file")
    p.add_argument("--width", type=int, default=120, help="Print width for wrapped SQL")
    args = p.parse_args()

    path_a, path_b = _localize(args.path_a), _localize(args.path_b)
    payload_a, payload_b = json.loads(path_a.read_text()), json.loads(path_b.read_text())

    rows_a = payload_a.get("results", [])
    rows_b = payload_b.get("results", [])
    if args.difficulty:
        rows_a = [r for r in rows_a if r.get("difficulty") == args.difficulty]
        rows_b = [r for r in rows_b if r.get("difficulty") == args.difficulty]
    if args.db_id:
        rows_a = [r for r in rows_a if r.get("db_id") == args.db_id]
        rows_b = [r for r in rows_b if r.get("db_id") == args.db_id]

    name_a, name_b = args.names if args.names else (_short_name(path_a), _short_name(path_b))
    diff = build_diff(rows_a, rows_b)

    print_summary(diff, name_a, name_b, args)
    print_breakdown(diff, "difficulty", "difficulty")
    print_breakdown(diff, "db_id", "db_id (rows with movement only)")
    if not args.no_cat:
        print_categorization(diff)
    print_samples(diff, args.samples, args.width)

    if args.out:
        out = {
            "name_a": name_a, "name_b": name_b,
            "n": len(diff["common_qids"]),
            "a_correct": diff["a_correct"], "b_correct": diff["b_correct"],
            "transitions": {
                k: [(a["question_id"], a["status"], b["status"], a.get("db_id"), a.get("difficulty"),
                     a.get("predicted_sql"), b.get("predicted_sql"),
                     categorize_change(a.get("predicted_sql") or "", b.get("predicted_sql") or ""))
                    for a, b in v]
                for k, v in diff["transitions"].items()
            },
        }
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\n[diff] wrote {args.out}")


if __name__ == "__main__":
    main()
