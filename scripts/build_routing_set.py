"""Build the routing set for the agentic ablation.

A question routes to the agent if ANY of the rules below fires:
  1. exec_error: greedy SQL errored or timed out
  2. degenerate result: greedy result is empty, all-NULL, or single zero/NULL row
  3. vote-share < `VOTE_SHARE_THRESHOLD` (v2: 6/8, v1: 5/8)
  4. cross-temperature disagreement: T=0.0 vs T=0.7 single-sample produce
     different result-sets (sourced from a Modal-helper sidecar JSON)
  5. hint-column-not-in-SQL: BIRD evidence/hint names a specific column that
     does not appear in the greedy SQL (heuristic from hint text).

Inputs (on the `bird-results` Modal volume):
  - baseline-qwen3-coder-30b-a3b-instruct-dev-full.json (greedy)
  - voting-qwen3-coder-30b-a3b-instruct-dev-full.json   (per-question voting_metadata)
  - degenerate-flags-...-dev.json                       (Modal helper output, rule 2)
  - t07-disagreement-...-dev.json                       (Modal helper output, rule 4)

Also reads BIRD dev.json off the bird-data volume (or local cache) to get the
`evidence` text and `question` for rule 5.

Outputs:
  - results/routing_set.json   {question_ids: [...], rules: {...}, breakdown: {...}}

Usage:
  python scripts/build_routing_set.py                            # uses local cached input files
  python scripts/build_routing_set.py --refresh                  # re-downloads from Modal volume
  python scripts/build_routing_set.py --vote-share-threshold 0.75
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = Path("/tmp/bird-routing")

BASELINE_NAME = "baseline-qwen3-coder-30b-a3b-instruct-dev-full.json"
VOTING_NAME = "voting-qwen3-coder-30b-a3b-instruct-dev-full.json"
DEGEN_NAME = "degenerate-flags-qwen3-coder-30b-a3b-instruct-dev.json"
T07_DIFF_NAME = "t07-disagreement-qwen3-coder-30b-a3b-instruct-dev.json"
DEV_JSON_NAME = "dev/dev.json"   # on the bird-data volume

# v2 default: 0.75 (i.e. < 6/8). v1 used 0.625 (< 5/8).
VOTE_SHARE_THRESHOLD = 0.75


def _ensure_cached(volume: str, name: str, refresh: bool) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = name.replace("/", "__")
    local = CACHE_DIR / f"{volume}__{safe}"
    if refresh or not local.exists():
        print(f"[routing] downloading {volume}:{name} -> {local}")
        subprocess.run(
            ["modal", "volume", "get", volume, name, str(local), "--force"],
            check=True,
        )
    return local


def _is_degenerate(rows: list) -> bool:
    """Empty, all-NULL, or single zero/NULL row."""
    if not rows:
        return True

    def _is_zero_or_null(v):
        if v is None:
            return True
        if isinstance(v, (int, float)) and v == 0:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    if all(all(c is None for c in row) for row in rows):
        return True
    if len(rows) == 1 and all(_is_zero_or_null(c) for c in rows[0]):
        return True
    return False


# ----- Rule 5: hint-column-not-in-SQL ---------------------------------------

# Backticked column-like substrings: `Free Meal Count`
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
# CamelCase / PascalCase / snake_case tokens of 4+ chars used as column names:
#   NCESDist, total_amount, FreeMealCount
_CAMEL_RE = re.compile(r"\b([A-Z][a-zA-Z]{3,}(?:[_][a-zA-Z]+)*)\b")
_SNAKE_RE = re.compile(r"\b([a-z]+(?:_[a-z]+)+)\b")
# Trivial words we'd otherwise match but mean nothing.
_STOPWORDS = {
    "select", "from", "where", "group", "order", "limit", "having", "join",
    "inner", "outer", "left", "right", "union", "case", "when", "then",
    "else", "end", "null", "true", "false", "and", "or", "not", "exists",
    "between", "like", "asc", "desc", "count", "sum", "avg", "min", "max",
    "with", "the", "for", "all", "any", "some", "this", "that",
}


def _normalize(s: str) -> str:
    """Lowercase + collapse non-alnum so 'Free Meal Count' == 'freemealcount'."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _extract_hint_columns(hint: str) -> list[str]:
    """Pull HIGH-PRECISION column names out of the hint string.

    Initial v2 attempt also included CamelCase / snake_case tokens, but on
    BIRD-dev that fired on 377/1534 questions (50.7% baseline-correct on the
    routed slice — barely better than the full-dev baseline). Restricting to
    backticked names only gives ~53 candidate hints, almost all of which name
    a real column. The reduced recall is the right tradeoff: rule 5 is meant
    to be high-precision, not exhaustive.
    """
    if not hint:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for c in _BACKTICK_RE.findall(hint):
        n = _normalize(c)
        if len(n) < 4:
            continue
        if n in seen:
            continue
        seen.add(n)
        out.append(c)
    return out


def _hint_column_missing(hint: str, predicted_sql: str) -> bool:
    """True if hint mentions a column that doesn't appear in the SQL."""
    cols = _extract_hint_columns(hint)
    if not cols:
        return False
    sql_norm = _normalize(predicted_sql)
    for c in cols:
        if _normalize(c) not in sql_norm:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-download input files from Modal volumes")
    ap.add_argument("--baseline", type=str, default=BASELINE_NAME,
                    help="baseline filename on bird-results volume")
    ap.add_argument("--voting", type=str, default=VOTING_NAME,
                    help="voting filename on bird-results volume")
    ap.add_argument("--degenerate-name", type=str, default=DEGEN_NAME,
                    help="degenerate-flags filename on bird-results volume")
    ap.add_argument("--t07-name", type=str, default=T07_DIFF_NAME,
                    help="T=0.7 disagreement filename on bird-results volume")
    ap.add_argument("--degeneracy-from", type=str, default="",
                    help="path to a {qid: bool} JSON; default: fetch DEGEN_NAME from bird-results")
    ap.add_argument("--t07-disagreement-from", type=str, default="",
                    help="path to a {qid: bool} JSON; default: fetch T07_DIFF_NAME from bird-results")
    ap.add_argument("--vote-share-threshold", type=float, default=VOTE_SHARE_THRESHOLD,
                    help="route if winner_count / n_candidates < this (default 0.75 = <6/8)")
    ap.add_argument("--dev-json-path", type=str, default="",
                    help="local BIRD dev.json (for hint text); default: fetch via Modal bird-data")
    ap.add_argument("--disable-rule-3", action="store_true",
                    help="skip vote-share rule (rule 3); use when no voting file is available")
    ap.add_argument("--disable-rule-4", action="store_true",
                    help="skip T=0.7 disagreement (rule 4); v1 has no rule 4")
    ap.add_argument("--disable-rule-5", action="store_true",
                    help="skip hint-column-not-in-SQL (rule 5); v1 has no rule 5")
    ap.add_argument("--v1", action="store_true",
                    help="shortcut: v1 routing (rules 1+2+3, threshold 0.625 / <5/8)")
    ap.add_argument("--out", type=str, default=str(RESULTS_DIR / "routing_set.json"))
    args = ap.parse_args()

    if args.v1:
        # v1: only rules 1 (exec_error), 2 (degenerate), 3 (vote-share < 5/8).
        args.vote_share_threshold = 0.625
        args.disable_rule_4 = True
        args.disable_rule_5 = True

    # Override globals from arg names.
    baseline_name = args.baseline
    voting_name = args.voting
    degen_name = args.degenerate_name
    t07_name = args.t07_name

    # ----- Baseline + voting -----
    baseline_path = _ensure_cached("bird-results", baseline_name, args.refresh)
    baseline = json.loads(baseline_path.read_text())
    if args.disable_rule_3:
        print("[routing] rule 3 (low vote-share) disabled by flag")
        voting = {"results": [], "n": 0, "ex": None}
    else:
        voting_path = _ensure_cached("bird-results", voting_name, args.refresh)
        voting = json.loads(voting_path.read_text())

    print(f"[routing] baseline EX = {baseline['ex']:.4f}  (n={baseline['n']})")
    voting_ex = voting.get('ex')
    if voting_ex is not None:
        print(f"[routing] voting EX   = {voting_ex:.4f}  (n={voting['n']})")
    else:
        print(f"[routing] voting EX   = (not scored; n={voting.get('n', '?')})")
    print(f"[routing] vote-share threshold = {args.vote_share_threshold} "
          f"(route if winner_count/n_candidates < this)")

    baseline_by_qid = {r["question_id"]: r for r in baseline["results"]}
    voting_by_qid = {r["question_id"]: r for r in voting["results"]}

    # ----- Rule 1: exec_error -----
    exec_error_qids: set[int] = set()
    for qid, r in baseline_by_qid.items():
        if r["status"] in ("exec_error", "timeout", "empty"):
            exec_error_qids.add(qid)

    # ----- Rule 3: low vote-share -----
    low_vote_qids: set[int] = set()
    for qid, r in voting_by_qid.items():
        md = r.get("voting_metadata", {}) or {}
        wc = md.get("winner_count")
        nc = md.get("n_candidates")
        if not nc or wc is None:
            continue
        share = wc / nc
        if share < args.vote_share_threshold:
            low_vote_qids.add(qid)

    # ----- Rule 2: degenerate result -----
    deg_path_local = args.degeneracy_from
    if not deg_path_local:
        deg_path_local = str(_ensure_cached("bird-results", degen_name, args.refresh))
    degenerate_qids: set[int] = set()
    deg_payload = json.loads(Path(deg_path_local).read_text())
    for k, v in deg_payload.items():
        try:
            qid = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, bool):
            if v:
                degenerate_qids.add(qid)
        elif isinstance(v, list):
            if _is_degenerate(v):
                degenerate_qids.add(qid)

    # ----- Rule 4: T=0.7 disagreement -----
    t07_qids: set[int] = set()
    t07_path_local = args.t07_disagreement_from
    if args.disable_rule_4:
        print("[routing] rule 4 (T=0.7 disagreement) disabled by flag")
        t07_path_local = ""
    elif not t07_path_local:
        candidate = CACHE_DIR / f"bird-results__{t07_name}"
        # If the file isn't on the volume yet (T=0.7 hasn't been generated),
        # skip rule 4 cleanly so the v2 ablation still runs partially.
        try:
            t07_path_local = str(_ensure_cached("bird-results", t07_name, args.refresh))
        except subprocess.CalledProcessError:
            print(f"[routing] WARNING: {t07_name} not on bird-results volume; "
                  "rule 4 (T=0.7 disagreement) will be EMPTY")
            t07_path_local = ""
    if t07_path_local and Path(t07_path_local).exists():
        t07_payload = json.loads(Path(t07_path_local).read_text())
        for k, v in t07_payload.items():
            try:
                qid = int(k)
            except (TypeError, ValueError):
                continue
            if bool(v):
                t07_qids.add(qid)

    # ----- Rule 5: hint-column-not-in-SQL -----
    hint_miss_qids: set[int] = set()
    dev_path = args.dev_json_path
    if args.disable_rule_5:
        print("[routing] rule 5 (hint-column-not-in-SQL) disabled by flag")
        dev_path = ""
    elif not dev_path:
        try:
            dev_path = str(_ensure_cached("bird-data", DEV_JSON_NAME, args.refresh))
        except subprocess.CalledProcessError:
            print(f"[routing] WARNING: couldn't fetch {DEV_JSON_NAME} from bird-data; "
                  "rule 5 (hint-column-not-in-SQL) will be EMPTY")
            dev_path = ""
    if dev_path and Path(dev_path).exists():
        dev_list = json.loads(Path(dev_path).read_text())
        # BIRD dev.json is a list of dicts with fields: question_id, evidence, ...
        for item in dev_list:
            qid = item.get("question_id")
            if qid is None:
                continue
            hint = item.get("evidence") or ""
            base = baseline_by_qid.get(qid)
            if not base:
                continue
            sql = base.get("predicted_sql") or ""
            if not sql.strip():
                continue
            if _hint_column_missing(hint, sql):
                hint_miss_qids.add(int(qid))

    routed = (
        exec_error_qids | degenerate_qids | low_vote_qids | t07_qids | hint_miss_qids
    )

    n_routed = len(routed)
    n_routed_correct = sum(
        1 for qid in routed
        if baseline_by_qid.get(qid, {}).get("status") == "correct"
    )
    n_routed_wrong = n_routed - n_routed_correct

    rule_sets = {
        "exec_error": exec_error_qids,
        "degenerate": degenerate_qids,
        "low_vote_share": low_vote_qids,
        "t07_disagree": t07_qids,
        "hint_col_missing": hint_miss_qids,
    }

    breakdown: dict[str, int] = {f"n_{k}": len(v) for k, v in rule_sets.items()}
    # Only-this-rule counts (set-difference against the union of the others)
    for k, v in rule_sets.items():
        others = set()
        for kk, vv in rule_sets.items():
            if kk == k:
                continue
            others |= vv
        breakdown[f"n_only_{k}"] = len(v - others)

    # Routed-correct rate per rule (how often the baseline is already correct
    # among questions a single rule fires on — high = lots of false positives).
    per_rule_routed_correct: dict[str, dict] = {}
    for k, v in rule_sets.items():
        n = len(v)
        n_correct = sum(
            1 for qid in v
            if baseline_by_qid.get(qid, {}).get("status") == "correct"
        )
        per_rule_routed_correct[k] = {
            "n": n,
            "n_baseline_correct": n_correct,
            "baseline_ex": (n_correct / n) if n else 0.0,
        }

    payload = {
        "vote_share_threshold": args.vote_share_threshold,
        "n_routed": n_routed,
        "n_routed_baseline_correct": n_routed_correct,
        "n_routed_baseline_wrong": n_routed_wrong,
        "baseline_routed_ex": n_routed_correct / n_routed if n_routed else 0.0,
        "breakdown": breakdown,
        "per_rule_baseline_ex": per_rule_routed_correct,
        "question_ids": sorted(routed),
        "rules": {k: sorted(v) for k, v in rule_sets.items()},
        "inputs": {
            "baseline": baseline_name,
            "voting": voting_name,
            "degenerate": degen_name,
            "t07_disagreement": t07_name,
            "dev_json": DEV_JSON_NAME,
        },
        "rules_enabled": {
            "rule_1_exec_error": True,
            "rule_2_degenerate": True,
            "rule_3_low_vote_share": not args.disable_rule_3,
            "rule_4_t07_disagree": not args.disable_rule_4,
            "rule_5_hint_col_missing": not args.disable_rule_5,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print()
    print("[routing] === SUMMARY ===")
    print(f"[routing] n_routed                  = {n_routed}")
    print(f"[routing] n_routed_baseline_correct = {n_routed_correct}")
    print(f"[routing] n_routed_baseline_wrong   = {n_routed_wrong}")
    print(f"[routing] baseline EX on routed     = {n_routed_correct / max(n_routed, 1):.4f}")
    print(f"[routing] breakdown                 = {json.dumps(breakdown, indent=2)}")
    print(f"[routing] per_rule_baseline_ex      = {json.dumps(per_rule_routed_correct, indent=2)}")
    print(f"[routing] wrote {out_path}")

    if n_routed < 50:
        print(f"[routing] WARNING: routed set < 50 questions ({n_routed}); aborting downstream")
        sys.exit(2)


if __name__ == "__main__":
    main()
