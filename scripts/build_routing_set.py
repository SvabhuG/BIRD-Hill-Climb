"""Build the routing set for the agentic ablation.

A question routes to the agent if ANY:
  1. exec_error: greedy SQL errored or timed out
  2. degenerate result: greedy result is empty, all-NULL, or single zero/NULL row
  3. vote-share < 0.625 (i.e. < 5/8 voters agree on the modal result-set)

Inputs (on the `bird-results` Modal volume):
  - baseline-qwen3-coder-30b-a3b-instruct-dev-full.json (greedy, per-question status + predicted_sql)
  - voting-qwen3-coder-30b-a3b-instruct-dev-full.json    (per-question voting_metadata with winner_count, n_candidates)

Degenerate-result detection requires executing the greedy SQL against the BIRD DBs.
Since DBs live on the `bird-data` Modal volume, this script invokes a Modal helper
(`modal_app_agentic.classify_degenerate`) to do the executions remotely and merge
the result-set classification into the local routing set.

Outputs:
  - results/routing_set.json   {question_ids: [...], rules: {...}, breakdown: {...}}

Usage:
  python scripts/build_routing_set.py                # uses local cached input files
  python scripts/build_routing_set.py --refresh      # re-downloads from Modal volume
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = Path("/tmp/bird-routing")

BASELINE_NAME = "baseline-qwen3-coder-30b-a3b-instruct-dev-full.json"
VOTING_NAME = "voting-qwen3-coder-30b-a3b-instruct-dev-full.json"

VOTE_SHARE_THRESHOLD = 0.625  # < 5/8


def _ensure_cached(name: str, refresh: bool) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = CACHE_DIR / name
    if refresh or not local.exists():
        print(f"[routing] downloading {name} from bird-results volume...")
        subprocess.run(
            ["modal", "volume", "get", "bird-results", name, str(local), "--force"],
            check=True,
        )
    return local


def _is_degenerate(rows: list) -> bool:
    """Empty, all-NULL, or single zero/NULL row."""
    if not rows:
        return True
    # Flatten to a single cell test if every row is a single zero/NULL
    def _is_zero_or_null(v):
        if v is None:
            return True
        if isinstance(v, (int, float)) and v == 0:
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    # all-NULL: every cell in every row is None
    if all(all(c is None for c in row) for row in rows):
        return True
    # single zero/NULL row
    if len(rows) == 1 and all(_is_zero_or_null(c) for c in rows[0]):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-download input files from Modal volume")
    ap.add_argument("--degeneracy-from", type=str, default="",
                    help="path to JSON {question_id: [rows]} or {question_id: degenerate_bool}")
    ap.add_argument("--out", type=str, default=str(RESULTS_DIR / "routing_set.json"))
    args = ap.parse_args()

    baseline_path = _ensure_cached(BASELINE_NAME, args.refresh)
    voting_path = _ensure_cached(VOTING_NAME, args.refresh)

    baseline = json.loads(baseline_path.read_text())
    voting = json.loads(voting_path.read_text())

    print(f"[routing] baseline EX = {baseline['ex']:.4f}  (n={baseline['n']})")
    print(f"[routing] voting EX   = {voting['ex']:.4f}  (n={voting['n']})")

    baseline_by_qid = {r["question_id"]: r for r in baseline["results"]}
    voting_by_qid = {r["question_id"]: r for r in voting["results"]}

    # ---- Rule 1: exec_error ----
    exec_error_qids: set[int] = set()
    for qid, r in baseline_by_qid.items():
        if r["status"] in ("exec_error", "timeout", "empty"):
            exec_error_qids.add(qid)

    # ---- Rule 3: low vote-share ----
    low_vote_qids: set[int] = set()
    for qid, r in voting_by_qid.items():
        md = r.get("voting_metadata", {}) or {}
        wc = md.get("winner_count")
        nc = md.get("n_candidates")
        if not nc or wc is None:
            continue
        share = wc / nc
        if share < VOTE_SHARE_THRESHOLD:
            low_vote_qids.add(qid)

    # ---- Rule 2: degenerate result ----
    # Optional: load from a sidecar produced by the Modal helper.
    degenerate_qids: set[int] = set()
    if args.degeneracy_from:
        deg_path = Path(args.degeneracy_from)
        if deg_path.exists():
            deg_payload = json.loads(deg_path.read_text())
            # Accept either { "qid": bool } or { "qid": rows }
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
        else:
            print(f"[routing] WARNING: --degeneracy-from {deg_path} not found; degenerate rule disabled")
    else:
        print("[routing] no --degeneracy-from supplied; "
              "degenerate-result rule will be empty unless populated by Modal helper")

    routed = exec_error_qids | degenerate_qids | low_vote_qids

    # ---- Baseline correctness on the routed set ----
    n_routed = len(routed)
    n_routed_correct = sum(
        1 for qid in routed
        if baseline_by_qid.get(qid, {}).get("status") == "correct"
    )
    n_routed_wrong = n_routed - n_routed_correct

    # Per-rule breakdown
    only_exec = exec_error_qids - degenerate_qids - low_vote_qids
    only_deg = degenerate_qids - exec_error_qids - low_vote_qids
    only_lvs = low_vote_qids - exec_error_qids - degenerate_qids
    overlap_exec_lvs = exec_error_qids & low_vote_qids
    overlap_deg_lvs = degenerate_qids & low_vote_qids
    overlap_exec_deg = exec_error_qids & degenerate_qids
    overlap_all = exec_error_qids & degenerate_qids & low_vote_qids

    breakdown = {
        "n_exec_error": len(exec_error_qids),
        "n_degenerate": len(degenerate_qids),
        "n_low_vote_share": len(low_vote_qids),
        "n_only_exec_error": len(only_exec),
        "n_only_degenerate": len(only_deg),
        "n_only_low_vote_share": len(only_lvs),
        "n_overlap_exec_lvs": len(overlap_exec_lvs),
        "n_overlap_deg_lvs": len(overlap_deg_lvs),
        "n_overlap_exec_deg": len(overlap_exec_deg),
        "n_overlap_all_three": len(overlap_all),
    }

    payload = {
        "vote_share_threshold": VOTE_SHARE_THRESHOLD,
        "n_routed": n_routed,
        "n_routed_baseline_correct": n_routed_correct,
        "n_routed_baseline_wrong": n_routed_wrong,
        "baseline_routed_ex": n_routed_correct / n_routed if n_routed else 0.0,
        "breakdown": breakdown,
        "question_ids": sorted(routed),
        "rules": {
            "exec_error": sorted(exec_error_qids),
            "degenerate": sorted(degenerate_qids),
            "low_vote_share": sorted(low_vote_qids),
        },
        "inputs": {
            "baseline": BASELINE_NAME,
            "voting": VOTING_NAME,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print()
    print(f"[routing] === SUMMARY ===")
    print(f"[routing] n_routed                 = {n_routed}")
    print(f"[routing] n_routed_baseline_correct = {n_routed_correct}")
    print(f"[routing] n_routed_baseline_wrong   = {n_routed_wrong}")
    print(f"[routing] baseline EX on routed     = {n_routed_correct / max(n_routed,1):.4f}")
    print(f"[routing] breakdown                 = {json.dumps(breakdown, indent=2)}")
    print(f"[routing] wrote {out_path}")

    if n_routed < 50:
        print(f"[routing] WARNING: routed set < 50 questions ({n_routed}); aborting downstream")
        sys.exit(2)


if __name__ == "__main__":
    main()
