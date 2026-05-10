"""Composed strategy: maj@k voting with within-vote self-correction retry.

This module is the algorithm-side of the loop only — the plumbing (sample,
execute, retry, vote) lives in `modal_app.py::run_with_voting_correction`.

Why compose them: voting alone is dominated by question-level diversity (the
right answer has to be in the bag at least once); correction alone is
dominated by question-level fixability (the model has to know the answer once
the error is in front of it). Combining gives more shots on goal AND a chance
to rescue exec_error votes back into the bucket — without re-rendering the
whole schema each retry, which is what `bird/correction.py` does.

Refinement: we prefer non-degenerate result-sets (empty rows / all-NULL rows)
ONLY when non-degenerate candidates are the strict majority. If degenerate
candidates are >= non-degenerate ones, the answer is plausibly genuinely empty
and we vote across the full executable pool. This avoids over-suppressing
correct empty-set answers (a real failure mode on BIRD's "find X with no Y"
questions).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .voting import canonicalize_result


def is_degenerate_result(results: Sequence[tuple]) -> bool:
    """Empty rows OR all-None cells across all rows.

    Degenerate result sets are the most common false-positive in voting:
    a syntactically-valid but semantically-broken query (wrong join, missing
    WHERE clause, typo'd literal) often returns 0 rows. If 4/8 candidates do
    this for the same wrong reason, naive voting picks them.
    """
    if not results:
        return True
    for row in results:
        for v in row:
            if v is not None:
                return False
    return True


@dataclass(frozen=True)
class VoteCorrectionOutcome:
    winner_sql: str
    vote_outcome: str          # "unanimous" | "majority" | "no_majority" | "all_failed"
    n_candidates: int
    n_executable: int
    n_degenerate: int
    n_distinct_results: int
    winner_count: int
    fallback_used: bool
    degenerate_filter_applied: bool

    def to_dict(self) -> dict:
        return {
            "winner_sql": self.winner_sql,
            "vote_outcome": self.vote_outcome,
            "n_candidates": self.n_candidates,
            "n_executable": self.n_executable,
            "n_degenerate": self.n_degenerate,
            "n_distinct_results": self.n_distinct_results,
            "winner_count": self.winner_count,
            "fallback_used": self.fallback_used,
            "degenerate_filter_applied": self.degenerate_filter_applied,
        }


def _classify_outcome(winner_count: int, n_executable: int) -> str:
    if n_executable == 0:
        return "all_failed"
    if winner_count == n_executable:
        return "unanimous"
    if winner_count > n_executable / 2:
        return "majority"
    return "no_majority"


def pick_winner(candidates: Sequence[dict], prefer_nondegenerate: bool = True) -> dict:
    """Pick the SQL whose result-set is the modal hash among executable candidates.

    `candidates` is a list of dicts with keys {sql, results, error}. `results`
    is the raw row list from `bird.eval._execute` (or a row-capped projection
    of it — the cap doesn't affect the hash since canonicalization sorts);
    `error` is None on success and a string message otherwise.

    Returns a dict (see VoteCorrectionOutcome.to_dict). Tie-break is
    deterministic: earliest original index whose result hashes to the winning
    bucket wins.
    """
    n_candidates = len(candidates)
    if n_candidates == 0:
        return VoteCorrectionOutcome(
            winner_sql="", vote_outcome="all_failed",
            n_candidates=0, n_executable=0, n_degenerate=0,
            n_distinct_results=0, winner_count=0,
            fallback_used=True, degenerate_filter_applied=False,
        ).to_dict()

    runs: list[tuple[int, dict]] = [
        (i, c) for i, c in enumerate(candidates) if c.get("error") is None
    ]
    n_executable = len(runs)

    if not runs:
        # Everyone failed — return the first non-empty SQL as a string we
        # couldn't run is no worse than any other string we couldn't run.
        first_sql = ""
        for c in candidates:
            s = (c.get("sql") or "").strip()
            if s:
                first_sql = c["sql"]
                break
        return VoteCorrectionOutcome(
            winner_sql=first_sql or (candidates[0].get("sql") or ""),
            vote_outcome="all_failed",
            n_candidates=n_candidates, n_executable=0, n_degenerate=0,
            n_distinct_results=0, winner_count=0,
            fallback_used=True, degenerate_filter_applied=False,
        ).to_dict()

    non_deg = [(i, c) for i, c in runs if not is_degenerate_result(c["results"])]
    deg = [(i, c) for i, c in runs if is_degenerate_result(c["results"])]
    n_degenerate = len(deg)

    # Drop degenerate candidates ONLY if non-degenerate is the strict majority.
    # Otherwise the empty answer might be the genuine truth — vote across all.
    degenerate_filter_applied = False
    if prefer_nondegenerate and len(non_deg) > len(deg):
        pool = non_deg
        degenerate_filter_applied = True
    else:
        pool = runs

    # Hash result sets; group by hash; preserve the earliest original index per
    # bucket for deterministic tie-break.
    buckets: dict[bytes, list[tuple[int, dict]]] = {}
    for orig_idx, c in pool:
        h = canonicalize_result(c["results"])
        buckets.setdefault(h, []).append((orig_idx, c))

    # Largest bucket wins; ties go to whichever bucket has the smallest min orig_idx.
    def _key(item: tuple[bytes, list[tuple[int, dict]]]):
        h, members = item
        return (-len(members), min(oi for oi, _ in members))

    winning_hash, winning_members = min(buckets.items(), key=_key)
    winner_count = len(winning_members)
    # Within the winning bucket, the candidate with the smallest orig_idx wins.
    winner_orig_idx, winner_cand = min(winning_members, key=lambda t: t[0])

    return VoteCorrectionOutcome(
        winner_sql=winner_cand["sql"],
        vote_outcome=_classify_outcome(winner_count, len(pool)),
        n_candidates=n_candidates,
        n_executable=n_executable,
        n_degenerate=n_degenerate,
        n_distinct_results=len(buckets),
        winner_count=winner_count,
        fallback_used=False,
        degenerate_filter_applied=degenerate_filter_applied,
    ).to_dict()


_SELF_CORRECT_TEMPLATE = """\

### Previous attempt
```sql
{failed_sql}
```

Execution error:
{error}

Fix the SQL. Output the corrected query in a fenced ```sql ... ``` block.
"""


def build_self_correct_user_prompt(
    original_user_content: str, failed_sql: str, error: str
) -> str:
    """Append the failed attempt + error to the original user prompt.

    No full re-render. The model already saw the schema 5 seconds ago in the
    same chat session — re-shipping the DDL would burn ~80% of the prompt
    budget on tokens the model has already conditioned on. We only append the
    new diagnostic signal.
    """
    failed_block = (failed_sql or "").strip() or "(no SQL produced)"
    error_block = (error or "").strip() or "(no error message)"
    return original_user_content + _SELF_CORRECT_TEMPLATE.format(
        failed_sql=failed_block, error=error_block,
    )


__all__ = [
    "is_degenerate_result",
    "pick_winner",
    "build_self_correct_user_prompt",
    "VoteCorrectionOutcome",
]
