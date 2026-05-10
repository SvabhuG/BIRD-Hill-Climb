"""Stacked voting+correction: confidence-escalation pipeline (B + R2 + R3).

Composes three changes on top of the baseline `voting_correction` module:

  B  — confidence-escalation routing. Run greedy correction first; only escalate
       to an n=8 voting+correction pool when greedy errored or returned a
       degenerate result. Most questions stop at the cheap path.

  R2 — greedy is candidate-0 in the escalated pool and gets a tie-break bonus.
       Strict-override variant: a non-greedy bucket only displaces greedy's
       bucket if it ranks strictly higher on the (original_valid, total_size)
       lexicographic key — pure size ties go to greedy.

  R3 — within-vote retries are tagged `rescued=True`. Rescued candidates can't
       outvote originally-valid ones: ranking is by (original_valid DESC,
       total_size DESC, ...). A bucket with more rescues but fewer originals
       loses to a bucket with more originals.

Bucket rank key (lexicographic, smaller-is-better in the tuple form):
    1. -original_valid   (R3: more non-rescued candidates first)
    2. -total_size       (otherwise the largest bucket)
    3. NOT_contains_greedy   (R2: greedy breaks ties)
    4. earliest_orig_idx     (deterministic final tie-break)

Inside the winning bucket, if greedy is present, greedy's SQL string wins
(the T=0.0 emission is more reproducible than a T=0.7 sample that happened
to hash the same).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .voting import canonicalize_result
from .voting_correction import is_degenerate_result


def should_escalate(greedy_result: dict) -> bool:
    """Decide whether to escalate from greedy correction to the diversity pool.

    Greedy result is the dict shape produced by the candidate executor:
    {sql, results, error}. We escalate iff:
      - greedy errored after its retry (still error != None), OR
      - greedy executed but returned a degenerate result (empty rows or
        all-None cells across all rows).

    A non-degenerate executable greedy is the cheap path — return it as-is.
    """
    if greedy_result.get("error") is not None:
        return True
    return is_degenerate_result(greedy_result.get("results") or [])


@dataclass(frozen=True)
class StackedOutcome:
    winner_sql: str
    vote_outcome: str               # "unanimous" | "majority" | "no_majority" | "all_failed"
    n_candidates: int
    n_executable_pre_retry: int
    n_executable_post_retry: int
    n_degenerate: int
    n_rescued: int
    n_retried: int
    winning_bucket_size: int
    greedy_bucket_size: int
    kept_greedy: bool
    fallback_used: bool
    degenerate_filter_applied: bool

    def to_dict(self) -> dict:
        return {
            "winner_sql": self.winner_sql,
            "vote_outcome": self.vote_outcome,
            "n_candidates": self.n_candidates,
            "n_executable_pre_retry": self.n_executable_pre_retry,
            "n_executable_post_retry": self.n_executable_post_retry,
            "n_degenerate": self.n_degenerate,
            "n_rescued": self.n_rescued,
            "n_retried": self.n_retried,
            "winning_bucket_size": self.winning_bucket_size,
            "greedy_bucket_size": self.greedy_bucket_size,
            "kept_greedy": self.kept_greedy,
            "fallback_used": self.fallback_used,
            "degenerate_filter_applied": self.degenerate_filter_applied,
        }


def _classify_outcome(winner_count: int, n_pool: int) -> str:
    if n_pool == 0:
        return "all_failed"
    if winner_count == n_pool:
        return "unanimous"
    if winner_count > n_pool / 2:
        return "majority"
    return "no_majority"


def pick_winner_stacked(
    candidates: Sequence[dict],
    *,
    prefer_nondegenerate: bool = True,
) -> dict:
    """Pick a winner from {sql, results, error, rescued, is_greedy} dicts.

    Extends `voting_correction.pick_winner` with R2 (greedy anchor) and R3
    (rescued tie-break).

    Required candidate fields:
      - sql:      str
      - results:  Sequence[tuple]  (row-capped projection is fine)
      - error:    str | None
      - rescued:  bool             (True iff produced by within-vote retry)
      - is_greedy: bool            (True for the greedy correction anchor)

    Returns: StackedOutcome.to_dict().
    """
    n_candidates = len(candidates)
    if n_candidates == 0:
        return StackedOutcome(
            winner_sql="", vote_outcome="all_failed",
            n_candidates=0, n_executable_pre_retry=0, n_executable_post_retry=0,
            n_degenerate=0, n_rescued=0, n_retried=0,
            winning_bucket_size=0, greedy_bucket_size=0,
            kept_greedy=False, fallback_used=True, degenerate_filter_applied=False,
        ).to_dict()

    # Telemetry counters scanned once over the candidate list.
    n_rescued = sum(1 for c in candidates if c.get("rescued"))
    # n_retried == n_rescued for our pipeline shape: every retried cell either
    # comes back rescued (success) or is left in place with rescued=False kept.
    # Caller can override n_retried via telemetry on the orchestrator side; we
    # report the rescued count here as a conservative proxy.
    n_retried = n_rescued

    runs: list[tuple[int, dict]] = [
        (i, c) for i, c in enumerate(candidates) if c.get("error") is None
    ]
    n_executable_post = len(runs)
    n_executable_pre = n_executable_post - n_rescued

    if not runs:
        # Everyone failed — first non-empty SQL as the fallback.
        first_sql = ""
        for c in candidates:
            s = (c.get("sql") or "").strip()
            if s:
                first_sql = c["sql"]
                break
        return StackedOutcome(
            winner_sql=first_sql or (candidates[0].get("sql") or ""),
            vote_outcome="all_failed",
            n_candidates=n_candidates,
            n_executable_pre_retry=n_executable_pre,
            n_executable_post_retry=0,
            n_degenerate=0, n_rescued=n_rescued, n_retried=n_retried,
            winning_bucket_size=0, greedy_bucket_size=0,
            kept_greedy=False, fallback_used=True, degenerate_filter_applied=False,
        ).to_dict()

    non_deg = [(i, c) for i, c in runs if not is_degenerate_result(c["results"])]
    deg = [(i, c) for i, c in runs if is_degenerate_result(c["results"])]
    n_degenerate = len(deg)

    # Same degenerate-filter rule as `voting_correction.pick_winner`: only drop
    # degenerates when non-degenerate are the strict majority.
    degenerate_filter_applied = False
    if prefer_nondegenerate and len(non_deg) > len(deg):
        pool = non_deg
        degenerate_filter_applied = True
    else:
        pool = runs

    # Bucket by canonicalized result hash. Keep candidate dicts so we can later
    # interrogate {is_greedy, rescued} per bucket without a second pass.
    buckets: dict[bytes, list[tuple[int, dict]]] = {}
    for orig_idx, c in pool:
        h = canonicalize_result(c["results"])
        buckets.setdefault(h, []).append((orig_idx, c))

    # Locate the greedy bucket (if greedy is present in the pool at all).
    greedy_hash: bytes | None = None
    for h, members in buckets.items():
        if any(c.get("is_greedy") for _, c in members):
            greedy_hash = h
            break
    greedy_bucket_size = len(buckets[greedy_hash]) if greedy_hash is not None else 0

    # Bucket rank key (smaller is better in the tuple form):
    #   1. -original_valid          (R3 — rescued can't outvote originals)
    #   2. -total_size              (else largest bucket wins)
    #   3. NOT_contains_greedy      (R2 — greedy wins ties, encoded as 0/1)
    #   4. min orig_idx             (deterministic final tie-break)
    def _sort_key(item: tuple[bytes, list[tuple[int, dict]]]):
        h, members = item
        size = len(members)
        original_valid = sum(1 for _, c in members if not c.get("rescued"))
        not_greedy = 0 if any(c.get("is_greedy") for _, c in members) else 1
        first_orig = min(oi for oi, _ in members)
        return (-original_valid, -size, not_greedy, first_orig)

    winning_hash, winning_members = min(buckets.items(), key=_sort_key)
    winning_bucket_size = len(winning_members)
    kept_greedy = (greedy_hash is not None) and (winning_hash == greedy_hash)

    # Within the winning bucket, the candidate with the smallest orig_idx wins.
    # If greedy is in the bucket, prefer greedy's SQL (it's the anchor — the
    # SQL the model emitted at T=0.0 is more reproducible than a T=0.7 sample
    # that happened to hash the same).
    greedy_in_bucket = [(oi, c) for oi, c in winning_members if c.get("is_greedy")]
    if greedy_in_bucket:
        winner_orig_idx, winner_cand = min(greedy_in_bucket, key=lambda t: t[0])
    else:
        winner_orig_idx, winner_cand = min(winning_members, key=lambda t: t[0])

    return StackedOutcome(
        winner_sql=winner_cand["sql"],
        vote_outcome=_classify_outcome(winning_bucket_size, len(pool)),
        n_candidates=n_candidates,
        n_executable_pre_retry=n_executable_pre,
        n_executable_post_retry=n_executable_post,
        n_degenerate=n_degenerate,
        n_rescued=n_rescued,
        n_retried=n_retried,
        winning_bucket_size=winning_bucket_size,
        greedy_bucket_size=greedy_bucket_size,
        kept_greedy=kept_greedy,
        fallback_used=False,
        degenerate_filter_applied=degenerate_filter_applied,
    ).to_dict()


__all__ = [
    "should_escalate",
    "pick_winner_stacked",
    "StackedOutcome",
]
