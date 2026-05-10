"""Majority-vote-on-execution self-consistency for BIRD predictions.

For each question we sample n candidate SQL queries, execute each on the SQLite DB,
hash the canonical result-set (set semantics — matches `bird/eval.py`'s correctness
comparison), group candidates by hash, and return one SQL from the largest group.

Why hash on the result rather than on the SQL string? Two semantically equivalent
queries (different join order, different aliases, equivalent WHERE clauses) usually
emit identical row sets; voting on results captures that agreement, voting on
strings doesn't. Broken candidates (syntax errors / timeouts) simply don't
contribute — which is exactly what we want.

If no candidate executes successfully, we return the first candidate as a fallback;
a string we couldn't run is no worse than any other string we couldn't run.

Tie-breaking is deterministic: the first SQL in `candidates` whose result hash
belongs to the largest bucket wins. (Ties on bucket size are broken by the bucket
that *first* reached its max count, again preserving caller order.)
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .eval import _execute


# Sentinel hashes so we can group "all candidates agree on this kind of failure"
# without colliding with real result hashes. Currently unused for voting (we vote
# only on executable rows), but useful if a future variant wants to count errors.
_ERROR_PREFIX = b"\x00err:"
_TIMEOUT_PREFIX = b"\x00timeout:"


def canonicalize_result(rows: Sequence[tuple]) -> bytes:
    """Hashable canonical form of a result set.

    Sorted-tuple comparison matches BIRD's set-equality check in `eval._rows_equal`.
    Individual cells: bytes/BLOBs are repr'd, NULLs are preserved as None, and the
    full row is repr'd before sort so unhashable / type-mixed rows still produce a
    stable order. Returns a SHA-1 digest so the result is fixed-size and cheap to
    compare in a Counter.
    """
    # repr handles None, bytes, ints, floats, strings, and tuples uniformly; using
    # repr-of-row as the sort key avoids TypeError on mixed-type column ordering
    # (e.g. one row has None where another has an int).
    serialized = [repr(tuple(row)) for row in rows]
    serialized.sort()
    blob = "\n".join(serialized).encode("utf-8", errors="replace")
    return hashlib.sha1(blob).digest()


@dataclass(frozen=True)
class VoteOutcome:
    winner_sql: str
    winner_count: int
    n_candidates: int
    n_executable: int
    n_distinct_results: int
    fallback_used: bool

    def to_dict(self) -> dict:
        return {
            "winner_sql": self.winner_sql,
            "winner_count": self.winner_count,
            "n_candidates": self.n_candidates,
            "n_executable": self.n_executable,
            "n_distinct_results": self.n_distinct_results,
            "fallback_used": self.fallback_used,
        }


def vote(
    candidates: Sequence[str],
    db_path: str | Path,
    timeout_s: float = 15.0,
) -> dict:
    """Pick the SQL whose result-set is the most common among executable candidates.

    Algorithm:
      1. Filter out empty / whitespace-only candidates.
      2. Dedupe identical SQL strings — execute each unique candidate once.
      3. Hash each successful result; group candidates by hash.
      4. Winner = SQL whose hash has the highest count among executables. Ties go
         to the bucket that filled first (which is itself filled in caller order).
      5. If no candidate executes, return the first non-empty candidate (or empty
         string if there isn't one) and set `fallback_used=True`.
    """
    n_candidates = len(candidates)
    if n_candidates == 0:
        return VoteOutcome(
            winner_sql="", winner_count=0, n_candidates=0,
            n_executable=0, n_distinct_results=0, fallback_used=True,
        ).to_dict()

    # Strip + drop empties, but keep the original first non-empty for fallback.
    stripped: list[tuple[int, str]] = []  # (orig_idx, sql)
    for i, c in enumerate(candidates):
        s = (c or "").strip()
        if s:
            stripped.append((i, s))

    if not stripped:
        return VoteOutcome(
            winner_sql=(candidates[0] or "") if n_candidates else "",
            winner_count=0, n_candidates=n_candidates,
            n_executable=0, n_distinct_results=0, fallback_used=True,
        ).to_dict()

    # Dedupe identical SQL strings: each unique SQL is executed exactly once,
    # but every original index that maps to it counts toward its bucket. This is
    # the difference between O(n) and O(n^2) when models emit lots of duplicates.
    unique_to_orig: dict[str, list[int]] = {}
    for orig_idx, sql in stripped:
        unique_to_orig.setdefault(sql, []).append(orig_idx)

    # Hash bucket -> list of (orig_idx, sql) that produced it, in caller order.
    buckets: dict[bytes, list[tuple[int, str]]] = {}
    n_executable = 0
    for sql, orig_indices in unique_to_orig.items():
        try:
            rows = _execute(db_path, sql, timeout_s=timeout_s)
        except sqlite3.OperationalError:
            continue
        except Exception:
            # Any other DB error (programming, integrity, type) is treated as a
            # broken candidate — same as a syntax error.
            continue
        h = canonicalize_result(rows)
        # Each duplicate of this SQL string contributes one vote.
        n_executable += len(orig_indices)
        bucket = buckets.setdefault(h, [])
        for oi in orig_indices:
            bucket.append((oi, sql))

    if not buckets:
        # Everyone failed: fall back to the first non-empty candidate.
        first_idx, first_sql = stripped[0]
        return VoteOutcome(
            winner_sql=first_sql,
            winner_count=0,
            n_candidates=n_candidates,
            n_executable=0,
            n_distinct_results=0,
            fallback_used=True,
        ).to_dict()

    # Sort buckets by (size DESC, min-orig-idx ASC) for deterministic ties.
    def _key(item: tuple[bytes, list[tuple[int, str]]]):
        h, members = item
        size = len(members)
        first_orig = min(oi for oi, _ in members)
        return (-size, first_orig)

    winning_hash, winning_members = min(buckets.items(), key=_key)
    # Winner SQL: the candidate with the lowest original index in the winning bucket.
    winning_members_sorted = sorted(winning_members, key=lambda t: t[0])
    winner_sql = winning_members_sorted[0][1]

    return VoteOutcome(
        winner_sql=winner_sql,
        winner_count=len(winning_members),
        n_candidates=n_candidates,
        n_executable=n_executable,
        n_distinct_results=len(buckets),
        fallback_used=False,
    ).to_dict()
