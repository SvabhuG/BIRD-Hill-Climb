"""Smoke test for the voting + within-vote correction module.

Builds a tiny in-memory SQLite (mirrors scripts/smoke_test.py) and exercises:
  * is_degenerate_result on the boundary cases
  * pick_winner across the five scenarios in the spec (clean majority, tied,
    degenerate-minority, degenerate-majority, all-failed)
  * build_self_correct_user_prompt structure

No Modal, no GPU, no BIRD download. Run from the repo root:
    python -m scripts.smoke_test_voting_correction
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

# Make `bird` importable when run as `python scripts/smoke_test_voting_correction.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird.eval import _execute  # noqa: E402
from bird.voting_correction import (  # noqa: E402
    build_self_correct_user_prompt,
    is_degenerate_result,
    pick_winner,
)


def _build_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE artist (
            artist_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            country TEXT
        );
        CREATE TABLE album (
            album_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            year INTEGER,
            artist_id INTEGER,
            FOREIGN KEY (artist_id) REFERENCES artist(artist_id)
        );
        INSERT INTO artist VALUES (1, 'Radiohead', 'UK'),
                                  (2, 'Daft Punk', 'FR'),
                                  (3, 'The Beatles', 'UK');
        INSERT INTO album VALUES
            (10, 'OK Computer',     1997, 1),
            (11, 'In Rainbows',     2007, 1),
            (12, 'Discovery',       2001, 2),
            (13, 'Abbey Road',      1969, 3);
        """
    )
    conn.commit()
    conn.close()


def _check(label: str, actual, expected) -> None:
    ok = actual == expected
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {label}: got={actual!r} expected={expected!r}")
    if not ok:
        sys.exit(1)


def _assert(label: str, cond: bool, detail: str = "") -> None:
    flag = "PASS" if cond else "FAIL"
    print(f"  [{flag}] {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        sys.exit(1)


def _exec(db_path: Path, sql: str) -> tuple[list[tuple] | None, str | None]:
    """Run sql; return (rows, None) on success or (None, err) on failure."""
    try:
        return _execute(db_path, sql, timeout_s=5.0), None
    except Exception as e:  # sqlite3.OperationalError or anything else
        return None, str(e)


def _cand(sql: str, results: list[tuple] | None, error: str | None) -> dict:
    return {"sql": sql, "results": results if results is not None else [], "error": error}


def main() -> None:
    print("== is_degenerate_result ==")
    _check("empty list", is_degenerate_result([]), True)
    _check("one populated row", is_degenerate_result([(1, "x")]), False)
    _check("all-None rows", is_degenerate_result([(None, None), (None, None)]), True)
    _check("mixed None/value row", is_degenerate_result([(None, "x")]), False)
    _check("single non-None cell", is_degenerate_result([(1,)]), False)
    _check("single None cell row", is_degenerate_result([(None,)]), True)
    # Edge: zero-int is NOT None and counts as a real value.
    _check("zero-int row", is_degenerate_result([(0,)]), False)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "music.sqlite"
        _build_db(db_path)

        # Reusable result sets pulled from the real DB so hashing is identical.
        uk_artists_sql = "SELECT name FROM artist WHERE country = 'UK';"
        uk_artists_alt = "SELECT name FROM artist WHERE country = 'UK' ORDER BY artist_id DESC;"
        fr_artists_sql = "SELECT name FROM artist WHERE country = 'FR';"
        empty_sql = "SELECT name FROM artist WHERE country = 'XX';"
        broken_sql = "SELEC name FROM artist;"

        uk_rows, _ = _exec(db_path, uk_artists_sql)
        uk_rows_alt, _ = _exec(db_path, uk_artists_alt)
        fr_rows, _ = _exec(db_path, fr_artists_sql)
        empty_rows, _ = _exec(db_path, empty_sql)
        _, broken_err = _exec(db_path, broken_sql)
        assert broken_err is not None, "expected broken SQL to error"
        # Sanity: UK and UK-alt produce the same set, hash should match.
        assert sorted(uk_rows) == sorted(uk_rows_alt)

        print("\n== pick_winner: 5 correct + 3 broken ==")
        cands = [_cand(uk_artists_sql, uk_rows, None) for _ in range(5)] + [
            _cand(broken_sql, [], broken_err) for _ in range(3)
        ]
        out = pick_winner(cands)
        _check("winner_sql", out["winner_sql"], uk_artists_sql)
        _check("n_executable", out["n_executable"], 5)
        _check("vote_outcome", out["vote_outcome"], "unanimous")
        _check("winner_count", out["winner_count"], 5)
        _check("n_degenerate", out["n_degenerate"], 0)

        print("\n== pick_winner: 4 broken + 4 correct ==")
        cands = [_cand(broken_sql, [], broken_err) for _ in range(4)] + [
            _cand(uk_artists_sql, uk_rows, None) for _ in range(4)
        ]
        out = pick_winner(cands)
        _check("winner_sql", out["winner_sql"], uk_artists_sql)
        _check("n_executable", out["n_executable"], 4)
        _check("vote_outcome", out["vote_outcome"], "unanimous")
        _check("winner_count", out["winner_count"], 4)

        print("\n== pick_winner: 5 correct-empty (degenerate) + 3 correct-rows ==")
        # 5 candidates return [] (degenerate); 3 return UK rows (non-degenerate).
        # Non-degenerate is the MINORITY (3 < 5), so the rule keeps ALL — degenerate wins.
        cands = [_cand(empty_sql, empty_rows, None) for _ in range(5)] + [
            _cand(uk_artists_sql, uk_rows, None) for _ in range(3)
        ]
        out = pick_winner(cands)
        _check("winner_sql", out["winner_sql"], empty_sql)
        _check("n_executable", out["n_executable"], 8)
        _check("n_degenerate", out["n_degenerate"], 5)
        _check("degenerate_filter_applied", out["degenerate_filter_applied"], False)
        _check("vote_outcome", out["vote_outcome"], "majority")
        _check("winner_count", out["winner_count"], 5)

        print("\n== pick_winner: 5 correct-rows + 3 correct-empty ==")
        # Non-degenerate is MAJORITY (5 > 3), so degenerate is dropped, non-deg wins.
        cands = [_cand(uk_artists_sql, uk_rows, None) for _ in range(5)] + [
            _cand(empty_sql, empty_rows, None) for _ in range(3)
        ]
        out = pick_winner(cands)
        _check("winner_sql", out["winner_sql"], uk_artists_sql)
        _check("n_executable", out["n_executable"], 8)
        _check("n_degenerate", out["n_degenerate"], 3)
        _check("degenerate_filter_applied", out["degenerate_filter_applied"], True)
        _check("vote_outcome", out["vote_outcome"], "unanimous")
        # Pool is the 5 non-deg, all hash to one bucket.
        _check("winner_count", out["winner_count"], 5)

        print("\n== pick_winner: 8 broken (all_failed) ==")
        cands = [_cand(broken_sql, [], broken_err) for _ in range(8)]
        out = pick_winner(cands)
        _check("winner_sql", out["winner_sql"], broken_sql)
        _check("n_executable", out["n_executable"], 0)
        _check("vote_outcome", out["vote_outcome"], "all_failed")
        _check("fallback_used", out["fallback_used"], True)
        _check("winner_count", out["winner_count"], 0)

        print("\n== pick_winner: empty candidate list ==")
        out = pick_winner([])
        _check("winner_sql", out["winner_sql"], "")
        _check("vote_outcome", out["vote_outcome"], "all_failed")
        _check("fallback_used", out["fallback_used"], True)

        print("\n== pick_winner: SQL-string variants hashing to the same bucket ==")
        # Two different SQL strings, same result set — they should bucket together
        # and the EARLIEST orig_idx wins the SQL slot.
        cands = [
            _cand(uk_artists_alt, uk_rows_alt, None),  # idx 0
            _cand(uk_artists_sql, uk_rows, None),      # idx 1
            _cand(uk_artists_sql, uk_rows, None),      # idx 2
            _cand(fr_artists_sql, fr_rows, None),      # idx 3 — different bucket
        ]
        out = pick_winner(cands)
        _check("winner is earliest-of-modal-bucket", out["winner_sql"], uk_artists_alt)
        _check("winner_count", out["winner_count"], 3)
        _check("vote_outcome", out["vote_outcome"], "majority")

        print("\n== pick_winner: tie between two non-degenerate buckets ==")
        # 4 UK + 4 FR — equal sizes. Tie-break: bucket whose smallest orig_idx is lower.
        cands = (
            [_cand(uk_artists_sql, uk_rows, None) for _ in range(4)]
            + [_cand(fr_artists_sql, fr_rows, None) for _ in range(4)]
        )
        out = pick_winner(cands)
        _check("tie -> earliest bucket wins", out["winner_sql"], uk_artists_sql)
        _check("winner_count", out["winner_count"], 4)
        _check("vote_outcome", out["vote_outcome"], "no_majority")

    print("\n== build_self_correct_user_prompt: structure ==")
    base = "### Database schema (SQLite)\n```sql\nCREATE TABLE t(x);\n```\n\n### Question\nWhat?\n"
    bad_sql = "SELEC x FROM t"
    err = "near \"SELEC\": syntax error"
    out = build_self_correct_user_prompt(base, bad_sql, err)
    _assert("starts with original user content", out.startswith(base))
    _assert("contains failed SQL", bad_sql in out)
    _assert("contains error message", err in out)
    _assert("instructs fenced sql block", "```sql ... ```" in out)
    _assert("has Previous attempt header", "### Previous attempt" in out)

    # Empty / None inputs should not crash and should leave placeholders.
    out_empty = build_self_correct_user_prompt(base, "", "")
    _assert("placeholder for missing SQL", "(no SQL produced)" in out_empty)
    _assert("placeholder for missing error", "(no error message)" in out_empty)

    print("\nALL VOTING-CORRECTION SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
