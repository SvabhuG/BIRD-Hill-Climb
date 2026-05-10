"""Smoke test for the stacked voting+correction module (B + R2 + R3).

Builds a tiny in-memory SQLite (mirrors scripts/smoke_test.py) and exercises:
  * should_escalate on the four routing inputs
  * pick_winner_stacked across the five spec scenarios:
      (a) tied size, R3 (more original-valid) decides — greedy in winner
      (b) Y has more raw votes but all rescued — X wins on original-valid
      (c) Y has strict majority over greedy — Y wins
      (d) all rescued, single bucket — that bucket wins (no greedy bucket)
      (e) end-to-end routing through should_escalate + pick_winner_stacked

No Modal, no GPU, no BIRD download. Run from the repo root:
    python -m scripts.smoke_test_stacked
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

# Make `bird` importable when run as `python scripts/smoke_test_stacked.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird.eval import _execute  # noqa: E402
from bird.voting_correction_stacked import (  # noqa: E402
    pick_winner_stacked,
    should_escalate,
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
        INSERT INTO artist VALUES (1, 'Radiohead', 'UK'),
                                  (2, 'Daft Punk', 'FR'),
                                  (3, 'The Beatles', 'UK');
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
    try:
        return _execute(db_path, sql, timeout_s=5.0), None
    except Exception as e:
        return None, str(e)


def _cand(
    sql: str,
    results: list[tuple] | None,
    error: str | None,
    *,
    rescued: bool = False,
    is_greedy: bool = False,
) -> dict:
    return {
        "sql": sql,
        "results": results if results is not None else [],
        "error": error,
        "rescued": rescued,
        "is_greedy": is_greedy,
    }


def main() -> None:
    print("== should_escalate ==")
    # 1. Greedy non-degenerate, no error → don't escalate.
    _check(
        "good greedy",
        should_escalate({"results": [(1, "a")], "error": None}),
        False,
    )
    # 2. Empty rows → degenerate → escalate.
    _check(
        "degenerate empty",
        should_escalate({"results": [], "error": None}),
        True,
    )
    # 3. All-None cells → degenerate → escalate.
    _check(
        "degenerate all-None",
        should_escalate({"results": [(None, None)], "error": None}),
        True,
    )
    # 4. Exec error → escalate.
    _check(
        "exec_error",
        should_escalate({"results": None, "error": "no such column: x"}),
        True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "music.sqlite"
        _build_db(db_path)

        uk_sql = "SELECT name FROM artist WHERE country = 'UK';"
        fr_sql = "SELECT name FROM artist WHERE country = 'FR';"
        de_sql = "SELECT name FROM artist WHERE country = 'DE';"  # degenerate (empty)
        broken_sql = "SELEC name FROM artist;"

        uk_rows, _ = _exec(db_path, uk_sql)
        fr_rows, _ = _exec(db_path, fr_sql)
        empty_rows, _ = _exec(db_path, de_sql)
        _, broken_err = _exec(db_path, broken_sql)
        assert broken_err is not None

        print("\n== (a) tied size, R3 decides — greedy in higher-orig bucket ==")
        # Bucket X (FR): 4 votes — 1 original, 3 rescued
        # Bucket Y (UK): 4 votes — 3 original (greedy at idx-0), 1 rescued
        # Tie on size; R3 picks the bucket with more originals (Y, has greedy).
        cands = [
            # idx 0 — greedy, UK, original
            _cand(uk_sql, uk_rows, None, rescued=False, is_greedy=True),
            # idx 1, 2 — UK, original (T=0.7 samples)
            _cand(uk_sql, uk_rows, None),
            _cand(uk_sql, uk_rows, None),
            # idx 3 — UK, rescued
            _cand(uk_sql, uk_rows, None, rescued=True),
            # idx 4 — FR, original
            _cand(fr_sql, fr_rows, None),
            # idx 5, 6, 7 — FR, rescued
            _cand(fr_sql, fr_rows, None, rescued=True),
            _cand(fr_sql, fr_rows, None, rescued=True),
            _cand(fr_sql, fr_rows, None, rescued=True),
        ]
        out = pick_winner_stacked(cands)
        _check("winner_sql == uk_sql", out["winner_sql"], uk_sql)
        _check("winning_bucket_size", out["winning_bucket_size"], 4)
        _check("greedy_bucket_size", out["greedy_bucket_size"], 4)
        _check("kept_greedy", out["kept_greedy"], True)
        _check("n_rescued", out["n_rescued"], 4)

        print("\n== (b) Y has more raw votes but all rescued — X wins on orig-valid ==")
        # Bucket X (UK): 3 votes, all original, greedy at idx-0
        # Bucket Y (FR): 4 votes, all rescued
        # X has more originals (3 > 0) → X wins.
        cands = [
            _cand(uk_sql, uk_rows, None, is_greedy=True),  # idx 0
            _cand(uk_sql, uk_rows, None),                  # idx 1
            _cand(uk_sql, uk_rows, None),                  # idx 2
            _cand(fr_sql, fr_rows, None, rescued=True),    # idx 3
            _cand(fr_sql, fr_rows, None, rescued=True),    # idx 4
            _cand(fr_sql, fr_rows, None, rescued=True),    # idx 5
            _cand(fr_sql, fr_rows, None, rescued=True),    # idx 6
        ]
        out = pick_winner_stacked(cands)
        _check("winner_sql (X with greedy)", out["winner_sql"], uk_sql)
        _check("winning_bucket_size", out["winning_bucket_size"], 3)
        _check("greedy_bucket_size", out["greedy_bucket_size"], 3)
        _check("kept_greedy", out["kept_greedy"], True)
        _check("n_rescued", out["n_rescued"], 4)

        print("\n== (c) Y strict majority — overrides greedy ==")
        # Bucket X (UK): 3 votes (all original, greedy at idx-0)
        # Bucket Y (FR): 5 votes (all original)
        # Y > X strictly on both originals and total → Y wins.
        cands = [
            _cand(uk_sql, uk_rows, None, is_greedy=True),  # idx 0
            _cand(uk_sql, uk_rows, None),                  # idx 1
            _cand(uk_sql, uk_rows, None),                  # idx 2
            _cand(fr_sql, fr_rows, None),                  # idx 3
            _cand(fr_sql, fr_rows, None),                  # idx 4
            _cand(fr_sql, fr_rows, None),                  # idx 5
            _cand(fr_sql, fr_rows, None),                  # idx 6
            _cand(fr_sql, fr_rows, None),                  # idx 7
        ]
        out = pick_winner_stacked(cands)
        _check("winner_sql == fr_sql", out["winner_sql"], fr_sql)
        _check("winning_bucket_size", out["winning_bucket_size"], 5)
        _check("greedy_bucket_size", out["greedy_bucket_size"], 3)
        _check("kept_greedy", out["kept_greedy"], False)

        print("\n== (d) all rescued, single bucket — wins by default ==")
        # No greedy. All 8 rescued, all same bucket.
        cands = [_cand(uk_sql, uk_rows, None, rescued=True) for _ in range(8)]
        out = pick_winner_stacked(cands)
        _check("winner_sql", out["winner_sql"], uk_sql)
        _check("winning_bucket_size", out["winning_bucket_size"], 8)
        _check("greedy_bucket_size", out["greedy_bucket_size"], 0)
        _check("kept_greedy", out["kept_greedy"], False)
        _check("n_rescued", out["n_rescued"], 8)
        _check("vote_outcome", out["vote_outcome"], "unanimous")

        print("\n== (e) end-to-end routing: greedy good → no escalation ==")
        # Greedy returned uk_rows non-degenerate. should_escalate False.
        greedy_result = {"sql": uk_sql, "results": uk_rows, "error": None}
        _check("should_escalate", should_escalate(greedy_result), False)
        # Caller in this case would short-circuit and return greedy_result["sql"]
        # without calling pick_winner_stacked. We model that here.
        chosen_sql = (
            greedy_result["sql"]
            if not should_escalate(greedy_result)
            else "(would-escalate)"
        )
        _check("chosen via routing", chosen_sql, uk_sql)

        print("\n== (e2) end-to-end routing: greedy degenerate → escalate, vote-overrides-greedy ==")
        # Greedy returned [] (degenerate empty). Escalate. The diversity pool's
        # majority hashes to UK; greedy's empty result is in its own bucket.
        greedy_result = {"sql": de_sql, "results": empty_rows, "error": None}
        _check("should_escalate", should_escalate(greedy_result), True)
        cands = [
            _cand(de_sql, empty_rows, None, is_greedy=True),  # idx 0 — greedy, degenerate
            _cand(uk_sql, uk_rows, None),                     # idx 1
            _cand(uk_sql, uk_rows, None),                     # idx 2
            _cand(uk_sql, uk_rows, None),                     # idx 3
            _cand(uk_sql, uk_rows, None),                     # idx 4
            _cand(uk_sql, uk_rows, None),                     # idx 5
            _cand(fr_sql, fr_rows, None),                     # idx 6
            _cand(fr_sql, fr_rows, None),                     # idx 7
        ]
        out = pick_winner_stacked(cands)
        # Non-degenerate are 7 vs 1 degenerate → strict majority → filter applied.
        _check("degenerate_filter_applied", out["degenerate_filter_applied"], True)
        # Among non-degenerates, UK has 5, FR has 2. UK wins.
        _check("winner_sql (vote-override-greedy)", out["winner_sql"], uk_sql)
        _check("kept_greedy (greedy was filtered out)", out["kept_greedy"], False)

        print("\n== (f) tie-break sanity: greedy vs non-greedy tied on (orig, size) ==")
        # X (UK): 2 original, greedy at idx-0
        # Y (FR): 2 original, no greedy
        # Same orig, same size → R2 picks greedy bucket.
        cands = [
            _cand(uk_sql, uk_rows, None, is_greedy=True),  # idx 0
            _cand(fr_sql, fr_rows, None),                  # idx 1
            _cand(uk_sql, uk_rows, None),                  # idx 2
            _cand(fr_sql, fr_rows, None),                  # idx 3
        ]
        out = pick_winner_stacked(cands)
        _check("R2 tie-break picks greedy bucket", out["winner_sql"], uk_sql)
        _check("kept_greedy", out["kept_greedy"], True)

        print("\n== (g) all-failed pool ==")
        cands = [
            _cand(broken_sql, [], broken_err, is_greedy=True),
            _cand(broken_sql, [], broken_err),
            _cand(broken_sql, [], broken_err, rescued=True),
        ]
        out = pick_winner_stacked(cands)
        _check("winner_sql is broken (fallback)", out["winner_sql"], broken_sql)
        _check("vote_outcome", out["vote_outcome"], "all_failed")
        _check("fallback_used", out["fallback_used"], True)
        _check("n_executable_post_retry", out["n_executable_post_retry"], 0)

        print("\n== (h) empty candidate list ==")
        out = pick_winner_stacked([])
        _check("winner_sql", out["winner_sql"], "")
        _check("vote_outcome", out["vote_outcome"], "all_failed")
        _check("fallback_used", out["fallback_used"], True)

    print("\nALL STACKED-VC SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
