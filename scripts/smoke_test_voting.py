"""Smoke test for `bird.voting` — local-only, no Modal/GPU.

Builds a tiny in-temp-dir SQLite, then validates:

  * 5/8 candidates produce the same correct rows -> winner is the correct SQL,
    fallback_used=False, winner_count=5, n_executable=8 (3 broken don't count).
  * Half the candidates have syntax errors -> vote ignores them and picks from
    the executable majority.
  * Two distinct executable result-sets, each by 4 candidates -> tie broken by
    deterministic ordering (first-occurrence wins).
  * All candidates fail -> winner is candidates[0], fallback_used=True.
  * Edge cases: empty list, candidates with NULL/bytes rows, duplicate SQL strings.

Run from the repo root:
    python -m scripts.smoke_test_voting
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird.voting import canonicalize_result, vote  # noqa: E402


def _build_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE artist (
            artist_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            country TEXT
        );
        CREATE TABLE blob_t (
            id INTEGER PRIMARY KEY,
            payload BLOB,
            note TEXT
        );
        INSERT INTO artist VALUES
            (1, 'Radiohead', 'UK'),
            (2, 'Daft Punk', 'FR'),
            (3, 'The Beatles', 'UK'),
            (4, 'Sigur Ros', NULL);
        INSERT INTO blob_t VALUES
            (1, X'DEADBEEF', 'one'),
            (2, X'C0FFEE',   NULL);
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


def _check_true(label: str, cond: bool) -> None:
    flag = "PASS" if cond else "FAIL"
    print(f"  [{flag}] {label}")
    if not cond:
        sys.exit(1)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "music.sqlite"
        _build_db(db_path)

        # ------------------------------------------------------------------
        print("== canonicalize_result: order-independent ==")
        h1 = canonicalize_result([(1, "a"), (2, "b")])
        h2 = canonicalize_result([(2, "b"), (1, "a")])
        _check("set-semantic equality", h1, h2)

        print("\n== canonicalize_result: NULLs and bytes ==")
        h3 = canonicalize_result([(1, None), (2, b"\xde\xad")])
        h4 = canonicalize_result([(2, b"\xde\xad"), (1, None)])
        _check("NULL/bytes order-independent", h3, h4)
        _check_true("NULL bucket distinct from non-NULL",
                    h3 != canonicalize_result([(1, "x"), (2, b"\xde\xad")]))

        # ------------------------------------------------------------------
        print("\n== vote: 5/8 correct, 3 broken ==")
        correct = "SELECT name FROM artist WHERE country = 'UK' ORDER BY name;"
        # Same set of rows but written differently — these all hash to the
        # same canonical result and should join the winning bucket.
        correct_alt1 = "SELECT name FROM artist WHERE country = 'UK';"
        correct_alt2 = "SELECT a.name FROM artist a WHERE a.country='UK';"
        wrong = "SELECT name FROM artist WHERE country = 'FR';"
        broken = "SELEC name FROM artist;"
        candidates = [
            correct, correct_alt1, correct_alt2, correct, correct_alt1,  # 5 correct
            wrong,                                                       # 1 wrong-but-runs
            broken, broken,                                              # 2 broken
        ]
        out = vote(candidates, db_path)
        _check_true("winner produces UK-artist rows",
                    out["winner_sql"] in {correct, correct_alt1, correct_alt2})
        _check("winner_count", out["winner_count"], 5)
        _check("n_candidates", out["n_candidates"], 8)
        _check("n_executable (5 correct + 1 wrong)", out["n_executable"], 6)
        _check("n_distinct_results (correct-set + FR-set)", out["n_distinct_results"], 2)
        _check("fallback_used", out["fallback_used"], False)

        # ------------------------------------------------------------------
        print("\n== vote: half broken, executable majority wins ==")
        candidates = [
            broken, broken, broken, broken,                              # 4 broken
            correct, correct_alt1, correct_alt2, wrong,                  # 4 executable, 3 agree
        ]
        out = vote(candidates, db_path)
        _check_true("winner is one of the correct variants",
                    out["winner_sql"] in {correct, correct_alt1, correct_alt2})
        _check("winner_count", out["winner_count"], 3)
        _check("n_executable", out["n_executable"], 4)
        _check("fallback_used", out["fallback_used"], False)

        # ------------------------------------------------------------------
        print("\n== vote: tie -> first occurrence wins ==")
        # 4 candidates produce result A, 4 produce result B. The bucket whose
        # earliest member appears first in caller order should win.
        a1 = "SELECT name FROM artist WHERE country = 'UK' ORDER BY name;"  # idx 0
        b1 = "SELECT name FROM artist WHERE country = 'FR';"                # idx 1
        a2 = "SELECT name FROM artist WHERE country='UK';"
        b2 = "SELECT a.name FROM artist a WHERE a.country='FR';"
        candidates = [a1, b1, a2, b2, a1, b1, a2, b2]
        out = vote(candidates, db_path)
        _check_true("tie winner is the UK-set (first in caller order)",
                    out["winner_sql"] in {a1, a2})
        _check("tie winner_count", out["winner_count"], 4)
        _check("tie n_distinct_results", out["n_distinct_results"], 2)

        # Reverse the order — now the FR bucket should win.
        candidates_rev = [b1, a1, b2, a2, b1, a1, b2, a2]
        out_rev = vote(candidates_rev, db_path)
        _check_true("tie winner flips with caller order",
                    out_rev["winner_sql"] in {b1, b2})

        # ------------------------------------------------------------------
        print("\n== vote: all candidates fail -> fallback ==")
        candidates = [
            "SELEC bad", "SELECT * FROM no_such_table", "garbage", "DROP TABLE x",
            "WITH x AS (SELECT) SELECT * FROM x",  # syntax error
        ]
        out = vote(candidates, db_path)
        _check("fallback winner is candidates[0]", out["winner_sql"], candidates[0])
        _check("fallback winner_count", out["winner_count"], 0)
        _check("fallback n_executable", out["n_executable"], 0)
        _check("fallback n_distinct_results", out["n_distinct_results"], 0)
        _check("fallback_used", out["fallback_used"], True)

        # ------------------------------------------------------------------
        print("\n== vote: empty candidate list ==")
        out = vote([], db_path)
        _check("empty winner_sql", out["winner_sql"], "")
        _check("empty n_candidates", out["n_candidates"], 0)
        _check("empty fallback_used", out["fallback_used"], True)

        # ------------------------------------------------------------------
        print("\n== vote: only blank candidates -> fallback to first ==")
        out = vote(["", "   ", "\n\t"], db_path)
        _check("blank n_candidates", out["n_candidates"], 3)
        _check("blank n_executable", out["n_executable"], 0)
        _check("blank fallback_used", out["fallback_used"], True)

        # ------------------------------------------------------------------
        print("\n== vote: dedup identical SQL strings ==")
        # Same SQL repeated 5 times; we should execute it once but record 5 votes.
        same = "SELECT name FROM artist WHERE country = 'UK';"
        candidates = [same, same, same, same, same]
        out = vote(candidates, db_path)
        _check("dedup winner", out["winner_sql"], same)
        _check("dedup winner_count", out["winner_count"], 5)
        _check("dedup n_executable", out["n_executable"], 5)
        _check("dedup n_distinct_results", out["n_distinct_results"], 1)

        # ------------------------------------------------------------------
        print("\n== vote: NULL row handling ==")
        # The NULL country row should not produce errors during canonicalize.
        nulled_a = "SELECT country FROM artist;"
        nulled_b = "SELECT artist.country FROM artist;"
        out = vote([nulled_a, nulled_b, nulled_a], db_path)
        _check_true("null-rows winner runs", out["fallback_used"] is False)
        _check("null-rows distinct results", out["n_distinct_results"], 1)

        # ------------------------------------------------------------------
        print("\n== vote: BLOB result handling ==")
        blob_q = "SELECT id, payload, note FROM blob_t ORDER BY id;"
        out = vote([blob_q, blob_q, "SELECT * FROM no_such;"], db_path)
        _check("blob winner_sql", out["winner_sql"], blob_q)
        _check("blob winner_count", out["winner_count"], 2)
        _check("blob n_executable", out["n_executable"], 2)

        # ------------------------------------------------------------------
        print("\n== vote: timeout candidate is treated as broken ==")
        runaway = (
            "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r) "
            "SELECT count(*) FROM r;"
        )
        candidates = [runaway, correct, correct]
        out = vote(candidates, db_path, timeout_s=1.0)
        _check_true("runaway is excluded, correct wins",
                    out["winner_sql"] == correct)
        _check("runaway n_executable", out["n_executable"], 2)

    print("\nALL VOTING SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
