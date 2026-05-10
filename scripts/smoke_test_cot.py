"""Smoke test for plan-then-SQL chain-of-thought — runs in seconds, no Modal/GPU/BIRD download.

Builds a tiny in-memory SQLite mirroring smoke_test.py's music-artist-album, asserts:
  * build_plan_messages produces system + user, with schema/evidence/question,
    and does NOT ask for SQL output
  * build_sql_with_plan_messages includes the plan content and DOES ask for SQL
  * extract_plan handles clean plans, chatty preambles, accidental code fences,
    and very long plans (truncation in stage-2 prompt)

Run from the repo root:
    python -m scripts.smoke_test_cot
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

# Make `bird` importable when run as `python scripts/smoke_test_cot.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird.cot import (  # noqa: E402
    build_plan_messages,
    build_sql_with_plan_messages,
    extract_plan,
)
from bird.data import BirdExample  # noqa: E402
from bird.schema import extract_schema  # noqa: E402


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


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "music.sqlite"
        _build_db(db_path)
        schema = extract_schema(db_path, "music", n_samples=2)

        ex = BirdExample(
            question_id=42,
            db_id="music",
            question="Which UK artists have an album titled OK Computer?",
            evidence="UK refers to country = 'UK'.",
            sql="SELECT artist.name FROM artist JOIN album ON album.artist_id = artist.artist_id "
                "WHERE artist.country = 'UK' AND album.title = 'OK Computer';",
            difficulty="moderate",
        )

        # ----- build_plan_messages -----
        print("== build_plan_messages ==")
        plan_msgs = build_plan_messages(ex, schema, n_samples=2)
        _check("message count", len(plan_msgs), 2)
        _check("roles", [m["role"] for m in plan_msgs], ["system", "user"])
        user_text = plan_msgs[1]["content"]
        _assert("schema present", "CREATE TABLE" in user_text and "album" in user_text)
        _assert("question present", ex.question in user_text)
        _assert("evidence present", "UK refers to country" in user_text)
        # The schema block is fenced with ```sql for readability, but the
        # *output instruction* must not ask for SQL — only a plan.
        output_section = user_text.split("### Output", 1)[-1]
        _assert(
            "no SQL fence requested in output instruction",
            "```sql" not in output_section and "SELECT statement" not in output_section,
            "stage 1 must not ask for SQL",
        )
        _assert(
            "plan instruction present",
            "plan" in user_text.lower() and "do not write sql" in user_text.lower(),
        )

        # ----- build_sql_with_plan_messages -----
        print("\n== build_sql_with_plan_messages ==")
        plan_text = (
            "Join artist and album on artist_id. Filter artist.country = 'UK' and "
            "album.title = 'OK Computer'. Select artist.name."
        )
        sql_msgs = build_sql_with_plan_messages(ex, schema, plan_text, n_samples=2)
        _check("message count", len(sql_msgs), 2)
        _check("roles", [m["role"] for m in sql_msgs], ["system", "user"])
        sql_user = sql_msgs[1]["content"]
        _assert("schema present", "CREATE TABLE" in sql_user)
        _assert("question present", ex.question in sql_user)
        _assert("plan content present", "OK Computer" in sql_user and "artist_id" in sql_user)
        _assert("plan section header present", "### Plan" in sql_user)
        _assert(
            "asks for SQL output",
            "```sql" in sql_user and "SELECT statement" in sql_user,
        )

        # Empty-plan path: stage 2 should still produce a sensible prompt.
        sql_msgs_empty = build_sql_with_plan_messages(ex, schema, "", n_samples=2)
        _assert(
            "empty plan handled gracefully",
            "no plan produced" in sql_msgs_empty[1]["content"],
        )

        # Long-plan path: should be truncated before injection.
        long_plan = "Step. " * 1000  # ~6000 chars
        sql_msgs_long = build_sql_with_plan_messages(ex, schema, long_plan, n_samples=2)
        _assert(
            "long plan truncated",
            len(sql_msgs_long[1]["content"]) < len(long_plan) + 4000,
            f"len={len(sql_msgs_long[1]['content'])}",
        )

        # ----- extract_plan -----
        print("\n== extract_plan ==")
        clean = (
            "Join artist and album on artist_id. Filter to country = 'UK' and "
            "title = 'OK Computer'. Select artist.name."
        )
        _check("clean plan unchanged", extract_plan(clean), clean)

        preamble = (
            "Sure! Here's the plan:\n\n"
            + clean
        )
        got = extract_plan(preamble)
        _assert(
            "preamble stripped",
            got == clean,
            f"got={got!r}",
        )

        # Plan that ends with an accidental SQL fence — fence should be stripped.
        accidental = (
            clean
            + "\n```sql\nSELECT artist.name FROM artist JOIN album ON ...;\n```\n"
        )
        got = extract_plan(accidental)
        _assert(
            "trailing SQL fence stripped",
            "SELECT" not in got and "```" not in got and clean in got,
            f"got={got!r}",
        )

        # Stacked preambles get peeled.
        stacked = "Certainly! Here is my plan: " + clean
        got = extract_plan(stacked)
        _assert(
            "stacked preamble stripped",
            got == clean,
            f"got={got!r}",
        )

        # Pure-SQL output (model failed to plan) → empty.
        pure_sql = "```sql\nSELECT artist.name FROM artist;\n```"
        got = extract_plan(pure_sql)
        _assert(
            "pure-SQL output yields empty plan",
            got == "",
            f"got={got!r}",
        )

        # Empty input → empty output.
        _check("empty input", extract_plan(""), "")
        _check("whitespace input", extract_plan("   \n\n  "), "")

    print("\nALL CoT SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
