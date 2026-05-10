"""Smoke test for the self-correction prompt builder.

Verifies build_correction_messages returns a well-formed message list with the
schema, question, failed SQL, and error message all present, and that edge
cases (empty evidence, multi-line error, very long failed SQL) don't get
mangled.

Run from the repo root:
    python -m scripts.smoke_test_correction
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird.correction import (  # noqa: E402
    CORRECTION_SYSTEM_PROMPT,
    build_correction_messages,
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
                                  (2, 'Daft Punk', 'FR');
        INSERT INTO album VALUES (10, 'OK Computer', 1997, 1),
                                 (11, 'Discovery',   2001, 2);
        """
    )
    conn.commit()
    conn.close()


def _check(label: str, cond: bool, info: str = "") -> None:
    flag = "PASS" if cond else "FAIL"
    print(f"  [{flag}] {label}{(' — ' + info) if info else ''}")
    if not cond:
        sys.exit(1)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "music.sqlite"
        _build_db(db_path)
        schema = extract_schema(db_path, "music", n_samples=2)

        ex = BirdExample(
            question_id=1,
            db_id="music",
            question="Which UK artists released an album in 1997?",
            evidence="UK refers to country = 'UK'.",
            sql="",  # not needed for correction prompt
            difficulty="moderate",
        )

        print("== correction: basic structure ==")
        failed_sql = "SELECT a.name FROM artist a JOIN album b ON a.id = b.artist_id WHERE b.year = 1997;"
        error = "no such column: a.id"
        msgs = build_correction_messages(ex, schema, failed_sql, error)
        _check("two messages", len(msgs) == 2, f"got {len(msgs)}")
        _check("system role", msgs[0]["role"] == "system")
        _check("user role", msgs[1]["role"] == "user")
        _check("system prompt content", msgs[0]["content"] == CORRECTION_SYSTEM_PROMPT)

        user = msgs[1]["content"]
        _check("schema in prompt", "CREATE TABLE" in user and "artist" in user and "album" in user)
        _check("question in prompt", ex.question in user)
        _check("evidence in prompt", "UK refers to country" in user)
        _check("failed SQL in prompt", failed_sql in user)
        _check("error msg in prompt", error in user)
        _check("hint bullets present", "gotchas" in user.lower() or "backticks" in user.lower())

        print("\n== correction: empty evidence falls back ==")
        ex_no_ev = BirdExample(
            question_id=2, db_id="music",
            question="How many artists are there?",
            evidence="",  # empty
            sql="", difficulty="simple",
        )
        msgs = build_correction_messages(ex_no_ev, schema, "SELECT count(*) FROM artists;", "no such table: artists")
        user = msgs[1]["content"]
        _check("evidence placeholder", "(none provided)" in user)
        _check("question still in prompt", ex_no_ev.question in user)

        print("\n== correction: multi-line error message preserved ==")
        multiline_err = 'near "Type": syntax error\n  at offset 42\n  in clause: WHERE Type = ...'
        msgs = build_correction_messages(ex, schema, failed_sql, multiline_err)
        user = msgs[1]["content"]
        # All three lines should appear verbatim
        for line in multiline_err.splitlines():
            _check(f"err line {line[:30]!r}", line in user)

        print("\n== correction: very long failed SQL not truncated ==")
        # Generate a deliberately long SQL — the model needs the whole thing.
        long_sql = (
            "SELECT a.name, b.title, "
            + ", ".join([f"a.col_{i}" for i in range(200)])
            + " FROM artist a JOIN album b ON a.artist_id = b.artist_id WHERE b.year = 1997;"
        )
        _check("long sql length sanity", len(long_sql) > 2000, f"len={len(long_sql)}")
        msgs = build_correction_messages(ex, schema, long_sql, "no such column: a.col_0")
        user = msgs[1]["content"]
        _check("full long sql included", long_sql in user)

        print("\n== correction: empty failed SQL still produces valid prompt ==")
        msgs = build_correction_messages(ex, schema, "", "syntax error: unexpected end of input")
        user = msgs[1]["content"]
        _check("placeholder for empty sql", "(no SQL produced)" in user)

        print("\n== correction: empty error still produces valid prompt ==")
        msgs = build_correction_messages(ex, schema, "SELECT 1;", "")
        user = msgs[1]["content"]
        _check("placeholder for empty error", "(no error message)" in user)

    print("\nALL CORRECTION SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
