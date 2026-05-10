"""Smoke test for eval + schema modules — runs in seconds, no Modal/GPU/BIRD download required.

Builds a tiny in-memory SQLite, asserts:
  * schema extraction reads tables, columns, FKs
  * the evaluator returns CORRECT for gold==pred
  * returns WRONG when answers diverge
  * returns EXEC_ERROR on bad SQL
  * returns TIMEOUT on a runaway recursive CTE
  * SQL extraction handles fenced + bare model outputs

Run from the repo root:
    python -m scripts.smoke_test
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

# Make `bird` importable when run as `python scripts/smoke_test.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird.data import BirdExample  # noqa: E402
from bird.eval import EvalStatus, evaluate_one  # noqa: E402
from bird.linking import (  # noqa: E402
    Selection,
    ensure_keys,
    lexical_link,
    parse_linker_output,
    restrict_schema,
)
from bird.linking_eval import gold_columns, linking_metrics  # noqa: E402
from bird.prompts import extract_sql  # noqa: E402
from bird.schema import extract_schema, render_ddl_with_samples  # noqa: E402


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


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "music.sqlite"
        _build_db(db_path)

        print("== schema extraction ==")
        schema = extract_schema(db_path, "music", n_samples=2)
        table_names = [t.name for t in schema.tables]
        _check("table list", sorted(table_names), ["album", "artist"])
        album = next(t for t in schema.tables if t.name == "album")
        _check("album column count", len(album.columns), 4)
        _check("album FK target", album.foreign_keys[0].to_table, "artist")
        ddl = render_ddl_with_samples(schema, n_samples=2)
        assert "CREATE TABLE" in ddl and "sample rows" in ddl

        print("\n== eval: correct prediction ==")
        gold = "SELECT name FROM artist WHERE country = 'UK' ORDER BY name;"
        r = evaluate_one(
            question_id=1, db_id="music", db_path=db_path,
            predicted_sql=gold, gold_sql=gold, difficulty="simple",
        )
        _check("status", r.status, EvalStatus.CORRECT)

        print("\n== eval: order-insensitive set match ==")
        # Different ORDER BY, same set of rows — BIRD's EX uses set equality.
        pred = "SELECT name FROM artist WHERE country = 'UK' ORDER BY artist_id DESC;"
        r = evaluate_one(
            question_id=2, db_id="music", db_path=db_path,
            predicted_sql=pred, gold_sql=gold, difficulty="simple",
        )
        _check("status", r.status, EvalStatus.CORRECT)

        print("\n== eval: wrong answer ==")
        pred = "SELECT name FROM artist WHERE country = 'FR';"
        r = evaluate_one(
            question_id=3, db_id="music", db_path=db_path,
            predicted_sql=pred, gold_sql=gold, difficulty="moderate",
        )
        _check("status", r.status, EvalStatus.WRONG)

        print("\n== eval: SQL syntax error ==")
        r = evaluate_one(
            question_id=4, db_id="music", db_path=db_path,
            predicted_sql="SELEC name FROM artist;", gold_sql=gold, difficulty="simple",
        )
        _check("status", r.status, EvalStatus.EXEC_ERROR)

        print("\n== eval: empty prediction ==")
        r = evaluate_one(
            question_id=5, db_id="music", db_path=db_path,
            predicted_sql="", gold_sql=gold,
        )
        _check("status", r.status, EvalStatus.EMPTY)

        print("\n== eval: timeout via runaway recursive CTE ==")
        runaway = (
            "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r) "
            "SELECT count(*) FROM r;"
        )
        r = evaluate_one(
            question_id=6, db_id="music", db_path=db_path,
            predicted_sql=runaway, gold_sql=gold, timeout_s=1.0,
        )
        _check("status", r.status, EvalStatus.TIMEOUT)

        print("\n== sql extraction ==")
        cases = [
            (
                "Here is your SQL:\n```sql\nSELECT 1;\n```\nThanks!",
                "SELECT 1;",
            ),
            (
                "```\nSELECT name FROM artist WHERE country='UK'\n```",
                "SELECT name FROM artist WHERE country='UK';",
            ),
            (
                "Sure — SELECT 2 FROM dual",
                "SELECT 2 FROM dual;",
            ),
        ]
        for raw, want in cases:
            got = extract_sql(raw)
            _check(f"extract({raw[:30]!r}...)", got.strip(), want)

    # --- linking + linking_eval, in-memory only ---
    with tempfile.TemporaryDirectory() as tmp2:
        db_path = Path(tmp2) / "music.sqlite"
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

        print("\n== linking: lexical (high recall) ==")
        sel_lex = lexical_link(ex, schema)
        assert ("artist", "country") in sel_lex.columns, f"missed artist.country: {sel_lex.columns}"
        assert ("album", "title") in sel_lex.columns, f"missed album.title: {sel_lex.columns}"
        print(f"  [PASS] lexical found {sorted(sel_lex.columns)}")

        print("\n== linking: parse_linker_output (JSON) ==")
        sel_json = parse_linker_output(
            'Here you go: {"artist": ["name", "country"], "album": ["title", "artist_id"]}',
            schema,
        )
        _check("json columns",
               sel_json.columns,
               frozenset({("artist", "name"), ("artist", "country"),
                         ("album", "title"), ("album", "artist_id")}))

        print("\n== linking: parse_linker_output (dotted fallback) ==")
        sel_dotted = parse_linker_output(
            "Use artist.name and artist.country plus album.title.",
            schema,
        )
        assert ("artist", "name") in sel_dotted.columns
        assert ("album", "title") in sel_dotted.columns

        print("\n== linking: parse_linker_output drops hallucinations ==")
        sel_halluc = parse_linker_output(
            '{"nonexistent_table": ["x"], "artist": ["fake_col", "name"]}',
            schema,
        )
        _check("hallucinated columns dropped",
               sel_halluc.columns, frozenset({("artist", "name")}))

        print("\n== linking: ensure_keys adds PK/FK ==")
        seed = Selection.from_pairs([("album", "title")])
        with_keys = ensure_keys(seed, schema)
        # Must keep album.title, add album.album_id (PK), album.artist_id (FK), artist.artist_id (FK target)
        assert ("album", "title") in with_keys.columns
        assert ("album", "album_id") in with_keys.columns
        assert ("album", "artist_id") in with_keys.columns
        assert ("artist", "artist_id") in with_keys.columns
        print(f"  [PASS] post-ensure_keys: {sorted(with_keys.columns)}")

        print("\n== linking: restrict_schema projects correctly ==")
        sel = Selection.from_pairs([("artist", "name"), ("artist", "country")])
        narrow = restrict_schema(schema, sel)
        _check("restricted table list", [t.name for t in narrow.tables], ["artist"])
        _check("restricted column count", len(narrow.tables[0].columns), 2)

        print("\n== linking_eval: gold_columns (basic) ==")
        gold, status = gold_columns(
            "SELECT artist.name FROM artist WHERE artist.country = 'UK';", schema
        )
        _check("gold_columns status", status, "ok")
        _check("gold_columns set",
               gold, {("artist", "name"), ("artist", "country")})

        print("\n== linking_eval: gold_columns (alias resolution) ==")
        gold, status = gold_columns(
            "SELECT a.name FROM artist AS a JOIN album AS b ON b.artist_id = a.artist_id "
            "WHERE b.title = 'OK Computer';", schema,
        )
        _check("alias-resolved gold", status, "ok")
        # Aliases a, b should resolve back to artist, album
        assert ("artist", "name") in gold, f"gold missing artist.name: {gold}"
        assert ("album", "title") in gold, f"gold missing album.title: {gold}"

        print("\n== linking_eval: gold_columns (SELECT *) ==")
        gold, status = gold_columns("SELECT * FROM artist;", schema)
        _check("SELECT * status", status, "ok")
        # Expanded star should include all artist columns
        assert ("artist", "name") in gold and ("artist", "country") in gold

        print("\n== linking_eval: linking_metrics ==")
        # Selection covers gold exactly
        sel_perfect = Selection.from_pairs([("artist", "name"), ("artist", "country")])
        gold_set = {("artist", "name"), ("artist", "country")}
        m = linking_metrics(sel_perfect.columns, gold_set)
        _check("perfect recall", m["recall"], 1.0)

        # Selection misses one
        sel_partial = Selection.from_pairs([("artist", "name")])
        m = linking_metrics(sel_partial.columns, gold_set)
        _check("partial recall", m["recall"], 0.5)
        _check("missed entry", m["missed"], [("artist", "country")])

    print("\nALL SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
