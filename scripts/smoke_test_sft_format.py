"""Smoke test for `bird.sft_format` — no GPU, no Modal, no Unsloth.

Verifies the SFT chat format matches the eval-time prompt distribution and
that the assistant target round-trips through `extract_sql` back to the gold SQL.

Run from the repo root:
    python -m scripts.smoke_test_sft_format
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird.data import BirdExample  # noqa: E402
from bird.prompts import SYSTEM_PROMPT, build_messages, extract_sql  # noqa: E402
from bird.schema import extract_schema  # noqa: E402
from bird.sft_format import (  # noqa: E402
    assistant_roundtrips,
    build_sft_dataset,
    canonical_sql,
    format_example_for_sft,
    format_gold_sql_as_assistant,
    write_sft_jsonl,
)


def _check(label: str, actual, expected) -> None:
    ok = actual == expected
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {label}: got={actual!r} expected={expected!r}")
    if not ok:
        sys.exit(1)


def _assert(label: str, cond: bool, detail: str = "") -> None:
    flag = "PASS" if cond else "FAIL"
    msg = f"  [{flag}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not cond:
        sys.exit(1)


def _build_music_db(path: Path) -> None:
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
        INSERT INTO artist VALUES (1, 'Radiohead', 'UK'), (2, 'Daft Punk', 'FR');
        INSERT INTO album VALUES
            (10, 'OK Computer', 1997, 1),
            (11, 'Discovery', 2001, 2);
        """
    )
    conn.commit()
    conn.close()


def _build_split_layout(root: Path, split: str, db_id: str, examples: list[dict]) -> None:
    """Build a minimal BIRD-style split directory: <root>/<split>.json +
    <root>/<split>_databases/<db_id>/<db_id>.sqlite.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{split}.json").write_text(json.dumps(examples, indent=2))
    db_root = root / f"{split}_databases" / db_id
    db_root.mkdir(parents=True, exist_ok=True)
    _build_music_db(db_root / f"{db_id}.sqlite")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "music.sqlite"
        _build_music_db(db_path)
        schema = extract_schema(db_path, "music", n_samples=2)

        gold = ("SELECT artist.name FROM artist JOIN album ON album.artist_id = artist.artist_id "
                "WHERE album.title = 'OK Computer'")  # NB: no trailing semicolon
        ex = BirdExample(
            question_id=0,
            db_id="music",
            question="Which artist made the album OK Computer?",
            evidence="OK Computer refers to album.title = 'OK Computer'.",
            sql=gold,
            difficulty="simple",
        )

        # ---------- 1. format_gold_sql_as_assistant ----------
        print("== format_gold_sql_as_assistant ==")
        asst = format_gold_sql_as_assistant(gold)
        _assert("starts with ```sql fence", asst.startswith("```sql\n"))
        _assert("ends with closing fence", asst.rstrip().endswith("```"))
        _assert("ends gold with exactly one ;", "OK Computer';\n```" in asst,
                detail=repr(asst[-40:]))

        # ---------- 2. format_example_for_sft ----------
        print("\n== format_example_for_sft ==")
        formatted = format_example_for_sft(ex, schema, n_samples=2)
        _check("messages length", len(formatted["messages"]), 3)
        roles = [m["role"] for m in formatted["messages"]]
        _check("message roles", roles, ["system", "user", "assistant"])
        _check("question_id passthrough", formatted["question_id"], 0)
        _check("db_id passthrough", formatted["db_id"], "music")

        # ---------- 3. system+user match eval-time exactly ----------
        print("\n== prompt distribution matches eval ==")
        eval_msgs = build_messages(ex, schema, n_samples=2)
        sft_system = formatted["messages"][0]
        sft_user = formatted["messages"][1]
        _check("system content", sft_system["content"], SYSTEM_PROMPT)
        _check("user content == eval-time", sft_user["content"], eval_msgs[1]["content"])

        # ---------- 4. assistant content is fenced SQL ----------
        print("\n== assistant fenced-SQL shape ==")
        asst_content = formatted["messages"][2]["content"]
        _assert("contains ```sql open fence", "```sql" in asst_content)
        _assert("contains closing fence", asst_content.rstrip().endswith("```"))

        # ---------- 5. extract_sql round-trips ----------
        print("\n== extract_sql round-trip ==")
        recovered = extract_sql(asst_content)
        _check("recovered == canonical gold", recovered, canonical_sql(gold))
        _assert("module-level roundtrip helper agrees", assistant_roundtrips(formatted))

        # ---------- 6. round-trip on goofy variants ----------
        print("\n== round-trip on SQL variants ==")
        variants = [
            "SELECT 1",  # no semicolon, single line
            "SELECT 1;",  # trailing semicolon
            "SELECT 1;;",  # double semicolon
            "  SELECT 1   ",  # whitespace
            "SELECT a\nFROM t\nWHERE b = 'x';",  # multiline
        ]
        for v in variants:
            ex_v = BirdExample(
                question_id=1, db_id="music",
                question="q", evidence="", sql=v, difficulty=None,
            )
            f = format_example_for_sft(ex_v, schema, n_samples=2)
            _assert(f"round-trip {v!r}", assistant_roundtrips(f),
                    detail=f"got {extract_sql(f['messages'][-1]['content'])!r}")

        # ---------- 7. missing-gold raises ----------
        print("\n== missing gold raises ==")
        bad = BirdExample(question_id=99, db_id="music", question="q",
                          evidence="", sql="", difficulty=None)
        try:
            format_example_for_sft(bad, schema)
        except ValueError as e:
            _assert("ValueError raised for empty gold", "no gold SQL" in str(e))
        else:
            _assert("ValueError raised for empty gold", False)

        # ---------- 8. build_sft_dataset on synthetic split ----------
        print("\n== build_sft_dataset (synthetic) ==")
        split_root = tmp_path / "synthetic_split"
        _build_split_layout(
            split_root,
            split="train",
            db_id="music",
            examples=[
                {"db_id": "music", "question": "List UK artists.",
                 "evidence": "", "SQL": "SELECT name FROM artist WHERE country='UK'",
                 "difficulty": "simple"},
                {"db_id": "music", "question": "All album titles.",
                 "evidence": "", "SQL": "SELECT title FROM album",
                 "difficulty": "simple"},
                {"db_id": "music", "question": "Bogus.",  # no gold => should be skipped
                 "evidence": "", "SQL": "",
                 "difficulty": "simple"},
            ],
        )
        examples, stats = build_sft_dataset(split_root, n_samples=2, split_name="train")
        _check("n_emitted", stats.n_emitted, 2)
        _check("n_total", stats.n_total, 3)
        _check("n_missing_gold", stats.n_missing_gold, 1)
        _check("examples returned", len(examples), 2)
        for f in examples:
            _assert(f"roundtrip qid={f['question_id']}", assistant_roundtrips(f))

        # ---------- 9. write_sft_jsonl ----------
        print("\n== write_sft_jsonl ==")
        out_path = tmp_path / "out" / "train.jsonl"
        write_sft_jsonl(examples, out_path)
        lines = out_path.read_text().splitlines()
        _check("jsonl line count", len(lines), 2)
        parsed = [json.loads(line) for line in lines]
        _check("jsonl preserves roles",
               [m["role"] for m in parsed[0]["messages"]],
               ["system", "user", "assistant"])

    print("\nALL SFT FORMAT SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
