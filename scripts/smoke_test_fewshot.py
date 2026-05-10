"""Smoke test for few-shot retrieval — runs in seconds, no Modal/GPU/BIRD download required.

Builds a synthetic train list and asserts:
  * retrieve prefers same-db_id examples over cross-db
  * retrieve falls back to cross-db when the in-domain pool is too small
  * retrieve handles k > available without crashing
  * retrieve returns deterministic order on score ties (qid-ascending)
  * empty-question query is non-fatal
  * build_messages_with_fewshot interleaves shots between the schema and the
    "### External knowledge / hint" block, in order
  * char-budget guard drops trailing shots when needed

Run from the repo root:
    python -m scripts.smoke_test_fewshot
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

# Make `bird` importable when run as `python scripts/smoke_test_fewshot.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird.data import BirdExample  # noqa: E402
from bird.fewshot import (  # noqa: E402
    _tokenize,
    build_train_index,
    retrieve,
)
from bird.prompts import build_messages_with_fewshot  # noqa: E402
from bird.schema import extract_schema  # noqa: E402


def _check(label: str, actual, expected) -> None:
    ok = actual == expected
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {label}: got={actual!r} expected={expected!r}")
    if not ok:
        sys.exit(1)


def _assert(label: str, cond: bool, detail: str = "") -> None:
    flag = "PASS" if cond else "FAIL"
    print(f"  [{flag}] {label}{(' :: ' + detail) if detail else ''}")
    if not cond:
        sys.exit(1)


def _ex(qid: int, db_id: str, question: str, sql: str = "SELECT 1;", evidence: str = "") -> BirdExample:
    return BirdExample(
        question_id=qid,
        db_id=db_id,
        question=question,
        evidence=evidence,
        sql=sql,
        difficulty="simple",
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
        INSERT INTO artist VALUES (1, 'Radiohead', 'UK'), (2, 'Daft Punk', 'FR');
        INSERT INTO album VALUES (10, 'OK Computer', 1997, 1), (11, 'Discovery', 2001, 2);
        """
    )
    conn.commit()
    conn.close()


def main() -> None:
    print("== fewshot: tokenize ==")
    toks = _tokenize("What is the average age of UK artists?")
    _assert("lowercased + stopworded", "average" in toks and "uk" in toks)
    _assert("stopwords removed", "the" not in toks and "is" not in toks and "of" not in toks)
    _check("empty input", _tokenize(""), set())

    print("\n== fewshot: build_train_index ==")
    train_examples = [
        # music DB
        _ex(1, "music", "List all UK artists by name.",
            sql="SELECT name FROM artist WHERE country = 'UK';"),
        _ex(2, "music", "How many albums did Radiohead release after 1995?",
            sql="SELECT COUNT(*) FROM album WHERE artist_id = 1 AND year > 1995;"),
        _ex(3, "music", "Find the title of the oldest album.",
            sql="SELECT title FROM album ORDER BY year ASC LIMIT 1;"),
        _ex(4, "music", "Which artists are from France?",
            sql="SELECT name FROM artist WHERE country = 'FR';"),
        _ex(5, "music", "Average year of UK artist albums?",
            sql="SELECT AVG(year) FROM album JOIN artist ON album.artist_id = artist.artist_id "
                "WHERE artist.country = 'UK';"),
        # books DB
        _ex(10, "books", "What is the title of the longest book?",
            sql="SELECT title FROM books ORDER BY pages DESC LIMIT 1;"),
        _ex(11, "books", "List authors who write fantasy novels.",
            sql="SELECT name FROM authors WHERE genre = 'fantasy';"),
        _ex(12, "books", "Average pages per genre?",
            sql="SELECT genre, AVG(pages) FROM books GROUP BY genre;"),
        # movies DB
        _ex(20, "movies", "Top-rated UK movies released after 2010?",
            sql="SELECT title FROM movies WHERE country = 'UK' AND year > 2010;"),
        _ex(21, "movies", "Which director directed the most movies?",
            sql="SELECT director FROM movies GROUP BY director ORDER BY COUNT(*) DESC LIMIT 1;"),
    ]
    train = build_train_index(train_examples)
    _check("examples count", len(train.examples), 10)
    _check("db_ids present", sorted(train.by_db_id.keys()), ["books", "movies", "music"])
    _check("music bucket size", len(train.by_db_id["music"]), 5)

    print("\n== fewshot: retrieve prefers same-db ==")
    shots = retrieve(
        question="Find UK artists with albums after 2000",
        db_id="music",
        train=train,
        k=4,
    )
    _check("shot count", len(shots), 4)
    _assert("all shots from music", all(s.db_id == "music" for s in shots),
            detail=str([s.db_id for s in shots]))
    # Top shot should be the one with highest overlap. "UK", "artists", "albums",
    # "2000" — qid 5 (UK + artist + albums + year-related) and qid 1 (UK artists)
    # are obvious candidates. We just sanity-check the question_id is among the
    # high-overlap ones.
    top_qids = [s.question_id for s in shots[:2]]
    _assert("top-2 are high-overlap music questions",
            any(q in top_qids for q in (1, 5)),
            detail=f"top={top_qids}")

    print("\n== fewshot: retrieve falls back to cross-db when in-domain is empty ==")
    shots = retrieve(
        question="What is the average age of artists?",
        db_id="nonexistent_db",
        train=train,
        k=3,
    )
    _check("fallback shot count", len(shots), 3)
    _assert("none from same db", all(s.db_id != "nonexistent_db" for s in shots))

    print("\n== fewshot: retrieve fills from cross-db when same-db is small ==")
    # `books` has 3 shots, asking for 5 — should grab all 3 books shots, then
    # 2 cross-db.
    shots = retrieve(
        question="Find the longest book by genre",
        db_id="books",
        train=train,
        k=5,
    )
    _check("total shots", len(shots), 5)
    in_db = [s for s in shots if s.db_id == "books"]
    out_db = [s for s in shots if s.db_id != "books"]
    _check("same-db filled first", len(in_db), 3)
    _check("cross-db fills the rest", len(out_db), 2)
    # Same-db shots should appear before cross-db shots.
    boundary = next((i for i, s in enumerate(shots) if s.db_id != "books"), len(shots))
    _assert("same-db shots come first",
            all(s.db_id == "books" for s in shots[:boundary])
            and all(s.db_id != "books" for s in shots[boundary:]))

    print("\n== fewshot: retrieve handles k > available ==")
    tiny = build_train_index([_ex(100, "tiny", "What is x?")])
    shots = retrieve(question="What about y?", db_id="tiny", train=tiny, k=10)
    _check("returns what's there", len(shots), 1)

    print("\n== fewshot: retrieve handles k <= 0 and empty index ==")
    _check("k=0 -> empty", retrieve("anything", "music", train, k=0), [])
    empty = build_train_index([])
    _check("empty index -> empty", retrieve("anything", "music", empty, k=4), [])

    print("\n== fewshot: retrieve handles empty question (degenerate) ==")
    shots = retrieve(question="", db_id="music", train=train, k=3)
    # All scores will be 0; we still return same-db shots in qid order.
    _check("empty-question shot count", len(shots), 3)
    _assert("all from music", all(s.db_id == "music" for s in shots))
    # Stable order on ties: ascending question_id.
    _check("tie-break order is qid asc", [s.question_id for s in shots], [1, 2, 3])

    print("\n== fewshot: build_messages_with_fewshot interleaves shots ==")
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "music.sqlite"
        _build_db(db_path)
        schema = extract_schema(db_path, "music", n_samples=2)

        target = BirdExample(
            question_id=999,
            db_id="music",
            question="Which UK artists have multiple albums?",
            evidence="UK refers to country = 'UK'.",
            sql="",
            difficulty="moderate",
        )
        shots = retrieve(target.question, target.db_id, train, k=3)
        msgs = build_messages_with_fewshot(target, schema, shots, n_samples=2)
        _check("message count", len(msgs), 2)
        _check("system role", msgs[0]["role"], "system")
        _check("user role", msgs[1]["role"], "user")

        user_content = msgs[1]["content"]
        # Position assertions: schema -> Examples -> hint -> question -> output.
        i_schema = user_content.find("### Database schema")
        i_examples = user_content.find("### Examples")
        i_hint = user_content.find("### External knowledge / hint")
        i_question = user_content.find("### Question")
        i_output = user_content.find("### Output")
        _assert("schema appears", i_schema >= 0)
        _assert("examples block appears", i_examples >= 0)
        _assert("hint appears after examples", i_examples < i_hint, f"{i_examples=} {i_hint=}")
        _assert("schema before examples", i_schema < i_examples, f"{i_schema=} {i_examples=}")
        _assert("question after hint", i_hint < i_question)
        _assert("output last", i_question < i_output)

        # Each shot's gold SQL should be embedded.
        for s in shots:
            _assert(f"shot qid={s.question_id} SQL embedded",
                    s.sql.strip().rstrip(";") in user_content)

        # Order of shots in the prompt should match order in `shots`.
        positions = [user_content.find(f"### Example {i}") for i in range(1, len(shots) + 1)]
        _assert("shots numbered 1..n", all(p > 0 for p in positions))
        _assert("shot order is monotonic", positions == sorted(positions))

    print("\n== fewshot: build_messages_with_fewshot with no shots == base prompt ==")
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "music.sqlite"
        _build_db(db_path)
        schema = extract_schema(db_path, "music", n_samples=2)
        msgs = build_messages_with_fewshot(target, schema, shots=[], n_samples=2)
        _assert("no Examples block when shots=[]", "### Examples" not in msgs[1]["content"])

    print("\n== fewshot: char-budget guard drops big shots ==")
    huge_sql = "SELECT " + ", ".join([f"col_{i}" for i in range(2000)]) + " FROM tbl;"
    bloated = [
        _ex(50, "music", "Tiny question one.", sql="SELECT 1;"),
        _ex(51, "music", "Tiny question two.", sql=huge_sql),
        _ex(52, "music", "Tiny question three.", sql="SELECT 3;"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "music.sqlite"
        _build_db(db_path)
        schema = extract_schema(db_path, "music", n_samples=2)
        msgs = build_messages_with_fewshot(target, schema, shots=bloated, n_samples=2)
        content = msgs[1]["content"]
        _assert("budget guard kept tiny shot 1", "Tiny question one." in content)
        _assert("budget guard dropped huge shot 2", huge_sql not in content)
        _assert("budget guard kept tiny shot 3 (after skip)", "Tiny question three." in content)

    print("\nALL FEWSHOT SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
