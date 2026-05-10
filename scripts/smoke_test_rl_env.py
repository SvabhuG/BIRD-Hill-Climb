"""Smoke test for the RL environment's reward function.

The reward function (`bird.rl_env.compute_reward`) is the one signal the model
is optimizing against during GRPO. If it has a bug — gives reward 1.0 to a
wrong query, fails to handle a timeout, etc. — the entire training run is
poisoned. So we test it the same way we test `bird.eval`:
   * build a tiny on-disk SQLite,
   * call `compute_reward` directly with crafted (completion, info) pairs,
   * assert exact reward outcomes for each crafted case.

We also exercise `build_sample` / `build_dataset` against a minimal fake BIRD
layout, so we catch import-time bugs in those helpers.

The verifiers library is NOT required to run this test — `compute_reward`
is the pure-Python core. If verifiers happens to be installed, we also try
to call `load_environment` to verify it constructs without error.

Run from the repo root:

    python -m scripts.smoke_test_rl_env
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

# Make `bird` importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird.rl_env import build_dataset, build_sample, compute_reward  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — mirror scripts/smoke_test.py's tiny artist/album DB
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Reward function tests — the load-bearing logic
# ---------------------------------------------------------------------------

def test_compute_reward(db_path: Path) -> None:
    gold = "SELECT name FROM artist WHERE country = 'UK' ORDER BY name;"
    info = {"db_path": str(db_path), "gold_sql": gold, "db_id": "music", "question_id": 1}
    info_json = json.dumps(info)

    print("== reward: correct SQL (fenced markdown completion) ==")
    completion = "Here's the SQL:\n```sql\n" + gold + "\n```"
    _check("reward", compute_reward(completion, info_json), 1.0)

    print("\n== reward: correct SQL (different ORDER BY — set-equality still 1.0) ==")
    alt = "SELECT name FROM artist WHERE country = 'UK' ORDER BY artist_id DESC;"
    completion = f"```sql\n{alt}\n```"
    _check("reward", compute_reward(completion, info_json), 1.0)

    print("\n== reward: chat-format completion (list of messages) ==")
    chat_completion = [{"role": "assistant", "content": f"```sql\n{gold}\n```"}]
    _check("reward", compute_reward(chat_completion, info_json), 1.0)

    print("\n== reward: wrong rows ==")
    wrong = "SELECT name FROM artist WHERE country = 'FR';"
    completion = f"```sql\n{wrong}\n```"
    _check("reward", compute_reward(completion, info_json), 0.0)

    print("\n== reward: syntax error (no crash) ==")
    bad = "SELEC name FROM artist;"
    completion = f"```sql\n{bad}\n```"
    _check("reward", compute_reward(completion, info_json), 0.0)

    print("\n== reward: empty completion ==")
    _check("reward", compute_reward("", info_json), 0.0)
    _check("reward", compute_reward([], info_json), 0.0)

    print("\n== reward: model produced prose only, no SQL ==")
    completion = "I'm sorry, I can't answer that."
    _check("reward", compute_reward(completion, info_json), 0.0)

    print("\n== reward: timeout on runaway recursive CTE ==")
    runaway = (
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r) "
        "SELECT count(*) FROM r;"
    )
    completion = f"```sql\n{runaway}\n```"
    # Tight 1.0-second timeout so the test doesn't hang if interrupt fails.
    _check("reward", compute_reward(completion, info_json, exec_timeout_s=1.0), 0.0)

    print("\n== reward: gold itself errors (corpus problem) — return 0, do not crash ==")
    bad_gold_info = json.dumps({"db_path": str(db_path), "gold_sql": "SELEC bad gold;", "db_id": "music", "question_id": 99})
    _check("reward", compute_reward(f"```sql\n{gold}\n```", bad_gold_info), 0.0)

    print("\n== reward: info as dict (not just JSON string) ==")
    _check("reward", compute_reward(f"```sql\n{gold}\n```", info), 1.0)

    print("\n== reward: missing db_path / gold_sql ==")
    _check("reward", compute_reward(f"```sql\n{gold}\n```", json.dumps({})), 0.0)


# ---------------------------------------------------------------------------
# Dataset builder tests — make sure prompts/info shapes are right
# ---------------------------------------------------------------------------

def test_build_sample_and_dataset(tmp: Path, db_path: Path) -> None:
    """Build a minimal fake BIRD layout: train.json + train_databases/music/music.sqlite."""
    from bird.data import BirdExample

    print("\n== build_sample: shape ==")
    ex = BirdExample(
        question_id=42,
        db_id="music",
        question="Which UK artists have an album?",
        evidence="UK refers to country = 'UK'.",
        sql="SELECT DISTINCT artist.name FROM artist JOIN album ON album.artist_id = artist.artist_id WHERE artist.country = 'UK';",
        difficulty="moderate",
    )
    sample = build_sample(ex, db_path, n_samples_schema=2)
    assert "prompt" in sample and isinstance(sample["prompt"], list), sample
    assert "info" in sample and isinstance(sample["info"], str), sample
    info = json.loads(sample["info"])
    _check("info.gold_sql present", bool(info.get("gold_sql")), True)
    _check("info.db_path present", info["db_path"], str(db_path))
    _check("info.question_id", info["question_id"], 42)
    # Prompt should be a list[dict] with system + user
    _check("prompt has system + user", len(sample["prompt"]), 2)
    _check("prompt[0].role", sample["prompt"][0]["role"], "system")
    _check("prompt[1].role", sample["prompt"][1]["role"], "user")
    # User message should embed the schema and the question
    user_content = sample["prompt"][1]["content"]
    assert "CREATE TABLE" in user_content, user_content[:300]
    assert ex.question in user_content, user_content[:300]

    print("\n== build_sample: roundtrip — sample's info works as compute_reward input ==")
    # Simulate a perfect rollout
    completion = f"```sql\n{ex.sql}\n```"
    _check("reward on built sample", compute_reward(completion, sample["info"]), 1.0)

    print("\n== build_dataset: end-to-end against a fake BIRD layout ==")
    # Build the directory shape load_split expects:  <root>/train.json,
    #                                                 <root>/train_databases/<db_id>/<db_id>.sqlite
    fake_root = tmp / "fake_bird"
    (fake_root / "train_databases" / "music").mkdir(parents=True, exist_ok=True)
    # Copy the test db in
    import shutil
    shutil.copyfile(db_path, fake_root / "train_databases" / "music" / "music.sqlite")
    (fake_root / "train.json").write_text(json.dumps([
        {
            "db_id": "music",
            "question": ex.question,
            "evidence": ex.evidence,
            "SQL": ex.sql,
            "difficulty": ex.difficulty,
        },
        {  # Second example — different question
            "db_id": "music",
            "question": "Name all French artists.",
            "evidence": "",
            "SQL": "SELECT name FROM artist WHERE country = 'FR';",
            "difficulty": "simple",
        },
        {  # Will be filtered: missing gold SQL
            "db_id": "music",
            "question": "An unlabeled question.",
            "evidence": "",
            "SQL": "",
            "difficulty": None,
        },
    ]))
    samples = build_dataset(fake_root, split="train", n_samples_schema=2)
    _check("build_dataset filters empties", len(samples), 2)
    _check("first sample structure", set(samples[0].keys()), {"prompt", "info"})


# ---------------------------------------------------------------------------
# Optional: if verifiers is installed, smoke-instantiate load_environment
# ---------------------------------------------------------------------------

def test_load_environment_if_available(tmp: Path, db_path: Path) -> None:
    print("\n== load_environment (only if `verifiers` is installed) ==")
    try:
        import verifiers  # noqa: F401
        import datasets  # noqa: F401
    except ImportError as e:
        print(f"  [SKIP] verifiers/datasets not installed: {e}")
        return

    # Build a fake BIRD root so load_environment can find a dataset
    import shutil
    fake_root = tmp / "fake_bird_env"
    (fake_root / "train_databases" / "music").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(db_path, fake_root / "train_databases" / "music" / "music.sqlite")
    (fake_root / "train.json").write_text(json.dumps([
        {
            "db_id": "music",
            "question": "List UK artists.",
            "evidence": "",
            "SQL": "SELECT name FROM artist WHERE country = 'UK';",
            "difficulty": "simple",
        },
    ]))

    from bird.rl_env import load_environment
    try:
        env = load_environment(
            train_root=str(fake_root),
            split="train",
            num_train_examples=1,
            n_samples_schema=1,
        )
    except Exception as e:
        # If verifiers' API has shifted, fail loudly rather than mask.
        print(f"  [FAIL] load_environment raised: {e!r}")
        sys.exit(1)
    print(f"  [PASS] load_environment returned {type(env).__name__}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_:
        tmp = Path(tmp_)
        db_path = tmp / "music.sqlite"
        _build_db(db_path)

        test_compute_reward(db_path)
        test_build_sample_and_dataset(tmp, db_path)
        test_load_environment_if_available(tmp, db_path)

    print("\nALL RL ENV SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
