"""Smoke test for the agentic loop — no Modal/GPU required.

Validates:
  * tool-call parsing (fenced JSON, bare JSON, missing-tool fallback)
  * execute_sql tool dispatch on a tiny in-memory DB (success + error)
  * full step_agent loop with a scripted chat_fn that drives:
      - draft -> execute_sql -> see result -> submit
      - submit-only path
      - budget exhaustion path

Run from the repo root:
    python -m scripts.smoke_test_agentic
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird.agentic import (  # noqa: E402
    build_initial_messages,
    parse_tool_call,
    run_tool,
    step_agent,
)
from bird.data import BirdExample  # noqa: E402
from bird.schema import extract_schema  # noqa: E402


def _build_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE artist (artist_id INTEGER PRIMARY KEY, name TEXT, country TEXT);
        INSERT INTO artist VALUES (1,'Radiohead','UK'),(2,'Daft Punk','FR'),(3,'Beatles','UK');
        """
    )
    conn.commit()
    conn.close()


def _check(label, got, want):
    ok = got == want
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {label}: got={got!r} expected={want!r}")
    if not ok:
        sys.exit(1)


def main() -> None:
    print("== parse_tool_call: fenced json ==")
    out = parse_tool_call('reasoning...\n```json\n{"tool":"submit","args":{"sql":"SELECT 1;"}}\n```')
    _check("tool", out["tool"], "submit")
    _check("sql arg", out["args"]["sql"], "SELECT 1;")

    print("\n== parse_tool_call: bare json ==")
    out = parse_tool_call('{"tool":"execute_sql","args":{"sql":"SELECT 1;"}}')
    _check("tool", out["tool"], "execute_sql")

    print("\n== parse_tool_call: no tool ==")
    out = parse_tool_call("here is some prose with ```sql\nSELECT 1;\n``` only")
    _check("no tool", out, None)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "music.sqlite"
        _build_db(db_path)

        print("\n== run_tool: execute_sql success ==")
        obs = run_tool("execute_sql", {"sql": "SELECT name FROM artist WHERE country='UK';"}, db_path)
        assert "Radiohead" in obs and "Beatles" in obs, obs
        print(f"  [PASS] returned rows; first 60 chars: {obs[:60]!r}")

        print("\n== run_tool: execute_sql error ==")
        obs = run_tool("execute_sql", {"sql": "SELEC name FROM artist;"}, db_path)
        assert obs.startswith("ERROR"), obs
        print(f"  [PASS] error obs: {obs[:80]!r}")

        print("\n== run_tool: unknown tool ==")
        obs = run_tool("inspect_table", {"name": "artist"}, db_path)
        assert obs.startswith("ERROR: unknown tool"), obs
        print(f"  [PASS] {obs[:60]!r}")

        print("\n== parse_tool_call: keep_baseline_sql ==")
        out = parse_tool_call('```json\n{"tool":"keep_baseline_sql","args":{}}\n```')
        _check("tool", out["tool"], "keep_baseline_sql")

        # Build a synthetic example
        schema = extract_schema(db_path, "music", n_samples=2)
        ex = BirdExample(
            question_id=1, db_id="music",
            question="Names of UK artists, alphabetical.",
            evidence="UK means country='UK'",
            sql="SELECT name FROM artist WHERE country='UK' ORDER BY name;",
            difficulty="simple",
        )
        msgs = build_initial_messages(ex, schema)
        assert msgs[0]["role"] == "system" and "execute_sql" in msgs[0]["content"]

        print("\n== step_agent: draft -> execute -> submit ==")
        replies = iter([
            # Turn 1: try a draft
            'Let me check.\n```json\n{"tool":"execute_sql","args":{"sql":"SELECT name FROM artist WHERE country=\'UK\' ORDER BY name;"}}\n```',
            # Turn 2: looks good, submit
            'Looks correct.\n```json\n{"tool":"submit","args":{"sql":"SELECT name FROM artist WHERE country=\'UK\' ORDER BY name;"}}\n```',
        ])
        trace = step_agent(msgs, db_path=db_path, max_turns=4, chat_fn=lambda _m: next(replies))
        _check("finish_reason", trace.finish_reason, "submit")
        _check("turns", trace.turns, 2)
        _check("n_tool_calls", trace.n_tool_calls, 1)
        assert trace.final_sql.lower().startswith("select name from artist"), trace.final_sql
        print(f"  [PASS] final_sql: {trace.final_sql!r}")

        print("\n== step_agent: budget exhaustion ==")
        msgs2 = build_initial_messages(ex, schema)
        bad = '```json\n{"tool":"execute_sql","args":{"sql":"SELEC bad sql"}}\n```'
        trace = step_agent(msgs2, db_path=db_path, max_turns=3, chat_fn=lambda _m: bad)
        _check("finish_reason", trace.finish_reason, "budget")
        assert trace.n_exec_errors >= 1
        print(f"  [PASS] budget hit; exec_errors={trace.n_exec_errors}")

        print("\n== step_agent: no_tool_call -> extract from fenced ==")
        msgs3 = build_initial_messages(ex, schema)
        text = "Final answer:\n```sql\nSELECT name FROM artist WHERE country='UK' ORDER BY name;\n```"
        trace = step_agent(msgs3, db_path=db_path, max_turns=2, chat_fn=lambda _m: text)
        _check("finish_reason", trace.finish_reason, "no_tool_call")
        assert "select name from artist" in trace.final_sql.lower(), trace.final_sql
        print(f"  [PASS] extracted: {trace.final_sql!r}")

        print("\n== step_agent: keep_baseline_sql returns baseline ==")
        baseline = "SELECT name FROM artist WHERE country='UK';"
        msgs4 = build_initial_messages(ex, schema, baseline_sql=baseline)
        text = 'No defect found.\n```json\n{"tool":"keep_baseline_sql","args":{}}\n```'
        trace = step_agent(
            msgs4, db_path=db_path, max_turns=3,
            chat_fn=lambda _m: text, baseline_sql=baseline,
        )
        _check("finish_reason", trace.finish_reason, "keep_baseline")
        assert trace.final_sql == baseline, trace.final_sql
        print(f"  [PASS] kept baseline: {trace.final_sql!r}")

        print("\n== build_initial_messages: baseline included in user prompt ==")
        msgs5 = build_initial_messages(ex, schema, baseline_sql=baseline)
        assert baseline in msgs5[1]["content"], msgs5[1]["content"]
        assert "Baseline candidate SQL" in msgs5[1]["content"]
        print("  [PASS] baseline embedded in user prompt")

    print("\nALL AGENTIC SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
