"""Agentic SQL solver: tool-using loop on top of an instruct LLM.

DESIGN
======
Goal: explore whether giving the model a `execute_sql` tool — so it can run
candidate queries against the read-only DB and see actual rows / errors —
beats the greedy zero-shot baseline that has only the schema + sample rows.

We deliberately keep the tool palette to TWO surfaces:

    1. execute_sql(sql: str) -> {"ok": bool, "rows": [...], "error": str|None}
       Read-only SQLite execution with timeout. Truncates result to N rows so
       a SELECT * doesn't blow the context window. This is the load-bearing tool:
       it gives the model ground-truth feedback (does this query parse? does it
       return data? does the row shape look like what the question asks?).

    2. submit(sql: str) -> terminates the loop, the SQL is the final answer.
       Modeling termination as an explicit tool call (rather than a special
       string) keeps parsing trivial and aligns with how Qwen3-Coder was trained
       for tool use.

Why no `inspect_table` / `list_tables`:
    The full schema with FKs and 3 sample rows per table is already in the
    initial system prompt. An `inspect_table` tool would mostly duplicate that
    info; the model can always just `SELECT * FROM t LIMIT 5` via execute_sql.
    Adding a third tool would mean the model has to choose between three
    options at every turn — worse decision rate, more drift. Two tools is the
    minimum that gives ground-truth feedback + termination.

LOOP
----
    system_prompt = SYSTEM (rules + tool catalog)
    messages = [system, user(schema + question + hint)]
    for turn in 1..MAX_TURNS:
        text = LLM.chat(messages)
        call = parse_tool_call(text)             # JSON in fenced ```json block
        if call is None:                          # no tool, treat as final SQL
            return extract_sql(text)
        if call.name == "submit":
            return call.args["sql"]
        if call.name == "execute_sql":
            obs = run_sql(call.args["sql"])      # read-only, timeout, truncate
            messages += [assistant(text), user(format_obs(obs))]
            continue
    return last seen SQL or empty                 # budget exhausted

Termination conditions:
    - submit(sql)        -> success (intended exit)
    - text without a parseable tool call -> assume the assistant gave up
      explaining or wrapped final SQL in a fence, extract and stop.
    - MAX_TURNS reached -> emit best-effort SQL (last execute_sql arg or empty).

Tool-call format choice:
    A fenced ```json {"tool": "...", "args": {...}} ``` block is dialect-neutral
    (works for any chat model, no special tokens) and easy to parse robustly.
    OpenAI-style "function calling" requires vLLM's tools= API; this is simpler
    and good enough to test the hypothesis.

Budget:
    MAX_TURNS = 6, MAX_OBS_ROWS = 20, MAX_OBS_BYTES = 4000.
    With ~2k schema-prompt tokens + ~1.5k average tool-overhead per turn,
    total context stays well under our 16k vLLM ceiling for a 6-turn loop.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .data import BirdExample
from .prompts import extract_sql
from .schema import DatabaseSchema, render_ddl_with_samples


# ----------------------- prompts -----------------------

AGENT_SYSTEM = """\
You are a senior data analyst writing correct, idiomatic SQLite SQL.

You are given a BASELINE candidate SQL drafted by a strong zero-shot model.
Your job is to probe with execute_sql and decide whether to keep it, fix it,
or replace it. The baseline is usually defensible; only override it when you
can point at a SPECIFIC defect.

You have three tools you can call by emitting a JSON block:

```json
{"tool": "execute_sql", "args": {"sql": "SELECT ..."}}
```
  Runs the SQL read-only against the target SQLite database and returns rows
  (truncated to 20) or an error string. Use it to verify your draft works,
  to peek at column values you're unsure about, or to count results.

```json
{"tool": "submit", "args": {"sql": "SELECT ..."}}
```
  Submits your final answer and ends the session. Call this exactly once,
  at the end, with the SELECT statement that answers the question.

```json
{"tool": "keep_baseline_sql", "args": {}}
```
  Keeps the baseline candidate SQL unchanged as the final answer and ends the
  session. Use this when probing did NOT reveal a concrete defect in the baseline.

Rules:
- Emit at most ONE JSON tool block per message. After it, stop.
- Use execute_sql freely (up to 5 times) to test drafts. Tool output is fed
  back to you; reason briefly on what changed, then either refine and try
  again, or call submit / keep_baseline_sql.
- If after probing you CANNOT identify a specific defect in the baseline SQL
  — a wrong column, wrong join, missing filter, missing null handling, or
  similar — you SHOULD call keep_baseline_sql() rather than submitting a
  modified query. Changing the SQL without a specific defect to fix usually
  makes things worse.
- Only the SQL inside submit's `args.sql` (or the baseline, if you called
  keep_baseline_sql) is graded. Make it a single SELECT (or WITH ... SELECT)
  ending without trailing prose."""


AGENT_USER = """\
### Database schema (SQLite)
```sql
{schema_block}
```

### External knowledge / hint
{evidence}

### Question
{question}

### Baseline candidate SQL (your starting point, may be wrong)
```sql
{baseline_sql}
```

Begin. Probe with execute_sql if helpful. Then either submit a fix or call
keep_baseline_sql if you can't point to a specific defect."""


# ----------------------- tool dispatch -----------------------


def _execute_readonly(db_path: str | Path, sql: str, timeout_s: float) -> tuple[list[str], list[tuple]]:
    """Run sql read-only, return (column_names, rows). Raises on error."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout_s)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b
    timer = threading.Timer(timeout_s, conn.interrupt)
    timer.daemon = True
    timer.start()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(64)  # hard cap; we'll truncate further below
        return cols, rows
    finally:
        timer.cancel()
        conn.close()


def _format_observation(cols: list[str], rows: list[tuple], max_rows: int = 20, max_bytes: int = 4000) -> str:
    if not cols:
        return "OK (no result set)."
    head = " | ".join(cols)
    body = []
    for r in rows[:max_rows]:
        body.append(" | ".join("NULL" if v is None else str(v) for v in r))
    out = f"{len(rows)} row(s) returned. First {min(len(rows), max_rows)}:\n{head}\n" + "\n".join(body)
    if len(out) > max_bytes:
        out = out[:max_bytes] + "\n... [truncated]"
    return out


def run_tool(name: str, args: dict, db_path: str | Path, timeout_s: float = 10.0) -> str:
    """Dispatch a tool call. Returns a string observation (success or error)."""
    if name == "execute_sql":
        sql = (args or {}).get("sql", "").strip()
        if not sql:
            return "ERROR: execute_sql requires a non-empty `sql` argument."
        try:
            cols, rows = _execute_readonly(db_path, sql, timeout_s=timeout_s)
        except sqlite3.OperationalError as e:
            msg = str(e)
            if "interrupt" in msg.lower():
                return f"ERROR: query exceeded {timeout_s:.0f}s timeout."
            return f"ERROR: {msg}"
        except Exception as e:  # pragma: no cover
            return f"ERROR: {e!r}"
        return _format_observation(cols, list(rows))
    return f"ERROR: unknown tool `{name}`. Available: execute_sql, submit, keep_baseline_sql."


# ----------------------- tool-call parsing -----------------------

_FENCE_JSON_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_TOOL_RE = re.compile(r'"tool"\s*:\s*"(execute_sql|submit|keep_baseline_sql)"', re.IGNORECASE)
_SQL_ARG_RE = re.compile(r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def _coerce_tool_call(blob: str) -> dict | None:
    """Try strict JSON first; on failure, regex-extract tool+sql so we recover from
    common malformations (unterminated strings, trailing fence, escaped backticks)."""
    blob = blob.strip()
    if not blob:
        return None
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict) and "tool" in obj:
            return {"tool": str(obj["tool"]), "args": obj.get("args") or {}}
    except json.JSONDecodeError:
        pass
    # Regex fallback: tool name + sql string. Handles q0/q5-style broken JSON
    # where the model dropped a `}` or trailing fence into the args.sql value.
    m_tool = _TOOL_RE.search(blob)
    m_sql = _SQL_ARG_RE.search(blob)
    if m_tool:
        sql = ""
        if m_sql:
            # Un-escape JSON string escapes manually since we couldn't parse.
            sql = m_sql.group(1).encode("utf-8").decode("unicode_escape", errors="ignore")
        return {"tool": m_tool.group(1), "args": {"sql": sql}}
    return None


def parse_tool_call(text: str) -> dict | None:
    """Find the first parseable tool call in `text`. Returns {"tool", "args"} or None.

    Robustness: model output sometimes has malformed JSON (unterminated string,
    extra `}`). We try strict json.loads, then a regex fallback that pulls out
    `"tool"` + `"sql"` keys directly. This makes the parser forgiving without
    requiring the model to be perfect.
    """
    for blob in _FENCE_JSON_RE.findall(text):
        call = _coerce_tool_call(blob)
        if call is not None:
            return call
    # Bare top-level JSON (no fence)
    stripped = text.strip()
    if stripped.startswith("{"):
        call = _coerce_tool_call(stripped)
        if call is not None:
            return call
    return None


# ----------------------- driver -----------------------


@dataclass
class AgentTrace:
    question_id: int
    db_id: str
    final_sql: str
    turns: int
    n_tool_calls: int
    n_exec_errors: int
    completion_chars: int  # sum of all assistant message lengths (proxy for tokens)
    finish_reason: str  # "submit" | "no_tool_call" | "budget" | "parse_fail"
    history: list[dict] = field(default_factory=list)  # final messages (sans system)


def build_initial_messages(
    example: BirdExample,
    schema: DatabaseSchema,
    n_samples: int = 3,
    baseline_sql: str = "",
) -> list[dict]:
    schema_block = render_ddl_with_samples(schema, n_samples=n_samples).strip()
    evidence = (example.evidence or "").strip() or "(none provided)"
    baseline_sql_clean = (baseline_sql or "").strip() or "-- (no baseline available)"
    user = AGENT_USER.format(
        schema_block=schema_block,
        evidence=evidence,
        question=example.question.strip(),
        baseline_sql=baseline_sql_clean,
    )
    return [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": user},
    ]


def step_agent(
    messages: list[dict],
    db_path: str | Path,
    *,
    max_turns: int = 6,
    tool_timeout_s: float = 10.0,
    chat_fn=None,
    baseline_sql: str = "",
) -> AgentTrace:
    """Run the agent loop until termination or budget exhaustion.

    `chat_fn(list[messages]) -> str` is injected so this is testable without
    Modal/vLLM (smoke tests can pass a mock function).

    `baseline_sql` is the candidate SQL the baseline produced for this question.
    If the agent calls `keep_baseline_sql`, this string becomes the final SQL.
    """
    assert chat_fn is not None, "chat_fn is required"
    if "question_id" in messages[0]:  # defensive: don't expect this layout
        raise ValueError("messages[0] should be the system message")

    n_tool_calls = 0
    n_exec_errors = 0
    completion_chars = 0
    last_executed_sql = ""
    finish_reason = "budget"
    final_sql = ""

    for turn in range(1, max_turns + 1):
        text = chat_fn(messages)
        completion_chars += len(text)
        messages = messages + [{"role": "assistant", "content": text}]

        call = parse_tool_call(text)
        if call is None:
            # Treat as final answer: try to extract SQL out of the text.
            final_sql = extract_sql(text)
            finish_reason = "no_tool_call"
            break

        if call["tool"] == "submit":
            sql = (call["args"] or {}).get("sql", "").strip()
            final_sql = sql
            finish_reason = "submit"
            break

        if call["tool"] == "keep_baseline_sql":
            final_sql = (baseline_sql or "").strip()
            finish_reason = "keep_baseline"
            break

        if call["tool"] == "execute_sql":
            n_tool_calls += 1
            sql_arg = (call["args"] or {}).get("sql", "")
            last_executed_sql = sql_arg or last_executed_sql
            obs = run_tool("execute_sql", call["args"] or {}, db_path, timeout_s=tool_timeout_s)
            if obs.startswith("ERROR"):
                n_exec_errors += 1
            messages = messages + [{"role": "user", "content": f"[execute_sql output]\n{obs}"}]
            continue

        # Unknown tool: feed back error and continue.
        n_exec_errors += 1
        messages = messages + [{"role": "user", "content": run_tool(call["tool"], call.get("args") or {}, db_path)}]

    if finish_reason == "budget" and not final_sql:
        # Best-effort: prefer baseline (anchored answer) over last-tried draft.
        final_sql = (baseline_sql or "").strip() or last_executed_sql

    # Strip trailing prose / fences / semicolons consistently with extract_sql.
    if final_sql:
        final_sql = extract_sql(final_sql) if "```" in final_sql else (final_sql.strip().rstrip(";") + ";")

    return AgentTrace(
        question_id=-1,  # filled by caller
        db_id="",
        final_sql=final_sql,
        turns=turn,
        n_tool_calls=n_tool_calls,
        n_exec_errors=n_exec_errors,
        completion_chars=completion_chars,
        finish_reason=finish_reason,
        history=[m for m in messages if m["role"] != "system"],
    )
