"""Prompt templates and SQL extraction.

Phase 1 keeps the prompt deliberately plain so later gains are attributable to
specific scaffolding (schema linking, value indices, self-correction) rather than
prompt-engineering noise.
"""
from __future__ import annotations

import re

from .data import BirdExample
from .schema import DatabaseSchema, render_ddl_with_samples


SYSTEM_PROMPT = (
    "You are a senior data analyst who writes correct, idiomatic SQLite SQL. "
    "You are given a database schema, a natural-language question, and an optional hint. "
    "Return one SQL query that answers the question against this schema."
)


USER_TEMPLATE = """\
### Database schema (SQLite)
```sql
{schema_block}
```

### External knowledge / hint
{evidence}

### Question
{question}

### Output
Write a single SQLite SELECT statement that answers the question, using only the tables and \
columns shown above. Wrap your final SQL in a fenced ```sql ... ``` block. Do not include \
explanations after the SQL.
"""


def messages_to_raw_text(messages: list[dict]) -> str:
    """Render a chat-message list into a single string for base-model completion.

    Base models (e.g. Qwen2.5-Coder-32B without -Instruct) don't have a chat
    template; calling vLLM's `chat()` would either fail or apply a default the
    model wasn't trained for. Concatenating into a plain prompt with simple role
    tags plus an explicit `ASSISTANT:` cue is the conventional fallback.
    """
    parts: list[str] = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        parts.append(f"{role}:\n{m['content'].strip()}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def build_messages(example: BirdExample, schema: DatabaseSchema, n_samples: int = 3) -> list[dict]:
    """Build a chat-format message list for an instruct model."""
    schema_block = render_ddl_with_samples(schema, n_samples=n_samples).strip()
    evidence = example.evidence.strip() or "(none provided)"
    user = USER_TEMPLATE.format(
        schema_block=schema_block,
        evidence=evidence,
        question=example.question.strip(),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_FENCE_RE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(generated: str) -> str:
    """Pull a SQL string out of a model response.

    Order of attempts:
      1. ```sql ... ``` fenced block (last one wins — models sometimes restate)
      2. ``` ... ``` unlabeled fenced block
      3. The raw text, stripped, with leading prose lines dropped if obvious
    """
    fences = _FENCE_RE.findall(generated)
    if fences:
        return fences[-1].strip().rstrip(";").strip() + ";"

    # Strip common preambles and trailing prose.
    text = generated.strip()
    # If a "SELECT" / "WITH" appears, anchor on the last one.
    m = re.search(r"\b(WITH|SELECT)\b.*", text, re.IGNORECASE | re.DOTALL)
    if m:
        sql = m.group(0).strip()
        # If the model continues with prose after a semicolon, cut at first `;`.
        if ";" in sql:
            sql = sql.split(";")[0]
        return sql.strip().rstrip(";") + ";"
    return text.rstrip(";") + ";"
