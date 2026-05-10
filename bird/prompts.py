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


# Keep total user content under this many chars when adding shots. 12k chars
# is a comfortable budget for a 16k-token context window since each shot's
# question + SQL is short relative to the schema block.
_FEWSHOT_CHAR_BUDGET = 12_000


def _format_shot(idx: int, shot: BirdExample) -> str:
    """Render a single retrieved demo. Compact format — schema is shared."""
    hint = shot.evidence.strip() or "(none provided)"
    sql = shot.sql.strip().rstrip(";") + ";"
    return (
        f"### Example {idx}\n"
        f"Question: {shot.question.strip()}\n"
        f"Hint: {hint}\n"
        f"SQL:\n```sql\n{sql}\n```"
    )


def build_messages_with_fewshot(
    example: BirdExample,
    schema: DatabaseSchema,
    shots: list[BirdExample],
    n_samples: int = 3,
) -> list[dict]:
    """Like `build_messages`, but interleaves retrieved demos before the question.

    Layout (single user message):
        ### Database schema (SQLite)
        ```sql ... ```

        ### Examples
        ### Example 1
        Question: ...
        Hint: ...
        SQL: ```sql ... ```
        ### Example 2
        ...

        ### External knowledge / hint
        <evidence>

        ### Question
        <question>

        ### Output
        ...

    Token-budget guard: we drop trailing shots whose inclusion would push the
    user content past `_FEWSHOT_CHAR_BUDGET` chars. Earlier shots win; they're
    already ranked by relevance.
    """
    schema_block = render_ddl_with_samples(schema, n_samples=n_samples).strip()
    evidence = example.evidence.strip() or "(none provided)"

    # Build base user content (without examples block) so we can compute the
    # remaining budget for shots.
    base_user = USER_TEMPLATE.format(
        schema_block=schema_block,
        evidence=evidence,
        question=example.question.strip(),
    )

    # Reserve a small overhead for the "### Examples\n\n" separator + spacing.
    overhead = 32
    remaining = max(_FEWSHOT_CHAR_BUDGET - len(base_user) - overhead, 0)

    rendered_shots: list[str] = []
    used = 0
    for i, shot in enumerate(shots, start=1):
        block = _format_shot(i, shot)
        # +2 for the "\n\n" join between shots
        cost = len(block) + (2 if rendered_shots else 0)
        if used + cost > remaining:
            # Skip this shot rather than truncate its SQL — a half-SQL example
            # would be actively misleading to the model.
            continue
        rendered_shots.append(block)
        used += cost

    if rendered_shots:
        examples_block = "### Examples\n" + "\n\n".join(rendered_shots) + "\n"
        # Splice the examples block immediately before "### External knowledge / hint".
        marker = "### External knowledge / hint"
        idx = base_user.index(marker)
        user = base_user[:idx] + examples_block + "\n" + base_user[idx:]
    else:
        user = base_user

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
