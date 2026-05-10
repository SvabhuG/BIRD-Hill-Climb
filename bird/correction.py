"""Self-correction prompts: hand the model its failed SQL + the SQLite error
and ask for a fix.

This module is the prompt-side of the loop only. The orchestration (first eval
pass -> filter exec_error -> retry -> re-eval) lives in `modal_app.py` because
it's coupled to Modal volumes / functions.

Why a dedicated system prompt: the original SYSTEM_PROMPT frames the task as
"write SQL from scratch". For correction we want the model to focus on the
specific failure mode, not re-derive the answer. Empirically, error-message-as-
context is the strongest signal we can give the model — keep it prominent.
"""
from __future__ import annotations

from .data import BirdExample
from .schema import DatabaseSchema, render_ddl_with_samples


CORRECTION_SYSTEM_PROMPT = (
    "You are a senior data analyst fixing a broken SQLite query. "
    "You will see the database schema, the original question, your previous SQL "
    "attempt, and the SQLite error it produced. Output one corrected SQLite "
    "SELECT statement that resolves the error and answers the question."
)


CORRECTION_USER_TEMPLATE = """\
### Database schema (SQLite)
```sql
{schema_block}
```

### External knowledge / hint
{evidence}

### Question
{question}

### Previous SQL attempt (failed)
```sql
{failed_sql}
```

### SQLite error
```
{error_msg}
```

### Common SQLite gotchas to check
- Column or table names with spaces/keywords need backticks (e.g. `Funding Type`).
- The column may live on a different table than the join used — verify it against the schema above.
- Integer division: cast one side to REAL (`CAST(x AS REAL) / y`) to get fractions/percentages.

### Output
Write a single corrected SQLite SELECT statement. Use only tables and columns shown above. \
Wrap your final SQL in a fenced ```sql ... ``` block. Do not include explanations after the SQL.
"""


def build_correction_messages(
    example: BirdExample,
    schema: DatabaseSchema,
    failed_sql: str,
    error_msg: str,
    n_samples: int = 3,
) -> list[dict]:
    """Build chat messages for a self-correction retry on an exec_error case.

    The schema is re-included verbatim — even though the model saw it on the
    first pass, retries are stateless across calls and the error often refers
    to a column the model needs to re-locate in the DDL.
    """
    schema_block = render_ddl_with_samples(schema, n_samples=n_samples).strip()
    evidence = example.evidence.strip() or "(none provided)"
    # `failed_sql` is intentionally not truncated — model needs the whole query
    # to identify the broken clause. `error_msg` may be multiline; pass through.
    user = CORRECTION_USER_TEMPLATE.format(
        schema_block=schema_block,
        evidence=evidence,
        question=example.question.strip(),
        failed_sql=(failed_sql or "").strip() or "(no SQL produced)",
        error_msg=(error_msg or "").strip() or "(no error message)",
    )
    return [
        {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
