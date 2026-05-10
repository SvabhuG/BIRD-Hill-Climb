"""Plan-then-SQL chain-of-thought — two-stage prompting.

Stage 1 ("plan"): the model receives the schema, evidence, and question, and is
asked to write a short natural-language plan (which tables to join, which
columns to select/filter/aggregate, how to apply BIRD evidence). No SQL yet.

Stage 2 ("sql"): the model receives the same schema/evidence/question PLUS the
plan from stage 1, and is asked to emit a single SQLite SELECT.

We keep the two prompts separate so we can:
  * cap stage-1 max_tokens cheaply (no SQL body to emit)
  * tune temperatures independently (slightly higher for plan, 0.0 for SQL)
  * inspect plans separately for failure analysis (the plan is a passthrough field)

`build_sql_with_plan_messages` mirrors `build_messages` from `prompts.py` so the
SQL extractor (`extract_sql`) and downstream eval don't need to change.
"""
from __future__ import annotations

import re

from .data import BirdExample
from .schema import DatabaseSchema, render_ddl_with_samples


# ---------- Stage 1: plan ----------

PLAN_SYSTEM_PROMPT = (
    "You are a senior data analyst planning a SQLite query. "
    "Given a schema, an optional hint, and a natural-language question, "
    "you write a short plan describing which tables to join, which columns to "
    "select/filter/aggregate, and how to apply the hint. Do not write SQL yet."
)


PLAN_USER_TEMPLATE = """\
### Database schema (SQLite)
```sql
{schema_block}
```

### External knowledge / hint
{evidence}

### Question
{question}

### Output
Write a 3-6 sentence plan in plain English describing how to answer the \
question against this schema. Mention the tables to use, the join keys (if \
any), the columns to select, any filters, ordering, grouping, or aggregation, \
and how the hint should be applied. Output the plan only. Do not write SQL.
"""


def build_plan_messages(
    example: BirdExample, schema: DatabaseSchema, n_samples: int = 3
) -> list[dict]:
    """Stage-1 chat messages: ask the model for a natural-language plan."""
    schema_block = render_ddl_with_samples(schema, n_samples=n_samples).strip()
    evidence = example.evidence.strip() or "(none provided)"
    user = PLAN_USER_TEMPLATE.format(
        schema_block=schema_block,
        evidence=evidence,
        question=example.question.strip(),
    )
    return [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------- Stage 2: SQL conditioned on the plan ----------

SQL_FROM_PLAN_SYSTEM_PROMPT = (
    "You are a senior data analyst who writes correct, idiomatic SQLite SQL. "
    "You are given a database schema, a question, an optional hint, and a plan "
    "you already drafted. Follow the plan when it is correct; deviate only if "
    "the plan is wrong. Return one SQL query that answers the question."
)


SQL_FROM_PLAN_USER_TEMPLATE = """\
### Database schema (SQLite)
```sql
{schema_block}
```

### External knowledge / hint
{evidence}

### Question
{question}

### Plan
{plan}

### Output
Write a single SQLite SELECT statement that answers the question, using only \
the tables and columns shown above and following the plan. Wrap your final SQL \
in a fenced ```sql ... ``` block. Do not include explanations after the SQL.
"""


# Cap plan length before injecting into stage 2 — very long plans are usually
# rambly model output rather than useful planning, and they eat into the token
# budget we'd rather spend on the SQL body and the schema.
_MAX_PLAN_CHARS = 1500


def build_sql_with_plan_messages(
    example: BirdExample,
    schema: DatabaseSchema,
    plan_text: str,
    n_samples: int = 3,
) -> list[dict]:
    """Stage-2 chat messages: same context as stage 1 plus the plan, ask for SQL."""
    schema_block = render_ddl_with_samples(schema, n_samples=n_samples).strip()
    evidence = example.evidence.strip() or "(none provided)"

    plan = (plan_text or "").strip()
    if not plan:
        plan = "(no plan produced — answer the question directly)"
    elif len(plan) > _MAX_PLAN_CHARS:
        plan = plan[:_MAX_PLAN_CHARS].rstrip() + " …"

    user = SQL_FROM_PLAN_USER_TEMPLATE.format(
        schema_block=schema_block,
        evidence=evidence,
        question=example.question.strip(),
        plan=plan,
    )
    return [
        {"role": "system", "content": SQL_FROM_PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------- Plan extraction ----------

# Match a fenced code block (with or without a language tag) — we strip these
# out of plan output. A plan that is entirely SQL inside a fence is treated as
# empty (the model failed to plan; let stage 2 work without one).
_PLAN_FENCE_RE = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)

_PLAN_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"sure[!,. ]*|"
    r"certainly[!,. ]*|"
    r"of course[!,. ]*|"
    r"absolutely[!,. ]*|"
    r"here(?:'s| is)(?: the| my| a)?(?: plan)?[:.\s]*|"
    r"plan[:\s]*"
    r")",
    re.IGNORECASE,
)


def extract_plan(generated: str) -> str:
    """Pull a clean natural-language plan out of a model response.

    Steps:
      1. Strip code fences. If the fenced content is the only substantive
         output (e.g. the model emitted SQL instead of a plan), drop it — the
         caller treats an empty plan as "no plan available".
      2. Strip common chatty preambles ("Sure! Here's the plan:" etc.).
      3. Strip leading/trailing whitespace.
    """
    if not generated:
        return ""

    text = generated

    # 1. Remove code fences. We treat the *fenced* content as not-a-plan;
    # whatever non-fenced prose surrounds them is the plan.
    text = _PLAN_FENCE_RE.sub("", text)

    text = text.strip()
    if not text:
        return ""

    # 2. Strip chatty preambles, possibly more than one stacked
    # ("Sure! Here's the plan: ...").
    while True:
        m = _PLAN_PREAMBLE_RE.match(text)
        if not m:
            break
        new_text = text[m.end():].lstrip()
        if new_text == text:
            break
        text = new_text

    return text.strip()
