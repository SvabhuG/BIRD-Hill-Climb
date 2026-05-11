"""SFT-Base prompt + completion format (flat, no chat wrapping, rich schema).

Training and inference share `build_sft_prompt` so the SFT loss and the eval
prompt are bytewise-identical.

Schema rendering uses `bird.exp18_schema.format_profile` — DDL + row counts +
sample rows + distinct-value lists for low-cardinality columns + explicit
foreign-key section. The distinct-value lists are load-bearing: the v3 SFT
retry confirmed that the prompt-format wrapping is NOT the cause of our
exec_error tail; the remaining hypothesis is the schema-content gap. Without
the distinct-value lists, the SFT'd model has to guess valid filter values
from 3 sample rows and frequently gets them wrong.

Preamble is the short 4-rule instruction (matches the proven recipe's V1).
SFT target is raw SQL + EOS, no fences.
"""
from __future__ import annotations

from .data import BirdExample


INSTRUCTION_PREAMBLE = """You are an expert SQLite SQL developer. Given a database schema and a natural language question, write a SQL query that answers the question.

Rules:
- Write SQLite-compatible SQL only
- Use backticks for column/table names with spaces or special characters
- Return ONLY the SQL query, no explanations or markdown
- Use the hint/evidence to understand domain-specific terminology"""


# vLLM stops: the Base sometimes continues with a hallucinated "### Question"
# section after the SQL on examples that didn't see a clean EOS during training.
# These stops match the section-header pattern.
INFERENCE_STOPS = ["\n\n###", "\n###"]


def build_sft_prompt(example: BirdExample, profile: dict) -> str:
    """Flat training/inference prompt. Ends with `### SQL\\n` so the model
    continues with SQL immediately. The trailing single newline is load-bearing:
    it matches the boundary the model saw in every training example.

    `profile` is the dict produced by `bird.exp18_schema.profile_database` —
    callers cache this per db_id (sqlite reads are cheap but distinct-value
    scans are O(row_count) per column).
    """
    from .exp18_schema import format_profile

    schema_block = format_profile(profile).strip()
    parts = [
        "### Database Schema",
        schema_block,
        "",
        "### Question",
        example.question.strip(),
    ]
    if example.evidence and example.evidence.strip():
        parts.extend(["", "### Hint", example.evidence.strip()])
    parts.extend(["", "### SQL"])
    body = "\n".join(parts)
    return INSTRUCTION_PREAMBLE + "\n\n" + body + "\n"


def build_sft_completion(gold_sql: str, eos_token: str) -> str:
    """Raw SQL + EOS, no fences. `extract_sql`'s SELECT/WITH-anchored fallback
    parses this cleanly without needing the fenced-block branch.
    """
    sql = gold_sql.strip().rstrip(";").strip()
    return sql + ";" + eos_token
