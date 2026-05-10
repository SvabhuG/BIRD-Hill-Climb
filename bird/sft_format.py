"""SFT-Base prompt + completion format (flat, no chat wrapping).

Training and inference share `build_sft_prompt` so the SFT loss and the eval
prompt are bytewise-identical. The format is intentionally flat (no
SYSTEM:/USER:/ASSISTANT: tags) because the Base model wasn't pretrained on
those shouting tokens — our v1 SFT used `messages_to_raw_text`'s chat wrapping
and hit a 16.6% exec_error rate dominated by a specific stray-paren artifact,
which we attribute to the Base model learning weird patterns around the
unfamiliar chat tokens instead of clean SQL structure.

Schema rendering keeps our existing `render_ddl_with_samples` for matrix
consistency with the baseline cells. Preamble is a short 4-rule instruction.
SFT target is raw SQL + EOS (no fences) — fences add another unfamiliar token
pattern for the Base to learn.
"""
from __future__ import annotations

from .data import BirdExample
from .schema import DatabaseSchema, render_ddl_with_samples


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


def build_sft_prompt(example: BirdExample, schema: DatabaseSchema, n_samples: int = 3) -> str:
    """Flat training/inference prompt. Ends with `### SQL\\n` so the model
    continues with SQL immediately. The trailing single newline is load-bearing:
    it matches the boundary the model saw in every training example.
    """
    schema_block = render_ddl_with_samples(schema, n_samples=n_samples).strip()
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
