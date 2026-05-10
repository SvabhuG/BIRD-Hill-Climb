"""BIRD-to-chat formatting for supervised fine-tuning.

Goal: produce (user, assistant) pairs whose distribution matches eval-time
exactly. We reuse `SYSTEM_PROMPT` + `USER_TEMPLATE` + `build_messages` from
`bird.prompts`, then append an assistant turn that wraps the gold SQL in a
```sql ... ``` fence — the same shape `extract_sql` recovers at eval time.

This module has zero Modal/Unsloth/torch deps so it can run locally during
smoke tests and on Modal during SFT.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .data import BirdExample, load_split
from .prompts import build_messages, extract_sql
from .schema import DatabaseSchema, extract_schema


# ---------- assistant target ----------

def format_gold_sql_as_assistant(gold_sql: str) -> str:
    """Wrap gold SQL in a ```sql ... ``` fenced block matching the prompt's
    "Output" instruction.

    We strip trailing whitespace + semicolons and re-add a single trailing `;`
    so the assistant target is deterministic and `extract_sql` round-trips
    cleanly (it appends `;` and strips inner ones — see bird/prompts.py).
    """
    sql = (gold_sql or "").strip().rstrip(";").strip()
    # Defensive: collapse Windows-style line endings the BIRD dump occasionally has.
    sql = sql.replace("\r\n", "\n").replace("\r", "\n")
    return f"```sql\n{sql};\n```"


def canonical_sql(gold_sql: str) -> str:
    """The canonical form `extract_sql` will recover from the formatted target.

    Used by the smoke test and as the expected SQL during eval.
    """
    return (gold_sql or "").strip().rstrip(";").strip() + ";"


# ---------- chat-format example ----------

def format_example_for_sft(
    example: BirdExample,
    schema: DatabaseSchema,
    n_samples: int = 3,
) -> dict:
    """Format one BIRD example as a chat-style training pair.

    Returns:
        {"messages": [{"role": "system", ...},
                      {"role": "user", ...},
                      {"role": "assistant", "content": "```sql\\n<gold>;\\n```"}],
         "question_id": int,
         "db_id": str,
         "difficulty": str | None}
    """
    if not example.sql:
        raise ValueError(
            f"example {example.question_id} (db={example.db_id}) has no gold SQL; "
            "SFT requires labelled examples — filter test split out before formatting."
        )

    prompt_msgs = build_messages(example, schema, n_samples=n_samples)
    assistant = {"role": "assistant", "content": format_gold_sql_as_assistant(example.sql)}
    return {
        "messages": [*prompt_msgs, assistant],
        "question_id": example.question_id,
        "db_id": example.db_id,
        "difficulty": example.difficulty,
    }


# ---------- dataset builder ----------

@dataclass
class BuildStats:
    n_total: int = 0
    n_emitted: int = 0
    n_missing_db: int = 0
    n_missing_gold: int = 0
    n_schema_errors: int = 0


def iter_sft_examples(
    train_root: str | Path,
    n_samples: int = 3,
    split_name: str = "train",
    limit: int | None = None,
) -> Iterator[tuple[dict, BuildStats]]:
    """Stream formatted SFT examples, caching schemas per db_id.

    Yields (formatted, running_stats) tuples so the caller can checkpoint /
    log progress without buffering the whole dataset in memory. The final
    `stats` is also returned via the last yield's second element.
    """
    sp = load_split(train_root, name=split_name)
    examples: Iterable[BirdExample] = sp.examples
    if limit:
        examples = sp.examples[:limit]

    schema_cache: dict[str, DatabaseSchema] = {}
    stats = BuildStats()

    for ex in examples:
        stats.n_total += 1
        if not ex.sql:
            stats.n_missing_gold += 1
            continue

        if ex.db_id not in schema_cache:
            db_path = sp.db_path(ex.db_id)
            if not db_path.exists():
                stats.n_missing_db += 1
                continue
            try:
                schema_cache[ex.db_id] = extract_schema(db_path, ex.db_id, n_samples=n_samples)
            except Exception as e:  # noqa: BLE001 — sqlite3 raises a wide family
                stats.n_schema_errors += 1
                print(f"[sft_format] schema error for {ex.db_id}: {e!r}")
                continue

        try:
            yield format_example_for_sft(ex, schema_cache[ex.db_id], n_samples=n_samples), stats
            stats.n_emitted += 1
        except Exception as e:  # noqa: BLE001
            stats.n_schema_errors += 1
            print(f"[sft_format] format error qid={ex.question_id} db={ex.db_id}: {e!r}")


def build_sft_dataset(
    train_root: str | Path,
    n_samples: int = 3,
    split_name: str = "train",
    limit: int | None = None,
) -> tuple[list[dict], BuildStats]:
    """Materialize the full SFT dataset in memory.

    Returns (examples, stats). For 9k BIRD train this fits comfortably in RAM
    (each example is ~5-15 KB of text).
    """
    out: list[dict] = []
    stats = BuildStats()
    for formatted, stats in iter_sft_examples(train_root, n_samples=n_samples,
                                              split_name=split_name, limit=limit):
        out.append(formatted)
    return out, stats


def write_sft_jsonl(
    examples: list[dict],
    out_path: str | Path,
) -> Path:
    """Write formatted examples to a JSONL file (one JSON object per line).

    HF `datasets.load_dataset("json", data_files=...)` consumes this directly.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return out_path


# ---------- roundtrip self-check ----------

def assistant_roundtrips(formatted_example: dict) -> bool:
    """True iff `extract_sql(assistant_content) == canonical_sql(gold)`.

    The smoke test relies on this — if False, the SFT format would teach the
    model an output shape `extract_sql` can't recover.
    """
    asst = next(m for m in formatted_example["messages"] if m["role"] == "assistant")
    # We need to recover the *original* gold; we encoded it via canonical_sql,
    # so recompute from the assistant content:
    recovered = extract_sql(asst["content"])
    # Compare against the canonical form of the SQL inside the fence
    # (which is what we'd compare against at eval time too).
    fence_inner = asst["content"].split("```sql", 1)[1].rsplit("```", 1)[0].strip()
    expected = canonical_sql(fence_inner)
    return recovered == expected
