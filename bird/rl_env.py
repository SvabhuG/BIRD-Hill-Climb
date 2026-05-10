"""Verifiers environment for BIRD execution-reward GRPO.

This module exposes a `load_environment(...)` factory matching the Prime
Intellect `verifiers` library convention (see e.g. verifiers/environments/gsm8k).
The verifiers + prime-rl stack discovers environment packages via this
top-level function — the orchestrator imports the module, calls
`load_environment(**args_from_toml)`, and treats the returned `vf.Environment`
as the rollout source.

Reward semantics — deliberately identical to our eval-time machinery:

  reward(rollout) = 1.0  iff  extract_sql(completion) executes against the
                              gold sqlite DB and `_rows_equal(pred_rows,
                              gold_rows)` is True
                  = 0.0  otherwise (syntax error, timeout, wrong rows, empty
                              completion, gold-side errors)

The point of reusing `bird/eval.py::_execute` and `_rows_equal` here is that
the training-time reward signal *is* the same scoring function we use to
report EX on dev. There's no drift between "what the model is rewarded for"
and "what we measure" — a class of bug we don't want.

Single-turn environment: prompt -> SQL completion -> reward. No tool use,
no multi-turn rollouts. This mirrors Arctic-R1's recipe (binary execution
reward, no length penalty, no intermediate format reward, no KL beyond the
GRPO default).

We deliberately keep the verifiers import lazy so this module remains
importable without the `rl` optional dependency group installed — the smoke
test exercises the reward function in isolation.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .data import BirdExample, load_split
from .eval import _execute, _rows_equal
from .prompts import SYSTEM_PROMPT, build_messages, extract_sql
from .schema import extract_schema


def compute_reward(
    completion: Any,
    info: dict | str,
    exec_timeout_s: float = 15.0,
) -> float:
    """Pure reward function — independent of the verifiers library.

    Inputs match the verifiers convention:
      * `completion`: either a string (text completion) or a list of chat
        messages; we look at the last assistant message's `content` if it's
        a list.
      * `info`: dict (or JSON string) carrying `db_path` and `gold_sql`.

    Returns 1.0 on result-set match, 0.0 otherwise. All exceptions caught —
    broken SQL gets reward 0, never a crash.
    """
    if isinstance(info, str):
        info = json.loads(info)

    # Pull SQL text out of either chat-format or string completions.
    if isinstance(completion, list) and completion:
        last = completion[-1]
        text = last.get("content", "") if isinstance(last, dict) else str(last)
    else:
        text = str(completion or "")

    predicted_sql = extract_sql(text)
    if not predicted_sql or not predicted_sql.strip():
        return 0.0

    db_path = info.get("db_path")
    gold_sql = info.get("gold_sql", "")
    if not db_path or not gold_sql:
        return 0.0

    # Execute gold first. If gold itself fails (corpus issue), don't punish.
    try:
        gold_rows = _execute(db_path, gold_sql, timeout_s=exec_timeout_s)
    except (sqlite3.OperationalError, Exception):
        return 0.0

    try:
        pred_rows = _execute(db_path, predicted_sql, timeout_s=exec_timeout_s)
    except (sqlite3.OperationalError, Exception):
        return 0.0

    try:
        return 1.0 if _rows_equal(pred_rows, gold_rows) else 0.0
    except Exception:
        return 0.0


def build_sample(
    example: BirdExample,
    db_path: str | Path,
    n_samples_schema: int = 3,
    schema_cache: dict | None = None,
) -> dict:
    """Build one verifiers-format sample dict.

    Returns:
        {
          "prompt": [{"role": ...}, ...],   # chat messages, ready for the model
          "info":   <json-string>           # passes db_path + gold_sql to reward
        }

    `info` is JSON-serialized because verifiers' DataLoader (HF `datasets`)
    coerces non-primitive columns; using a JSON string is the documented
    pattern for arbitrary structured metadata (see verifiers docs).
    """
    db_path = Path(db_path)
    if schema_cache is not None and example.db_id in schema_cache:
        schema = schema_cache[example.db_id]
    else:
        schema = extract_schema(db_path, example.db_id, n_samples=n_samples_schema)
        if schema_cache is not None:
            schema_cache[example.db_id] = schema

    messages = build_messages(example, schema, n_samples=n_samples_schema)
    info = {
        "db_path": str(db_path),
        "gold_sql": example.sql,
        "db_id": example.db_id,
        "question_id": example.question_id,
        "difficulty": example.difficulty,
    }
    return {"prompt": messages, "info": json.dumps(info)}


def build_dataset(
    train_root: str | Path,
    split: str = "train",
    limit: int = 0,
    n_samples_schema: int = 3,
) -> list[dict]:
    """Materialize a list[dict] of (prompt, info) samples for the verifiers env.

    Schema extraction is cached per db_id within this call.
    """
    sp = load_split(Path(train_root), name=split)
    examples = sp.examples[:limit] if limit else sp.examples
    schema_cache: dict = {}
    out: list[dict] = []
    for ex in examples:
        # Skip examples without a gold SQL (test split, etc.) — reward is undefined.
        if not ex.sql or not ex.sql.strip():
            continue
        db_path = sp.db_path(ex.db_id)
        if not Path(db_path).exists():
            continue
        out.append(build_sample(ex, db_path, n_samples_schema=n_samples_schema, schema_cache=schema_cache))
    return out


def load_environment(
    train_root: str = "/data/bird/train",
    split: str = "train",
    eval_root: str | None = None,
    eval_split: str = "dev",
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    n_samples_schema: int = 3,
    exec_timeout_s: float = 15.0,
    system_prompt: str = SYSTEM_PROMPT,
    **kwargs,
):
    """Verifiers entrypoint — called by prime-rl / `prime eval run` with TOML args.

    Returns a `vf.SingleTurnEnv` whose rubric scores rollouts with
    `compute_reward` (binary execution accuracy).

    Args:
        train_root: directory containing `<split>.json` and `<split>_databases/`.
        split: which split to use for training rollouts (typically "train").
        eval_root: optional path for an evaluation split. Defaults to `train_root`.
        eval_split: name of the eval split (default "dev").
        num_train_examples: cap on training samples; -1 = all.
        num_eval_examples: cap on eval samples; -1 = all.
        n_samples_schema: how many sample rows to embed per table in the
                          schema block of the prompt.
        exec_timeout_s: SQLite execution timeout per rollout (gold + pred).
        system_prompt: override the default BIRD system prompt.
    """
    # Lazy imports — keep the module CPU-importable without verifiers installed.
    import verifiers as vf
    from datasets import Dataset

    def _build_train() -> "Dataset":  # noqa: F821
        samples = build_dataset(
            train_root,
            split=split,
            limit=0 if num_train_examples < 0 else num_train_examples,
            n_samples_schema=n_samples_schema,
        )
        return Dataset.from_list(samples)

    def _build_eval() -> "Dataset" | None:  # noqa: F821
        root = eval_root or train_root
        try:
            samples = build_dataset(
                root,
                split=eval_split,
                limit=0 if num_eval_examples < 0 else num_eval_examples,
                n_samples_schema=n_samples_schema,
            )
        except FileNotFoundError:
            # No eval split available — that's fine, training will use the train dataset for eval.
            return None
        return Dataset.from_list(samples)

    async def execution_accuracy(completion, info) -> float:
        """Binary reward: 1.0 if predicted SQL's result rows match gold's."""
        return compute_reward(completion, info, exec_timeout_s=exec_timeout_s)

    rubric = vf.Rubric(funcs=[execution_accuracy])

    return vf.SingleTurnEnv(
        dataset=_build_train,
        eval_dataset=_build_eval,
        system_prompt=system_prompt,
        rubric=rubric,
    )
