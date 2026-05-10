"""Audit BIRD-dev prompt token counts against the 16384 max_seq_len budget.

Did we silently truncate any prompt during the baseline runs? vLLM's V1 engine
will warn if a prompt exceeds `max_model_len`, but those warnings can be missed
in a long log. This script computes per-question token counts using the actual
Qwen2.5-Coder-7B-Instruct tokenizer (same family as our 32B Qwen2.5 variants;
Qwen3 uses the same tokenizer too) and reports the distribution.

Run from repo root after the dev volume has been synced to /tmp/bird-audit:
    .venv/bin/python -m scripts.audit_prompt_sizes
"""
from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird.data import load_split  # noqa: E402
from bird.prompts import build_messages  # noqa: E402
from bird.schema import extract_schema  # noqa: E402

# vLLM was run with max_model_len=16384 and max_tokens=1024 → input budget ~15360 tokens.
MAX_MODEL_LEN = 16384
MAX_OUTPUT = 1024
INPUT_BUDGET = MAX_MODEL_LEN - MAX_OUTPUT

DEFAULT_TOKENIZER = "Qwen/Qwen2.5-Coder-7B-Instruct"


def _percentile(xs: list[int], p: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER, trust_remote_code=True)
    print(f"[audit] tokenizer: {DEFAULT_TOKENIZER} (vocab={tok.vocab_size})")

    sp = load_split("/tmp/bird-audit/dev", name="dev")
    print(f"[audit] dev examples: {len(sp.examples)}")

    schema_cache: dict[str, object] = {}
    per_db: dict[str, list[int]] = {}
    all_lens: list[int] = []
    over_budget: list[tuple[int, str, int]] = []  # (qid, db_id, n_tokens)

    for ex in sp.examples:
        if ex.db_id not in schema_cache:
            schema_cache[ex.db_id] = extract_schema(sp.db_path(ex.db_id), ex.db_id, n_samples=3)
        msgs = build_messages(ex, schema_cache[ex.db_id], n_samples=3)
        # Apply the chat template (matches what vLLM does internally on chat()).
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        n_tokens = len(tok.encode(rendered, add_special_tokens=False))
        per_db.setdefault(ex.db_id, []).append(n_tokens)
        all_lens.append(n_tokens)
        if n_tokens > INPUT_BUDGET:
            over_budget.append((ex.question_id, ex.db_id, n_tokens))

    print(f"\n=== OVERALL  ({len(all_lens)} prompts; budget = {INPUT_BUDGET} tokens) ===")
    print(f"  min:    {min(all_lens):>6}")
    print(f"  median: {int(median(all_lens)):>6}")
    print(f"  mean:   {int(mean(all_lens)):>6}")
    print(f"  p95:    {_percentile(all_lens, 0.95):>6}")
    print(f"  p99:    {_percentile(all_lens, 0.99):>6}")
    print(f"  max:    {max(all_lens):>6}")
    print(f"  over budget: {len(over_budget)} / {len(all_lens)}  ({len(over_budget)/len(all_lens):.2%})")

    print(f"\n=== BY DB  (sorted by max-prompt-tokens, desc) ===")
    db_rows = []
    for db, lens in per_db.items():
        db_rows.append((db, len(lens), max(lens), int(mean(lens)), _percentile(lens, 0.95)))
    db_rows.sort(key=lambda r: -r[2])
    print(f"  {'db_id':<26s} {'n':>5s} {'max':>6s} {'mean':>6s} {'p95':>6s}")
    for row in db_rows:
        flag = " ⚠" if row[2] > INPUT_BUDGET else ""
        print(f"  {row[0]:<26s} {row[1]:>5d} {row[2]:>6d} {row[3]:>6d} {row[4]:>6d}{flag}")

    if over_budget:
        print(f"\n=== OVER-BUDGET QUESTIONS ({len(over_budget)}) ===")
        for qid, db, n in sorted(over_budget, key=lambda x: -x[2])[:30]:
            print(f"  q{qid:<5d}  {db:<26s}  {n} tokens (over by {n - INPUT_BUDGET})")
        if len(over_budget) > 30:
            print(f"  ... and {len(over_budget) - 30} more")

    print(f"\n=== VERDICT ===")
    if not over_budget:
        print(f"  ✓ All {len(all_lens)} dev prompts fit comfortably within the {INPUT_BUDGET}-token budget.")
        print(f"  Headroom from p99 to budget: {INPUT_BUDGET - _percentile(all_lens, 0.99)} tokens.")
    else:
        print(f"  ⚠ {len(over_budget)} prompts exceeded budget — vLLM either truncated or rejected them.")
        print(f"  Action: bump max_model_len, or apply schema linking (Phase 2) to compress the schema block.")


if __name__ == "__main__":
    main()
