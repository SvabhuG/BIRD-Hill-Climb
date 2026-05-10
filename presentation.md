# BIRD Hill-Climbing — Presentation Narrative

> Living document. Updated as runs land. Source for the 10-min informal presentation.

## The setup

- **Goal:** maximize execution accuracy (EX) on BIRD dev (1,534 questions, 11 DBs, 37+ domains) with an open-weight model ≤32B params.
- **Resources:** Modal + 8× B200 GPU, BIRD train (~12.7k) + dev. Anything goes — scaffolding, inference-time scaling, post-training.
- **Strategic bet:** inference-time scaffolding *before* RL. Why? Published BIRD ablations show majority-vote-on-execution and schema linking yield big gains-per-hour; RL is days, scaffolding is hours.

## Journey

### Phase 0 — foundation (≈60 minutes)
- Modal app: persistent volumes (`bird-data`, `hf-cache`, `bird-results`), `Inference` vLLM-on-B200 class, multi-process eval (`sqlite3.Connection.interrupt()` via `threading.Timer` — only thread-safe way to abort runaway SQL).
- **Eval status taxonomy** (the single most important early decision): `correct / wrong / exec_error / gold_error / timeout / empty`. Drives every later prioritization — correction targets `exec_error`, voting targets `wrong+exec_error`, linking targets schema confusion.
- Local-only smoke tests so iteration is free (no GPU/Modal cost to run).
- **Side trip — vllm/transformers version-pinning trap (~30 min lost):** burned through 3 failed Modal runs guessing `(vllm, transformers)` pins. Lesson: web-search the exact error before guessing. Saved as feedback memory so it never happens again.

### Phase 1 — baseline + observability
- Prompt format: full schema (DDL from `sqlite_master` + 3 sample rows per table) + BIRD evidence + question. Greedy decoding, n=1.
- **Anchor:** Qwen2.5-Coder-7B-Instruct → **EX = 47.85% (734/1,534)**, breakdown 42% wrong / 10% exec_error / 0.1% timeout.
- Failure-analysis tools: `inspect_failures.py` (per-status/DB/difficulty filter, side-by-side gold), `compare_runs.py` (per-DB matrix + pairwise wins + ensemble ceiling).
- **Prompt-budget audit:** confirmed all 1,534 dev prompts fit (max 5,371 tokens vs 15,360 budget; ~10K tokens of headroom). No truncation in any non-thinking run.

### Phase 2 — 32B model sweep (5 variants in parallel)

| model | gen | architecture | EX | exec_err |
|---|---|---|---:|---:|
| Qwen2.5-Coder-7B-Instruct | 2.5 | dense, coder | 47.85% | 150 |
| Qwen2.5-Coder-32B (base, raw-completion) | 2.5 | dense, coder, base | 47.26% | — |
| Qwen2.5-32B-Instruct (general) | 2.5 | dense, general | 51.37% | — |
| Qwen2.5-Coder-32B-Instruct | 2.5 | dense, coder | 57.37% | 38 |
| **Qwen3-Coder-30B-A3B-Instruct** | 3 | **MoE, coder** | **60.63%** | 25 |
| Qwen3-32B (dense, thinking-on, max_tokens=1024) | 3 | dense, general | 37.55% ⚠ | 623 ⚠ — *truncation artifact* |
| Qwen3-32B (dense, thinking-on, max_tokens=8192) | 3 | dense, general | _running_ | — |
| Qwen3.6-27B | 3.6 | dense, general | _agent debugging vllm pin_ | — |

**Insights from the sweep:**
- **Generation gap dominates** at 32B-class: Qwen3 → Qwen2.5 closes ~3.3pp on the same coder-instruct base.
- **Coder-tuning matters** at the Qwen2.5 generation: Coder-32B-Instruct beats general-32B-Instruct by 6pp.
- **Better models concentrate failures away from exec_error**: 150 → 38 → 25 going from 7B-Instruct → 32B-Coder-Instruct → 30B-Coder-MoE. Larger/newer models write more syntactically-valid SQL out of the box.
- **Thinking on Qwen3 needs ≥8192 max_tokens**: the 1024 default cuts off SQL after the reasoning trace. Documented as a configuration trap.

### Phase 3 — inference-time strategies (4 parallel git worktrees, 4 subagents)

Each strategy implemented in an isolated worktree by an autonomous subagent, complete with smoke tests + a Modal entrypoint (`run_with_<strategy>`). Branches: `feat/voting`, `feat/correction`, `feat/cot`, `feat/fewshot`. Schema linking lives on `main` (`run_with_linking`).

| strategy | branch | how it works | predicted Δ on 7B (47.85%) |
|---|---|---|---|
| **Schema linking** | `main` | LLM linker pass → filtered DDL → SQL gen. Linking-recall meter via sqlglot. | +3–6pp (mostly on `exec_error`) |
| **Self-consistency / voting** | `feat/voting` | n=8 samples at T=0.6, execute each, group by canonical result-hash, return majority winner. | +2–4pp |
| **Self-correction** | `feat/correction` | First pass; for `exec_error` predictions, retry with the SQLite error in the prompt at T=0.2. | +4–6.5pp ceiling (capped at 150 = exec_err count) |
| **Plan-then-SQL CoT** | `feat/cot` | Two-stage: cheap plan call, then SQL conditioned on plan. Plan at T=0.3, SQL at T=0. | ~+3pp, concentrated on moderate |
| **Few-shot from train** | `feat/fewshot` | Retrieve k=4 BIRD-train examples by jaccard on question text, same-db preferred. BIRD's *disjoint* train/dev DBs mean shots are stylistic only, not schema-cheating. | +3pp (lower end of 3–7pp band) |

**Matrix:** 5 strategies × 3 models (Qwen2.5-Coder-32B-Instruct, Qwen3-Coder-30B-A3B-Instruct, Qwen3-32B-thinking) = 15 runs. Train download is in flight to unblock the 3 fewshot cells.

## Tooling that paid off

1. **Status taxonomy** in `bird/eval.py` — drives strategy prioritization with a single glance at the breakdown.
2. **Per-DB failure breakdown** in `inspect_failures.py` — caught that our first 50-question run was 100% `california_schools` (not representative). Avoided drawing the wrong conclusion.
3. **Worktree-isolation for parallel agent work** — 4 subagents implementing 4 strategies simultaneously, no merge conflicts during work, easy to compare deltas later.
4. **Modal volume layering** (BIRD data + HF model cache + results separately) — model weights pulled once, BIRD downloaded once, every subsequent run reuses both.
5. **Prompt-size audit** (`audit_prompt_sizes.py`) — formal verification, not vibes.
6. **Memory system** — saved the vllm/transformers pinning gotcha so it never costs us $$ again.

## Results (filled in as runs land)

### Baselines (full dev, n=1, greedy)
- Qwen2.5-Coder-7B-Instruct: **47.85%**
- Qwen2.5-Coder-32B-Instruct: **57.37%**
- Qwen3-Coder-30B-A3B-Instruct (MoE): **60.63%** ← current leader
- Qwen3-32B (dense, thinking-on, fair): _pending_
- Qwen3.6-27B: _pending_

### Strategy × Model matrix (5×3 = 15 cells)
_pending the in-flight runs_

### Combined best
_pending — once we know which strategies move the needle, layer the winners_

### Comparison to top public 32B-class results (Arctic-Text2SQL-R1-32B ≈ 73–74%)
_to be filled_

## What I'd do differently with more time

1. **RL on top of the strongest base + scaffolding.** Arctic-Text2SQL-R1's recipe (reasoning-first SFT → execution-grounded RL with verl/SkyRL) is the documented path past 70%.
2. **Use BIRD's `tables.json` column descriptions in prompts.** They're hand-written domain hints (e.g., "`Free Meal Count` represents…") — currently unused.
3. **Streaming partial eval** for early-stopping clearly broken runs. Would have caught the truncated Qwen3-32B (37.55%) within the first 100 prompts instead of after the full 1,534.
4. **Prompt explicit guidance on BIRD's idioms** — raw-count ratios over precomputed `Percent` columns; `CAST(... AS REAL)` for integer division. Recurring trap in our 7B failures.
5. **Per-question instrumentation for the simple→moderate gap.** Was the failure schema-linking, semantic interpretation, or SQL-translation? The status taxonomy helps but doesn't separate these.
6. **Reward-model selection over voting** — instead of majority on result-set, score candidates with a learned reward. Higher ceiling but needs SFT data.

## Open questions

- **MoE vs dense within the same generation:** Qwen3-Coder-MoE (60.63%) vs Qwen3-32B-fair (pending). If dense ≈ MoE, the 60.63% is just generation. If MoE > dense, expert specialization helps on 11-domain BIRD.
- **Thinking + voting compounding:** does thinking already extract the gain voting would, or are they orthogonal?
- **Correction's ceiling shrinks with model strength:** Qwen3-Coder-MoE has only 25 exec_errors → max +1.6pp from correction alone. Worth running anyway as a control.
- **Does linking help on stronger bases?** All BIRD prompts already fit comfortably; linking's "denoising" justification weakens when the base model already handles full schemas.

## Architecture trace (for the slide)

```
modal_app.py
├─ Inference (vLLM on B200) ────────→ chat() / complete()
├─ evaluate_predictions ────────────→ multiprocess CPU pool, status taxonomy
├─ download_bird ───────────────────→ one-shot, idempotent
├─ run_baseline ────────────────────→ greedy single-pass (the anchor)
├─ run_with_linking ────────────────→ linker pass → filtered DDL → SQL gen
├─ run_with_voting (feat/voting) ───→ n=8 samples + execute-vote (CPU pool)
├─ run_with_correction (feat/corr) ─→ first pass + exec_error retry
├─ run_with_cot (feat/cot) ─────────→ stage-1 plan + stage-2 SQL
└─ run_with_fewshot (feat/fewshot) ─→ jaccard-retrieve k=4 train shots

bird/
├─ data.py        BIRD JSON loading
├─ schema.py      SQLite introspection + DDL+samples rendering
├─ eval.py        execution-accuracy w/ timeout, status taxonomy, multiprocess
├─ prompts.py     baseline + raw-completion templates, SQL extraction
├─ inference.py   vLLM wrapper
├─ linking.py     Selection, parse_linker_output, lexical_link, ensure_keys, restrict_schema
├─ linking_eval.py  gold_columns via sqlglot, linking-recall metric
├─ voting.py      canonical result-hash, majority vote (feat/voting)
├─ correction.py  build_correction_messages (feat/correction)
├─ cot.py         plan + sql-from-plan messages (feat/cot)
└─ fewshot.py     TrainIndex, jaccard retrieve (feat/fewshot)

scripts/
├─ smoke_test.py            local pure-Python sanity (no GPU)
├─ inspect_failures.py      filter + side-by-side viewer
├─ compare_runs.py          multi-run table + pairwise wins + ensemble ceiling
└─ audit_prompt_sizes.py    tokenizer-grounded prompt-budget verification
```
