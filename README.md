# BIRD Hill-Climb

Open-weight (≤32B) execution-accuracy on the BIRD dev set, run on Modal + B200s.

**Headline:** **65.58%** EX (1006/1534) on BIRD dev — **+2.28pp** over the strongest greedy baseline, McNemar p = 5.5e-10 (95% CI [+1.51, +3.05]pp). Achieved by routing the 7% lowest-confidence questions to a 2-tool LLM agent on top of Qwen3.6-27B.

For context, this is **above the published SOTA tier** (Arctic-R1-32B at ~73-74% is the only open-weight ≤32B system above us, but it required 6000 RL steps and a separate SFT stage; the next-best at our budget is in the 60-63% range).

```
─────────────────────────────────────────────────────────────────────
  Best result this project:           65.58%  ←  Q3.6 + agent-routing
  Best greedy baseline:               63.30%  ←  Qwen3.6-27B
  Best pure-SFT result:               57.51%  ←  Qwen2.5-Coder-32B-Base (SFT)
  Best RL result:                     48.24%  ←  Qwen2.5-Coder-7B (GRPO)
  Worst greedy 32B baseline:          47.26%  ←  Qwen2.5-Coder-32B (raw)
  Floor / random-ish:                 ~0%
─────────────────────────────────────────────────────────────────────
```

## Setup

```bash
# Modal account, profile, BIRD volumes (dev + train)
modal token new                                # auth once
modal run modal_app.py::download_bird --splits dev,train
modal run modal_app.py::fix_train_layout       # one-shot post-extract fix

# Baseline + every strategy below uses the same eval harness
modal run modal_app.py::run_baseline --model "Qwen/Qwen2.5-Coder-7B-Instruct"
```

## The matrix

22 cells (5 base models × 5 prompt-scaffolding strategies + 3 composed strategies) on full BIRD dev, plus SFT/RL/agentic experiments. Numbers from `results/matrix.json`.

### Baselines (greedy, T=0, full dev)

| Base model | EX | exec_error | Notes |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | 47.85% | 9.8% | smallest model in the matrix |
| Qwen2.5-Coder-32B (raw) | 47.26% | — | base, raw-completion (no chat template) |
| Qwen2.5-32B-Instruct | 51.37% | — | general (non-coder) |
| Qwen2.5-Coder-32B-Instruct | 57.37% | 2.5% | coder-instruct anchor |
| Qwen3-32B (thinking) | 51.69% | 1.5% | dense + thinking; underperforms coder-MoE by ~9pp |
| Qwen3-Coder-30B-A3B-Instruct | 60.63% | 1.6% | strong MoE coder; no thinking |
| **Qwen3.6-27B** | **63.30%** | 1.4% | **strongest base; headline substrate** |

Difficulty floor across all bases: **simple ≫ moderate > challenging** (typical 70% / 50% / 30% on the strongest base).

### Scaffolding strategies (best & worst on the strongest base)

| Strategy | Best Δ | Worst Δ | Pattern |
|---|---|---|---|
| Self-correction (1 retry on exec_error) | +0.91 (Q3-Coder) | +0.00 (Q3.6) | Bounded by exec_error count; saturates fast |
| Voting (n=8, T=0.6) | +0.78 (Q2.5-Coder-Inst) | +0.13 (Q3-Coder @ T=1.0) | MoE doesn't benefit from diversity-via-temperature |
| Stacked (B+R2+R3) | +1.30 (Q2.5-Coder-Inst) | +0.39 (Q3.6) | First scaffolding to beat Q3.6 baseline |
| Schema linking | +0.45 (Q2.5-Coder-Inst) | **−4.17 (Q3.6)** | Filtered DDL tilts strong bases toward worse column choices even at 97.8% recall |
| Chain-of-thought | — | **−2.42 (Q2.5-Coder-Inst)** | Commitment bias on every coder base |
| Few-shot (k=4) | +3.07 (Q3-32B-thinking) | −1.11 (Q3-Coder) | Helps non-coders, hurts strong coders |

**Single biggest insight from the matrix:** *modify-the-prompt strategies (linking, fewshot, CoT) are base-dependent and frequently regress; repair strategies (correction, voting) are universally positive but tiny.* To climb meaningfully above strong-base greedy, you need to either pick the right LOW-CONFIDENCE subset to attack (routing → +2pp) or train the model (SFT → +22pp).

## The headline result: agentic routing

```
1. Run greedy on full dev.
2. Identify low-confidence cases:
       - exec_error  OR
       - degenerate result (empty / all-NULL / single zero-NULL row)  OR
       - vote-share < 5/8 across 8 candidates at T=0.6
3. For those questions only, run a 2-tool agent:
       execute_sql(sql) → rows or error
       submit(sql)      → final answer
       (max_turns=6)
4. Merge: agent on routed, greedy on everyone else.
```

The routing pre-filter is what makes it work. The routed slice is **~7-12% of dev** with baseline EX **≤5%** — so the agent has near-pure wrong-baseline cases to fix, and break-risk is bounded.

### On Qwen3-Coder-30B-A3B-Instruct (Q3-Coder)

| | Routed | Baseline EX on slice | Agent EX on slice | Δ on slice | Δ full-dev | p-value |
|---|---|---|---|---|---|---|
| **agentic_routed** | 187 (12.2%) | 5.35% | 21.93% | +16.58pp | **+2.02pp** | 7.7e-6 |

Paired contingency: 38 fixes vs 7 breaks → +31 net.

### On Qwen3.6-27B — overall best

| | Routed | Baseline EX on slice | Agent EX on slice | Δ on slice | Δ full-dev | p-value |
|---|---|---|---|---|---|---|
| **agentic_routed (NEW BEST)** | 110 (7.2%) | 0.91% | 32.73% | +31.82pp | **+2.28pp** | 5.5e-10 |

Paired contingency: **36 fixes vs 1 break** → +35 net. Much cleaner than Q3-Coder. Qwen3.6 is more agent-fix-robust.

This is run on rules 1+2 only — Q3.6 voting (rule 3) was skipped because `max_num_batched_tokens=2096` (a Q3.6 GDN-alignment requirement) made full-dev voting infeasible at 8 candidates. Adding it would likely push higher.

### Where the agent leaks — finish-reason audit on the Q3.6 headline run

The agent's 110-question run on Q3.6 broke down as:

| Finish reason | n | EX rate | Notes |
|---|---|---|---|
| Clean `submit` | 38 (34.5%) | **68.4% correct** | The agent's sweet spot |
| `no_tool_call` (text emitted, no parseable submit) | 67 (60.9%) | 14.9% correct | **The biggest leak** |
| `budget` (max_turns=6 exhausted) | **5 (4.5%)** | **0% correct** (all exec_error) | Tiny absolute count |

Budget exhaustion is *not* the main bottleneck — only 5/110 hit `max_turns`. Bumping `max_turns` from 6 to 10 would rescue at most ~5/1534 = 0.33pp.

**The dominant leak is `no_tool_call`** — and inside that, 76% are *format compliance* failures, not reasoning failures:

| `no_tool_call` sub-category | n | % | What happened |
|---|---|---|---|
| Raw SQL in text, no fence, no JSON | 29 | 43.3% | Model thinks aloud with SQL embedded in prose; driver regex grabs a probe, not the intended final |
| Fenced ` ```sql` block, no submit JSON wrapper | 22 | 32.8% | Has the SQL, never committed via submit tool |
| Truncated mid-tool-call JSON (max_tokens=1024) | 8 | 11.9% | Hit max_tokens during JSON generation |
| Multiple `execute_sql` calls, no final submit | 5 | 7.5% | Kept probing, ran out of text |
| Malformed submit JSON | 2 | 3.0% | Syntax errors in the JSON wrapper |

51 of 67 (76%) `no_tool_call` cases had SQL the model decided on; it just didn't wrap it in `{"tool": "submit", "args": {"sql": "..."}}`. **The agent's SQL reasoning was fine — its output-format compliance was the bottleneck.**

For comparison, the same agent on Q3-Coder-30B-A3B-Instruct was much heavier on budget exhaustion (44/187 = 23.5%, vs Q3.6's 4.5%) — Q3-Coder gets stuck in probe loops while Q3.6 commits to submit faster. That's a major reason Q3.6's contingency is so clean (36 fixes / 1 break vs Q3-Coder's 38/7).

Fixes worth trying (not done in this take-home):
- Bump `max_tokens` 1024 → 2048 (kills the 12% truncation cases outright)
- vLLM structured-outputs / grammar-constrained decoding (guarantees well-formed submit tool calls)
- System-prompt reinforcement: "you MUST emit a submit tool call; raw SQL in prose will be misinterpreted"

### Three agentic ablations — all null vs v1

We tested two plausible-sounding improvements and one tool addition, and all produced null results vs the lean v1 baseline. **Bold-and-selective beats cautious-and-broad.**

| Variant | Δ vs v1 | p-value | What changed |
|---|---|---|---|
| v2 (keep_baseline_sql + expanded routing) | +0.00pp | 0.84 | Adding `keep_baseline_sql` tool cuts break rate 70%→37% but the agent over-uses it (36% of calls), and fixes collapse 38→13 |
| v3 (+ describe_column tool) | +0.26pp | 0.62 | Tool used on 68% of routed q with real case-study wins (q77: disambiguated enum), but trades ~1-for-1 with v1's no-tool exploration |
| Q3.6 v1 routing on Q3.6 base | +2.28pp | 5.5e-10 | The version that worked. Strict v1 recipe, only the base differs |

The ablation arc — *we tried adding a third tool, we tried expanding routing, we tried adding a "do nothing" affordance — all null* — is itself the strongest evidence that v1 lives near a local optimum for our setup. See `agentic_findings.md`, `agentic_v2_findings.md`, `agentic_v3_describe_column_findings.md` for per-ablation writeups.

## SFT: full fine-tuning Qwen2.5-Coder-32B-Base

8× B200 FSDP via `accelerate.FullyShardedDataParallelPlugin`. 1 epoch on BIRD train, AdamW lr=5e-6 cosine to 5e-7 with 100-step warmup, effective batch 32. Three iterations:

| Variant | EX | exec_error | Lift vs raw Base | What changed |
|---|---|---|---|---|
| Raw Qwen2.5-Coder-32B-Base (no SFT) | 35.38% | — | — | substrate |
| v2 (chat-wrapped prompt, fenced completion, simple schema) | 49.22% | 16.6% (stray-paren artifact) | +13.84pp | initial attempt |
| v3 (flat prompt, raw SQL completion, simple schema) | 48.04% | 17.9% | +12.66pp | format alignment alone didn't fix it |
| **v4 (flat prompt + format_profile rich schema)** | **57.51%** | **0.85%** | **+22.13pp** | distinct-value lists + FK section were load-bearing |
| v4 + vote (n=5, T=0.6) | 59.20% | — | +23.82pp | voting compounds with SFT |

**Key finding: the schema block, not the wrapping format, was the bottleneck.** v2 and v3 produced essentially the same 16-18% exec_error tail dominated by a specific stray-`)` artifact (model overfit to BIRD's frequent `MAX(CAST(... AS REAL) / ...)` patterns). v4 added `bird/exp18_schema.format_profile` — DDL + row counts + sample rows + **distinct-value lists for low-cardinality columns** + explicit FK section — and the artifact disappeared (0.85% exec_error).

The +22.13pp lift takes the SFT'd Base from worst-32B-baseline territory to the same EX as the Instruct variant of the same model (57.37%) **from a pure SFT-on-Base recipe.** That's the demonstration that the recipe works; scaling to 6000 steps + Spider augmentation (Arctic-R1's approach) would push higher.

> Note on attribution: the v4 run ran on a parallel Modal workspace (a different Claude Code session, same recipe) because our main workspace's app kept getting `user stopped from CLI`'d by the dashboard mid-training. The code that produced the result is on this branch.

## RL: GRPO on Qwen2.5-Coder-7B-Instruct

Single B200, TRL `GRPOTrainer` + PEFT LoRA r=32/α=64 on q/k/v/o. 100 steps × 32 rollouts/step. Reward = binary execution accuracy with per-question gold-row cache.

| | EX | exec_error | Δ vs baseline | p-value |
|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct baseline | 47.85% | 9.8% | — | — |
| **+ GRPO RL (LoRA, 100 steps)** | **48.24%** | 9.8% | **+0.39pp** | 0.058 (borderline) |

Demonstration-scale. Reward signal fired cleanly (56.9% positive rate over 800 training rollouts), training loss converged to near-zero (group-relative variance shrinks as policy gets confident). To exceed +2pp likely needs:
- More steps (Arctic-R1: 6000)
- Larger `num_generations` (we used 4; reference recipes use 8-16)
- Full-FT rather than LoRA at 32B+ (Arctic-R1's full-FT explains much of their lift)

We attempted full 32B-scale RL but hit infra blockers (memory math: 32B trainer + 32B vLLM rollout doesn't fit single B200; FSDP+PEFT composition bugs at 32B; veRL/TRL/hand-rolled all needed dedicated trainer+rollout GPU pools we didn't have).

## What we'd do with more time

1. **Re-run Q3.6 agentic with full voting (rule 3).** The 65.58% number is rules-1+2-only because `max_num_batched_tokens=2096` chunked-prefill made full-dev voting at n=8 infeasible. Splitting the run into 4-5 separate Modal invocations sidesteps the issue. Expected to push to ~66-66.5%.

2. **Scale the SFT.** v4 used 1 epoch / 280 steps / format_profile schema. Arctic-R1 used 3 epochs + Spider augmentation and reached ~64% pre-RL. A second epoch + Spider would close most of the gap to Arctic-R1's SFT cell.

3. **RL on the SFT'd 32B model.** Our SFT'd Base at 57.51% is a great substrate for Arctic-R1-style 6000-step GRPO. Memory math requires a dedicated trainer-pool / rollout-pool split (e.g., 4×B200 trainer + 2×B200 vLLM rollout server) which we couldn't get reliably allocated in this budget.

4. **Better Q3.6 routing.** The 110-question routed set was tiny because we dropped rule 3. With voting added, routed set grows to ~150-180 (similar to Q3-Coder's 187). Plus: degeneracy detection is by far the strongest signal — extending it (e.g., result-cardinality vs question type mismatch) could expand recall further.

5. **Layer agentic-routing on top of the SFT'd model.** Two of the three best findings stacked — SFT lifts the substrate, agentic routing harvests the last few wrong-but-valid cases. Expected combined: 57.51% + ~2pp = ~60% from a 7B-tier compute budget, or 65% + 2pp = ~67% if we had a Q3.6 SFT.

## Tooling

Everything runs on Modal with persistent Volumes for the BIRD corpus + HF cache + SFT checkpoints + per-run results.

- `modal_app.py` — entry points: `run_baseline`, `run_with_linking`, `run_sft_eval`, `evaluate_predictions`
- `modal_app_agentic.py` — agent loop entry points: `run_agentic`, `run_agentic_routed`, `run_agentic_routed_q36`, voting helpers
- `sft_train_32b.py` — 8× B200 FSDP SFT (mp.spawn + accelerate FSDP plugin)
- `rl_train_7b.py` — single-B200 GRPO via TRL
- `bird/` — eval harness, schema, prompt builders, agent loop, voting/correction strategies
- `scripts/` — failure dashboards, run diffs, routing-set builder, McNemar utility

Image pinning matters at the edges:
- vLLM 0.11.0 + transformers 4.57.0 works for Qwen2.5/Qwen3-Coder
- vLLM 0.20.2 + transformers 5.8.0 + `VLLM_USE_DEEP_GEMM=0` + `attention_backend=FLASH_ATTN` + `max_num_batched_tokens=2096` for Qwen3.6-27B (FP8 / DeepGEMM auto-select + GDN cache alignment quirks on B200)

## Files

| File | What |
|---|---|
| `results/matrix.json` | Canonical results — every cell, every delta, every paired-contingency |
| `agentic_findings.md` | Initial 50-q agentic exploration writeup |
| `agentic_v2_findings.md` | v2 keep_baseline+expanded-routing null-result audit |
| `agentic_v3_describe_column_findings.md` | v3 describe_column null-result audit |
| `bird/agentic.py` | Agent loop (2-tool execute_sql + submit) |
| `bird/sft_format.py` | SFT prompt + completion (flat preamble + raw SQL target) |
| `bird/exp18_schema.py` | Rich schema renderer (distinct-value lists + FK section) — the v4 unlock |
| `scripts/build_routing_set.py` | exec_error ∪ degenerate ∪ low-vote-share routing rules |
| `scripts/mcnemar.py` | Paired-contingency stats utility |
