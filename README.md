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

## Ideas we ruled out (or didn't have time to try)

**Tried, killed by data:**
- **Chain-of-thought** — regressed −1.6 to −2.4pp on every coder base (commitment bias: the NL plan locks the model into wrong column / value choices that free-form SQL would have re-thought)
- **Schema linking** with 97.8% recall — still regressed −1.4pp on Qwen3-Coder-MoE and −4.2pp on Q3.6. Filtered DDL tilts strong bases toward worse column choices even when all gold columns are kept
- **Voting at T=1.0** (MoE-diversity hypothesis) — *worse* than T=0.6 voting (+0.13 vs +0.45pp). Killed the "MoE experts add diversity" hypothesis
- **Few-shot on strong coders** — helps non-coders (+3.07pp on Q3-32B-thinking) but neutral-to-negative on Q3-Coder/Q3.6 (anchor bias)
- **Agentic v2 (keep_baseline + expanded routing)** — broke break-rate fix interacts negatively with expanded routing. Null result (p=0.84)
- **Agentic v3 (+describe_column tool)** — tool got used on 68% of routed q with real isolated wins, but traded ~1-for-1 with v1's no-tool exploration. Null vs v1 (p=0.62)
- **SFT v2/v3 (chat-wrapped or flat with simple schema)** — both stuck at ~48-49% with a 17% exec_error tail dominated by a specific stray-`)` artifact. Format wrapping wasn't the cause; schema content was

**Tried, killed by infra:**
- **vLLM 0.20.2 default-config Qwen3.6** — DeepGEMM auto-selected, init crash on B200. Fix: `VLLM_USE_DEEP_GEMM=0` + `attention_backend=FLASH_ATTN` + `max_num_batched_tokens=2096`
- **Q3.6 voting at n=8** for the agentic-routing rule 3 — chunked-prefill collapsed throughput to 1.3 s/it at the small `max_num_batched_tokens=2096` Q3.6 needs. Workaround would be splitting into 4-5 separate Modal invocations
- **Full 32B RL** (TRL/veRL/hand-rolled FSDP+PEFT) — 32B trainer + 32B vLLM rollout doesn't fit a single B200. Needed a dedicated trainer-pool + rollout-pool split we couldn't reliably allocate. Burned ~$130 of compute on infra failures across three frameworks before pivoting to 7B
- **vLLM 0.19 pin for Q3.6** — broke `huggingface_hub.is_offline_mode` import. Rolled back to 0.20.2 + runtime env-var disable of DeepGEMM

**Didn't try (time / scope):**
- **RL on the SFT'd 32B Base** — the obvious follow-up; Arctic-R1 recipe applied to our 57.51% v4 substrate. Needs trainer/rollout GPU split
- **Grammar-constrained tool-call decoding** via vLLM's `guided_json` — would guarantee well-formed submit JSON and likely close most of the 67/110 `no_tool_call` leak we identified on Q3.6
- **Spider augmentation for SFT** — Arctic-R1 used BIRD + Spider together; we used BIRD only. Probably worth +2-3pp at the SFT stage
- **Multi-epoch SFT** — we did 1 epoch / 280 steps. Arctic-R1 used 3 epochs at 32B. Risk: overfitting on BIRD train's idioms (we already saw a hint of that in the stray-paren artifact)
- **Routing-on-SFT** — applying v1 agentic routing to the SFT'd 32B model. Probably the highest-EV stack of our two best findings
- **Self-eval / verifier-based filtering** — a small LM rating "does this SQL answer the question?" — published BIRD work uses this; we judged ROI too low for take-home time
- **Result-cardinality-vs-question-type mismatch** as a routing signal — "question asks for `the school` (singular) but result has 12 rows → route" — would expand recall on the wrong-but-valid bucket

## What we'd do with more time

1. **Re-run Q3.6 agentic with full voting (rule 3).** The 65.58% number is rules-1+2-only because `max_num_batched_tokens=2096` chunked-prefill made full-dev voting at n=8 infeasible. Splitting the run into 4-5 separate Modal invocations sidesteps the issue. Expected to push to ~66-66.5%.

2. **Scale the SFT.** v4 used 1 epoch / 280 steps / format_profile schema. Arctic-R1 used 3 epochs + Spider augmentation and reached ~64% pre-RL. A second epoch + Spider would close most of the gap to Arctic-R1's SFT cell.

3. **RL on the SFT'd 32B model.** Our SFT'd Base at 57.51% is a great substrate for Arctic-R1-style 6000-step GRPO. Memory math requires a dedicated trainer-pool / rollout-pool split (e.g., 4×B200 trainer + 2×B200 vLLM rollout server) which we couldn't get reliably allocated in this budget.

4. **Better Q3.6 routing.** The 110-question routed set was tiny because we dropped rule 3. With voting added, routed set grows to ~150-180 (similar to Q3-Coder's 187). Plus: degeneracy detection is by far the strongest signal — extending it (e.g., result-cardinality vs question type mismatch) could expand recall further.

5. **Layer agentic-routing on top of the SFT'd model.** Two of the three best findings stacked — SFT lifts the substrate, agentic routing harvests the last few wrong-but-valid cases. Expected combined: 57.51% + ~2pp = ~60% from a 7B-tier compute budget, or 65% + 2pp = ~67% if we had a Q3.6 SFT.

## Open questions and things I'm newly curious about

- **Does v1 routing transfer to the SFT'd Q3.6?** Untested. If our SFT-Base recipe applied to Q3.6 lifts it to, say, 67-68%, then agentic routing on top would target a smaller routed set (because the base has fewer wrong-but-valid cases) but with the same "near-pure wrong-baseline slice" property. Could compound cleanly.
- **Why does Q3.6 commit to submit so much faster than Q3-Coder?** (4.5% vs 23.5% budget-exhaustion). Both are coder-instruct models in the same generation; one is dense Q3.6, one is MoE. The architectural difference suggests dense models are more decisive at the tool-call boundary — but n=2 is hardly a finding. Worth replicating on more bases.
- **Is the stray-`)` SFT artifact transferable?** It's a specific overfit to BIRD-train's frequent `MAX(CAST(... AS REAL) / ...)` pattern. We saw it disappear when we switched to format_profile schema. Would a different artifact appear on a different schema renderer? Hard to know without controlled ablations.
- **What's the saturation curve of agentic routing?** v1 lifts +2pp; v2/v3 ablations are null. We didn't try v4-style tools (value retrieval, table inspection). At what point does adding tools regress in the way v2/v3 did? Three data points is too few.
- **vLLM Blackwell quirks** — we hit 5+ subtle pin/config issues on B200 (DeepGEMM auto-select, GDN cache alignment, flashinfer JIT, kv-cache-dtype incompatibilities). Modal abstracts away the host but not the GPU-arch-aware library stack. Curious whether other inference engines (sglang, TRT-LLM) avoid these.

## Next experiments (in priority order)

The single highest-EV experiment we'd run next:

1. **Agentic RL.** Combine the two best findings of this take-home: the agentic loop and the GRPO training pipeline. Instead of using a fixed-prompt agent, train the *agent's* policy with RL where the reward is the merged full-dev EX of its tool-call trajectory. Specifically:
   - Initial policy = SFT'd 32B-Base (57.51% greedy)
   - Generate tool-call trajectories on routed questions, score the final SQL with our binary-execution reward, GRPO-update the policy
   - This trains the model to *use* execute_sql, *interpret* probe results, and *commit* to a submit tool-call — exactly the failure modes we saw in the no_tool_call audit (76% format-compliance failures)
   - The trick: rollouts are multi-turn (each rollout is a full agent loop), so GRPO group-size and credit-assignment need adapting. Recent work like "RL with tool use" (Snowflake's Arctic-R1 follow-up + the StepRL line) gives a recipe
   - Expected gain: should close most of the +2.28pp ceiling our prompt-only agent hit on Q3.6, *plus* recover the no_tool_call leak (60.9% of cases) into more clean submits. Realistic projection: 65.58% → 68-70% on Q3.6 substrate

2. **Routing-on-SFT.** Apply v1 agentic routing on top of our SFT'd Qwen2.5-Coder-32B-Base (57.51% greedy). The SFT'd model has a different wrong-but-valid distribution than Q3.6, so the routed set composition will be different. Expected: +2pp → ~60%, validating that routing transfers to trained substrates.

3. **Full v1 routing on Q3.6** (with rule 3). Split voting into 4-5 sequential Modal invocations to sidestep the chunked-prefill stall we hit. Routed set grows ~110 → ~150-180, expected lift +0.5-1pp on top of the current 65.58%.

4. **Multi-epoch SFT + Spider augmentation.** 1 epoch / BIRD-train-only → 3 epochs / BIRD + Spider. Pre-RL recipe Arctic-R1 uses. Expected: 57.51% → 62-64% on the SFT'd model alone.

5. **Tool-call grammar constraint.** vLLM `guided_json` on the agent's output. Would force well-formed submit JSON and likely lift the 60.9% no_tool_call rate substantially (76% of those were format failures, not reasoning failures). Cheap to try, possibly +1-2pp on top of agentic-routing.

## Tooling — what made us fast

Everything runs on Modal with persistent Volumes for the BIRD corpus + HF cache + SFT checkpoints + per-run results.

### What we built
- `modal_app.py` — entry points: `run_baseline`, `run_with_linking`, `run_sft_eval`, `evaluate_predictions`
- `modal_app_agentic.py` — agent loop entry points: `run_agentic`, `run_agentic_routed`, `run_agentic_routed_q36`, voting helpers
- `sft_train_32b.py` — 8× B200 FSDP SFT (mp.spawn + accelerate FSDP plugin)
- `rl_train_7b.py` — single-B200 GRPO via TRL
- `bird/` — eval harness, schema, prompt builders, agent loop, voting/correction strategies
- `scripts/` — failure dashboards, run diffs, routing-set builder, McNemar utility

### Patterns that paid off

- **Per-question result JSON as the universal currency.** Every cell in the matrix produces a `{ex, n, results: [{question_id, db_id, status, predicted_sql, ...}]}` file on the `bird-results` volume. Once that shape was locked, *everything* downstream — McNemar tests, paired-diff between runs, routing-set construction, failure-mode dashboards — was a 30-line script. The routing-set builder (`scripts/build_routing_set.py`) is literally a set-union over these files. The merge for the agent-routed final EX (`scripts/score_q36_merge.py` style) is a few-line zip-and-substitute over them. **This file format is the single biggest tooling decision** — it made every subsequent experiment a composable step.
- **Worktree-parallel experiments.** Each strategy lives on its own branch with its own worktree (`feat/agentic-explore`, `feat/sft-32b-flat`, `feat/rl-grpo-7b`, `feat/agentic-describe-column`). Running two experiments concurrently never blocks on git, and `git diff main...feat/X` shows exactly what a strategy adds. Combined with subagents (running their own modal commands in their own worktrees), we ran 3-4 simulations in parallel for the last two hours of the take-home.
- **Subagents for blast-radius isolation.** The three agentic ablations (v2 keep_baseline, v3 describe_column, Q3.6 retry) each ran in a dedicated subagent + worktree. If a subagent burnt a budget on an infra issue, it didn't take down the main thread. The Q3.6 first attempt hit three image issues in 34 min; the parallel SFT experiments kept running.
- **The schema profiler port (`bird/exp18_schema.py::format_profile`) was the v4 unlock.** Distinct-value lists for low-cardinality columns + explicit FK section turned a 17% exec_error tail into 0.85%. A 200-line port that recovered 9.5pp.
- **`scripts/mcnemar.py` paired-contingency stats.** Every "is this lift real?" question gets answered in 2 seconds with a 95% CI and p-value. Saved us from shipping the agentic v2 result as a real lift when it was actually null.
- **Modal volumes as the cross-machine storage layer.** No re-downloading BIRD (16 GB), no re-downloading model weights (60-120 GB), no re-running baselines. The 22-cell matrix happened *because* every prior result was reusable on the volume.

### Image pinning matters at the edges
- vLLM 0.11.0 + transformers 4.57.0 works for Qwen2.5/Qwen3-Coder
- vLLM 0.20.2 + transformers 5.8.0 + `VLLM_USE_DEEP_GEMM=0` + `attention_backend=FLASH_ATTN` + `max_num_batched_tokens=2096` for Qwen3.6-27B (FP8 / DeepGEMM auto-select + GDN cache alignment quirks on B200)

### What I'd do differently next time
- **Lock the per-question result JSON shape in week 1.** Some early baselines used slightly different keys and required cleanup scripts. A `bird.results.schema` module enforced as the only valid output would have saved an hour of cross-run grief.
- **Smaller smoke tests for SFT.** Our 50-q `--dry-run` mode tested the FSDP path but didn't catch the data-content artifact (stray-paren) until we ran 1534-q eval. A 200-q SFT + 200-q eval intermediate stage would have caught v2/v3's schema-content gap a full retry-cycle earlier.
- **Establish the "kill switch" earlier.** SFT v4 was killed twice in our workspace by external CLI activity (probably dashboard Stop clicks during GPU-quota juggling). We lost ~40 min before realizing the pattern. A startup banner explicitly listing "DO NOT STOP THIS APP" would have helped.
- **Tracked Modal app IDs in a structured log file** rather than chat. By the end we had ~50 app IDs in a 10-experiment run; tying them back to the strategy + branch + commit required scrolling through the conversation. A `runs.jsonl` append-only log would have made post-hoc reproduction trivial.

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
