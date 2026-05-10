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
| Qwen3-32B (dense, thinking-on, max_tokens=8192 — *fair*) | 3 | dense, general | 51.69% | 23 |
| Qwen3.6-27B | 3.6 | dense, general | _agent debugging vllm pin_ | — |

**Insights from the sweep:**
- **Generation gap dominates** at 32B-class: Qwen3 → Qwen2.5 closes ~3.3pp on the same coder-instruct base.
- **Coder-tuning matters more than thinking on SQL**: at the Qwen3 generation, Coder-MoE-no-thinking (60.63%) beats dense-general-with-thinking (51.69%) by **9pp**. For SQL specifically, dedicated coder post-training is worth more than chain-of-thought reasoning.
- **Better models concentrate failures away from exec_error**: 150 → 38 → 25 → 23 going from 7B-Instruct → 32B-Coder-Instruct → 30B-Coder-MoE → Qwen3-32B-thinking. Stronger / newer / thinking-mode models all write more syntactically-valid SQL out of the box, which **shrinks the ceiling for self-correction strategies**.
- **Thinking on Qwen3 needs ≥8192 max_tokens**: the 1024 default cuts off SQL after the reasoning trace, dropping EX from 51.69% (fair) to 37.55% (broken). Documented as a configuration trap.

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

| | Qwen2.5-Coder-32B-Inst (57.37%) | Qwen3-Coder-30B-A3B-Inst (60.63%) | Qwen3-32B fair (51.69%) |
|---|---:|---:|---:|
| linking (post backtick-fix) | **57.82%** (+0.45) | **59.26%** (-1.37) ⬇ | **50.46%** (-1.23) ⬇ |
| voting (n=8) | **58.15%** (+0.78) | **61.08%** (+0.45) | TIMED OUT (Modal 3600s cap; n=8 thinking too slow) |
| correction | **58.15%** (+0.78) | **61.54%** (+0.91) | **52.09%** (+0.40) |
| CoT | **54.95%** (-2.42) ⬇ | **58.74%** (-1.89) ⬇ | **50.07%** (-1.62) ⬇ |
| fewshot (k=4) | **57.89%** (+0.52) | **59.52%** (-1.11) ⬇ | **54.76%** (+3.07) ⬆⬆ |

**Insights from the cells we have so far:**

- **Correction is a sanity-check strategy, not a hill-climbing lever.** On 32B-Coder-Instruct (38 exec_errors): retry rescued 12, +0.78pp. Ceiling shrinks fast as base models get cleaner — Qwen3-Coder-MoE has only 25 exec_errors → max +1.6pp possible.
- **Voting trims exec_errors but doesn't unlock semantic fixes.** +0.78pp on Qwen2.5-Coder-32B (exec_err 38 → 7), +0.45pp on Qwen3-Coder-MoE (exec_err 25 → 13). Same mechanism as correction: result-set vote naturally filters broken candidates. Above the exec_error floor, n=8 sampling adds noise that mostly cancels out on confident greedy bases.
- **CoT regresses on coder-tuned models.** Qwen2.5-Coder-32B: -2.42pp; Qwen3-Coder-MoE: -1.89pp. Forcing a natural-language plan stage hurts models post-trained to go directly to SQL — the plan is brittle (coder models don't reason in NL as cleanly as in code), and the SQL stage second-guesses itself. exec_error went *up* from 38 → 49 on Qwen2.5-Coder, confirming added confusion. **Implication: CoT is the wrong tool for coder-instruct bases.**
- **Schema linking initially regressed (-0.9pp / -3.2pp / -1.1pp) — but the cause was a bug in our pipeline, not the strategy itself.** Linker recall was excellent (97.8%); the SQL generator just couldn't use the filtered schema. **Failure-mode dive (compare with `compare_runs.py` + a transition-matrix analysis on q3cm)** revealed that 75 of the 101 "broke_it" cases had perfect linker recall — the linker was *not* dropping needed columns. Looking at predicted SQL on those cases: the model was inventing column names like `FreeMealCountK12` instead of `` `Free Meal Count (K-12)` ``. The bug was in `restrict_schema → _column_def`: it emitted column names raw without backticks, so multi-word columns produced invalid DDL. The model "fixed" the malformed DDL by inventing identifier names that don't exist → exec_error. **One-line fix (always backtick-quote identifiers in restricted DDL); re-running the 3 linking cells.** This is the highest-leverage finding of the matrix run: *linker recall ≠ linking helping EX*, the prompt-rendering layer matters as much as the column-selection logic.
- **The strongest pattern in the matrix: which strategies help depends on the base.**
  - On **Qwen2.5-Coder-32B-Instruct** (57.37%), 4 of 5 strategies help: linking +0.45, voting +0.78, correction +0.78, fewshot +0.52, CoT –2.42. (Older-gen coder benefits broadly from scaffolding.)
  - On **Qwen3-Coder-MoE** (60.63%, the leader), only 2 of 5 help: voting +0.45, correction +0.91; everything else regresses. Strongest base → strategies leave little on the table.
  - On **Qwen3-32B (dense, thinking, general)** (51.69%), correction +0.40 and **fewshot +3.07** ← biggest single-strategy lift in the matrix. The general (non-coder) thinking model has a domain gap that few-shot examples actually fill, while the coder bases already have those patterns baked in.
- **Punchline:** "what helps" is not a property of the strategy; it's a property of the base × strategy interaction. Repair strategies (correction, voting) are the most universally positive. Modify-the-prompt strategies (linking, CoT, fewshot) are *specific*: linking and fewshot help weaker / non-coder bases; CoT hurts every coder base we tried.
- **Implication:** as we get to stronger coder bases, the way to climb further is *training* (RL on execution rewards) or a *better base*, not better prompts.

### Combined best
_pending — once we know which strategies move the needle, layer the winners_

### Comparison to top public 32B-class results (Arctic-Text2SQL-R1-32B ≈ 73–74%)
_to be filled_

## Composition: voting + within-vote correction (Qwen3-Coder-MoE)

Combined the two repair strategies into a single 3-pass flow (sample n=8 at T=0.7 → execute all → batched retry on exec_error candidates at T=0.0 → vote on the upgraded pool with `prefer_nondegenerate` set to drop empty/all-NULL results when non-degenerate is the majority).

Result: **61.15% (+0.52pp)** — *between* voting alone (+0.45) and correction alone (+0.91), not beyond either. **The composition did not compound.**

| variant | EX | Δ vs baseline | exec_error |
|---|---:|---:|---:|
| baseline | 60.63% | — | 25 |
| voting alone (n=8) | 61.08% | +0.45 | 13 |
| **correction alone** | **61.54%** | **+0.91** | 8 |
| voting + correction | 61.15% | +0.52 | **1** |

Why the failure to compound:
1. **Voting and correction target the same failure mode (exec_error rescue) via different mechanisms.** Voting fixes "1 of 8 samples gets it right" via diversity; correction fixes "explicit retry with the error in context." Different mechanisms, mostly overlapping coverage.
2. **The vote step can pick a *wrong* non-error answer that correction alone would never produce.** Correction alone is anchored on greedy SQL (T=0); voting's pool at T=0.7 has more variance, and the result-hash majority sometimes converges on a confidently-wrong query.
3. **Exec_error went 25 → 1**: the within-vote retry mechanically *works* — almost all errors got fixed. But fixing them just shifts the population from "fail" to "wrong", which doesn't help EX.

**Lesson: composing strategies with overlapping mechanisms doesn't add up.** For meaningful gains, compose strategies that address *orthogonal* failure modes — e.g., correction (exec_error) + a verifier-style filter (semantic-wrong detection) + a column-disambiguation pass.

### Per-question diff: which questions did each strategy win or lose?

| | combined → wrong | combined → correct | net |
|---|---:|---:|---:|
| simple (corr 590 ✓) | 7 broken | 8 fixed | **+1** |
| moderate (corr 280 ✓) | 6 broken | 0 fixed | **−6** |
| challenging (corr 74 ✓) | 2 broken | 1 fixed | **−1** |
| **total** | **15** | **9** | **−6** |

Every transition was a *different SQL* — no eval flakes, both pipelines are deterministic. The mechanism for each direction:

- **Voting BREAKS moderates because diversity introduces wrong-column variance.** Sample broken cases: greedy-correction filtered by `District = 'Fresno County Office of Education'`; combined filtered by `StatusType = 'Active'` (wrong-but-confident vote). Greedy used `d.A2 = 'Hl.m. Praha'` (correct Czech name); combined used `d.A2 = 'Prague'` (English; wrong-but-popular vote winner). Greedy returned `c.client_id`; combined returned `c.gender` — completely wrong column. **In all six moderate losses, the correct SQL existed in the candidate pool — voting just picked a different one.**

- **Voting FIXES some simples by averaging out greedy's brittle quirks.** Cases where correction's first pass had a clearly broken JOIN (`a.client_id = d.client_id` on a chain that doesn't connect that way) or returned more columns than gold expected (3 cols when gold wants 1, or `CASE WHEN ... THEN 'Yes' ELSE 'No'` when gold wants raw `label`). On simples, the diversity pool surfaces a cleaner answer.

**The asymmetry is the lesson:** voting helps when the greedy answer has a stylistic quirk (over-projection, redundant CASE, wrong literal language); voting hurts when the moderate question has a *single* sharp interpretation that greedy gets but diversity disrupts. Combined isn't strictly worse — it's a different bias-variance tradeoff. **For Qwen3-Coder-MoE on BIRD, lower variance (correction alone) is the better operating point.**

## Audit: are the regressions real, or pipeline bugs?

Ran a cell-by-cell audit on all 12 finalized strategy×model results. **All cells structurally clean** (1534 predictions each, status counts re-sum, no empty SQL on non-empty status, no truncation on non-thinking cells). One bug was caught and fixed mid-matrix (linking's `_column_def` wasn't backtick-quoting BIRD's spaces-in-names columns; cost ~1.4pp on Qwen3-Coder-MoE alone). After the fix, the *residual* regressions on `linking`, `CoT`, and `fewshot` are genuine model failures, not pipeline issues:

- **CoT regression dive (Qwen2.5-Coder-32B-Instruct, sample of 124 broke_it cases):** plans are 387–985 chars of well-structured natural language; raw_completion fences are clean; predicted SQL faithfully implements the plan. Failures are **commitment bias** — the plan stage locks the model into a specific column/value choice (e.g., `NCESDist` instead of `NCESSchool`; `'Directly Funded'` instead of the lower-case `'Directly funded'` that's actually in the data; `County` instead of `City`), and the SQL stage no longer second-guesses. Coder models that already do this disambiguation implicitly during free-form SQL writing get worse when forced to commit upfront in NL.
- **Linking regression dive (Qwen3-Coder-MoE, 77 broke_it cases post backtick fix):** splits into two camps. Recall<1.0 cases (linker dropped a needed column) directly cap the answer. Recall=1.0 cases — the linker kept everything but the filtered DDL caused the model to *prefer* a different column (e.g., the precomputed `Percent (%) Eligible Free (K-12)` over the raw-count ratio that BIRD's gold actually uses). The full schema gives the model *both* options; the filtered version still has both but the visual emphasis shifts subtly. Net effect: marginal.
- **Fewshot regression dive (Qwen3-Coder-MoE, 90 broke_it cases):** classic **anchor bias** — when train shots happen to use percentage outputs (`* 100`), aggregations (`AVG(...)`), or a particular JOIN style, the model copies those patterns into the current question even when inappropriate. Strong base models already know the right pattern; the shots add noise.

**Verdict: no further bugs found. The deltas in the matrix are real signal.**

## Predictions for the Qwen3.6 strategy matrix (recorded *before* runs land)

The Qwen3.6-27B baseline is **63.30%**. Strategy predictions, anchored on the Qwen3-32B-thinking row (closest analog: dense, thinking, general) and adjusted for Qwen3.6's higher base quality:

| strategy | predicted Δ | reasoning |
|---|---|---|
| linking | −1.0 to −2.0pp | Modify-prompt strategies regress on confident bases. Qwen3.6 is even more confident than Qwen3-32B-thinking (-1.23 there) → expect similar or slightly worse. |
| voting (n=8) | 0 to +0.5pp | Thinking already explores variants internally; n=8 sampling has overlapping mechanism. Bounded by 18 exec_errors. |
| correction | +0.3 to +0.6pp | Universal small lift. 18 exec_errors → max ~+1.2pp ceiling; realistic 30-50% rescue rate. |
| CoT | −1.5 to −2.5pp | Explicit plan stage doubly redundant on a thinking model + commitment-bias trap. Worst regression of the row. |
| fewshot (k=4) | +1.0 to +2.5pp | Biggest lift on Qwen3-32B-thinking (+3.07pp); Qwen3.6 has stronger SQL priors so less room for shots to teach. **Predicted strongest single strategy.** |

**Combined-best prediction:** fewshot + correction (orthogonal mechanisms) ≈ +2.0pp → ~65.3% EX.

### Actual vs predicted (4/5 cells; voting killed for cost)

| strategy | predicted | actual | hit/miss |
|---|---|---:|---|
| linking | −1.0 to −2.0pp | **−4.17pp** | miss (worse than expected — biggest regression in matrix) |
| correction | +0.3 to +0.6pp | **+0.00pp** | miss (correction floor reached: too few exec_errors to rescue) |
| fewshot | +1.0 to +2.5pp | **−0.39pp** | **big miss** — predicted strongest single lift; was actually slightly negative |
| CoT | −1.5 to −2.5pp | **−1.04pp** | hit (regressed less severely than expected) |
| voting | 0 to +0.5pp | killed (3h ETA) | — |

**No strategy improved Qwen3.6.** Best is correction tied with baseline at 63.30%. This is the cleanest possible signal that scaffolding is a function of base-model weakness — when the base is strong (Qwen3.6 has the fewest exec_errors of any base, the highest simple-bucket EX, the highest moderate-bucket EX), every form of intervention either does nothing or degrades.

**What the misses tell us:** Qwen3.6 is *more confident* than the closest analog (Qwen3-32B-thinking). The "modify-the-prompt" strategies (linking, fewshot) hurt more on confident bases — the gap between "model already knows the right SQL" and "scaffolding adds noise" widens with model strength. The correction floor was telegraphed by the small exec_error count (~18) but I underestimated how cleanly it'd hit zero. **The prediction framework "Qwen3.6 ≈ Qwen3-32B-thinking" was the wrong mental model — Qwen3.6 is a different beast: more parametric SQL knowledge, less responsive to in-context teaching.**

The takeaway for *future* work on Qwen3.6: don't try to feed it more prompt context. The leverage is *training* (RL on execution rewards à la Arctic-R1) or *better verification* (CHESS-style unit tester rejecting wrong-but-plausible SQL) — not more in-context examples.

## What's still failing on our current best (correction × Qwen3-Coder-MoE = 61.54%)

579 of 1,534 questions are still wrong (37.7%). Distribution:
- **By difficulty**: 304 simple / 202 moderate / 73 challenging — the *simple* bucket has the most absolute losses; surprising, but reflects that the dataset is mostly simple (925/1534), and the model misses ~33% of those (`100% − 67%`).
- **Worst DBs** (sorted by % wrong): `card_games` 49.7% wrong, `formula_1` 50%, `california_schools` 47.2% wrong, `thrombosis_prediction` 55.8% wrong. These four DBs account for ~50% of all errors.

Heuristic categorization of the 579 wrong predictions (one row can match multiple categories):
- **JOIN-count mismatch (54.6%)**: pred has 0 joins, gold has 2 — pred uses a single-table query when gold pulls from joined tables. The model under-anticipates that questions about (e.g.) "schools in Riverside with charter funding" need the `frpm` join.
- **ORDER BY mismatch (12.6%)**: pred missing or wrong ORDER BY, often paired with a `LIMIT`.
- **GROUP BY mismatch (8.1%)**: aggregation grouping wrong (or absent).
- **CAST mismatch (6.6%)**: classic integer-division trap — gold uses `CAST(... AS REAL) / ...`, pred uses bare `/`.
- **Aggregation mismatch**: SUM/MAX/COUNT/MIN/AVG wrongly chosen ~20% combined.
- **LIKE vs `=`**: 4.3% — typically when filter is "starts with X" but pred uses exact equality.

Sampled cases tell a sharper story:
- **`Percent (%) Eligible Free`** vs **`Free Meal Count / Enrollment`** (q1) — model uses the precomputed % column; BIRD's gold prefers the raw-count ratio. Recurring trap visible across many DBs that have both forms of a metric.
- **Same-named columns on wrong tables** (q28 — `Charter Funding Type` exists on `frpm`; pred uses it on `schools` (no such column)). The audit's earlier *commitment-bias* finding for CoT shows up here too — the model commits to a column without verifying which table holds it.
- **Question-interpretation ambiguity** (q36 — "administrator" → one name vs all six FName/LName slots). BIRD's gold tends to be inclusive when the schema has `Adm1/2/3` slots; the model picks one.

**What would address these:**
- The JOIN-undercount pattern is exactly what schema linking *should* fix when done right — but our linking strategy regressed (-1.37) because the filtered DDL itself confused the model. **A "join-required" pre-prompt hint** (deterministic: count occurrences of FK references that the question's columns might need) is a cheap unexplored lever.
- The `Percent` vs `ratio` trap is **prompt-engineering**: a system-prompt addendum like "*BIRD prefers raw counts (e.g. `Free Meal Count (K-12) / Enrollment (K-12)`) over precomputed percentage columns; use `CAST(... AS REAL)` for integer division*" would likely close 3-5pp of these losses. Free, untested.
- The "wrong-table-for-this-column" failure pattern is what **CHESS's "Information Retriever"** agent addresses: a per-(question, schema) retrieval that returns column→table assignments before SQL generation. Missing piece in our scaffolding.

## The structural ceiling: what no strategy can solve

Built a per-question matrix over **22 of our runs** (5 baselines + 17 strategy cells). For each of the 1534 dev questions, counted how many runs got it right. Bucket distribution:

| bucket | count | % of dev |
|---|---:|---:|
| universally solved (22/22) | 398 | 25.9% |
| mostly solved (≥12/22) | 505 | 32.9% |
| sometimes solved (3-11/22) | 257 | 16.8% |
| rarely solved (1-2/22) | 75 | 4.9% |
| **universally unsolved (0/22)** | **299** | **19.5%** |

**The 299 universally-unsolved questions are the structural ceiling on scaffolding-only approaches.** Maximum theoretical EX with perfect scaffolding = 1535 − 299 = **80.5%**. To exceed that, you need either RL training (Arctic-R1's path) or a fundamentally more capable base.

**Where the unsolvable bucket clusters:**
- By difficulty: moderate 24.4%, challenging 24.8%, simple 16.2% (most absolute losses are on simples because of the dataset weighting)
- By DB: **`card_games` is the worst** (68/191 = 35.6% unsolvable). `formula_1`, `toxicology`, `thrombosis_prediction` close behind at 21–24%. `student_club` is the best (8.2%).

**Gold-SQL feature enrichment in the unsolvable bucket** (vs base rate over all 1534):

| feature | enrichment | implication |
|---|---:|---|
| ≥10 JOINs | 2.57× | extreme multi-table reasoning |
| `LIKE` pattern | 1.95× | string-pattern matching / normalization |
| CTE | 1.71× | multi-stage decomposition |
| subquery | 1.71× | nested SELECT |
| multi-SELECT compose | 1.67× | compositional structure in the gold |
| multi-col SELECT (4+) | 1.60× | wide projection (e.g., "give all admin name slots") |
| NULL handling | 1.57× | `IS NULL` / `IS NOT NULL` filters |
| `CASE WHEN` | 1.31× | conditional projection logic |

**Four root-cause buckets** (read 5 samples manually):

1. **Idiomatic SQL the model doesn't write.** q514's gold uses `LIMIT 0, 10` (MySQL-style `offset, count`). The model writes `LIMIT 10`, returns the wrong slice. Pure syntax-knowledge gap.
2. **Entity-string normalization.** q995 needs filters on exact strings `'Lewis'`, `'Hamilton'`, `'Turkish Grand Prix'` joined across drivers / driverStandings / races. The model often falls back to `LIKE '%Hamilton%'` or drops the race-name filter — schema linking theoretically helps here but our matrix shows it doesn't in practice.
3. **Multi-stage compositional arithmetic.** q125 (challenging, financial): `CAST((T3.A13 - T3.A12) AS REAL) * 100 / T3.A12` — year-over-year percentage. Requires CAST + subtract + multiply + divide chained. Even thinking-mode models drop steps.
4. **BIRD-specific gold idioms.** Recurring across simples and moderates: "raw-count ratio over precomputed `Percent (%)` column" — gold prefers `Free Meal Count / Enrollment` over the existing `Percent (%) Eligible Free` column. The model has no signal that gold prefers this idiom.

**What this implies for "what to do next":**
- The 80.5% scaffolding ceiling is well below Arctic-R1's 70.5% — meaning RL on the same base would hit our scaffolding ceiling and keep climbing. RL is the right next step, not more scaffolding.
- The 299 unsolvable cluster is dominated by *capability* gaps (multi-stage arithmetic, idiom learning, syntax-pattern coverage) — these are the failure modes RL with execution-only reward directly trains *against*.
- The `card_games` wedge (35.6% unsolvable!) suggests targeted training on that DB's question patterns could move EX +2-3pp on its own.

## How the leaders at our scale tier actually win

| approach | size | BIRD dev EX | core lever |
|---|---|---:|---|
| Arctic-Text2SQL-R1-32B (Snowflake) | 32B | **70.5%** | Pure RL on execution-correctness rewards. GRPO (not PPO). Online RL (not batch). Initialize from a strong coder-instruct base — same Qwen2.5-Coder family we used. Tiny reward function: `exec_correct + syntax_valid`, nothing else. |
| OmniSQL-32B (Renmin University) | 32B | ~72% | LoRA fine-tune on **SynSQL-2.5M** — 2.5M synthetic text-to-SQL examples. Built on top of Qwen2.5-Coder-32B-Instruct (the same 57.37% baseline we measured). The data, not the architecture, is the lever. |
| XiYanSQL-QwenCoder-32B (Alibaba) | 32B | 69% | Test-time scaling + ensemble: candidate generation + reranking. Closer to scaffolding family. |
| CHESS | varies | ~81% (corrected BIRD) | Multi-agent: Information Retriever + Candidate Generator + **Unit Tester** verifier. The unit-tester is what we'd add to push correction further. |

**Implications for our position (current best ~62.6% projected):**
- Arctic-Text2SQL-R1's gain over their initialization (their initialization was a Qwen2.5-Coder-32B-Instruct ~ish at 57% → 70.5% = +13pp from RL alone). That's the size of gain we'd plausibly target.
- OmniSQL's gain comes from data. SynSQL-2.5M is the leverage. Without 2.5M synthetic examples, we're playing a different game.
- CHESS's "Unit Tester" is the missing piece in our scaffolding stack — a verifier agent that checks each candidate's result against constraints from the question (e.g., "are there really 0 rows or did you over-filter?") — exactly what our `is_degenerate_result` filter approximates but at the answer-shape level rather than per-question.
- Test-time ensemble + reranking (XiYanSQL): close to our voting+correction strategy, but with a learned reranker rather than majority vote. Could be a future swap.

Sources: [Arctic-Text2SQL-R1 paper](https://arxiv.org/abs/2505.20315) · [Snowflake blog](https://www.snowflake.com/en/engineering-blog/arctic-text2sql-r1-sql-generation-benchmark/) · [OmniSQL paper](https://arxiv.org/html/2503.02240v1) · [OmniSQL repo](https://github.com/RUCKBReasoning/OmniSQL) · [BIRD homepage](https://bird-bench.github.io/)

## What I'd do differently with more time

1. **RL on top of the strongest base + scaffolding.** Arctic-Text2SQL-R1's recipe (reasoning-first SFT → execution-grounded RL with verl/SkyRL) is the documented path past 70%. **GRPO + online RL + execution-only reward** is the specific recipe to copy. We've already got Qwen2.5-Coder-32B-Instruct as the natural starting checkpoint (57.37% → +13pp would target 70%).
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
