# Agentic v2 — keep_baseline_sql + expanded routing

Brain: Qwen3-Coder-30B-A3B-Instruct. Same as v1.

## TL;DR

**v2 does not improve over v1.** Merged full-dev EX is exactly the baseline:
60.63% (n=1534), net delta vs greedy = 0 (13 fixes, 13 breaks). v1 stays as the
headline `agentic_routed` cell at 62.65% (+2.02pp).

But the experiment was informative: `keep_baseline_sql` worked as designed (cut
per-question break rate from 70% to 37%), and we now know which routing rules
are productive.

## Routing-set composition (vote-share threshold raised 0.625 -> 0.75)

| | v1 | v2 |
|---|---|---|
| n_routed                 | 187 | **256** |
| baseline_correct in routed | 10 | 35 |
| baseline EX on routed    | 5.35% | 13.67% |
| breakdown (any-overlap)  | exec=28 deg=118 lvs=70 | exec=28 deg=118 lvs=107 t07=76 hint=7 |

Rule 4 (T=0.7-vs-greedy result-set disagreement on 1506 executable pairs):
76 disagreements, of which 25 were baseline-correct cases — i.e. T=0.7
disagreement is a poor proxy for greedy-being-wrong (33% precision, vs ~95%
for the degenerate-result rule). Rule 5 (hint-column-not-in-SQL) was tightened
post-pilot to backticked names only after the loose CamelCase + snake_case
regex matched 377 hints (50.7% baseline-correct, basically chance). Final
rule 5 fires on only 7 questions.

## Did keep_baseline_sql cut the break rate?

Yes, but at a cost. Among baseline-correct routed cases:

| | v1 | v2 |
|---|---|---|
| baseline-correct routed cases    | 10 | 35 |
| broken by agent                  | 7 (70%) | 13 (37%) |
| kept by agent                    | 3 (30%) | 22 (63%) |

Agent finish-reason mix on v2 routed: submit=39 (15%), **keep_baseline=92 (36%)**,
no_tool_call=68 (27%), budget=57 (22%). The agent leaned hard on the new
affordance — 36% of routed calls returned the baseline unchanged.

So Improvement A delivered: break rate halved per question. But it also tanked
the fix rate. v1 produced 38 fixes; v2 produced 13. Net delta collapsed from
+31 to 0.

## Per-rule breakdown on v2 (paired vs baseline)

| rule | n | baseline_correct | agent_fixes | agent_breaks | net |
|---|---|---|---|---|---|
| exec_error          |  28 |  0 |  0 |  0 |  +0 |
| degenerate          | 118 |  1 | 11 |  0 | **+11** |
| low_vote_share      | 107 | 18 |  0 |  4 |  -4 |
| t07_disagree        |  76 | 25 |  3 | 10 |  -7 |
| hint_col_missing    |   7 |  1 |  0 |  1 |  -1 |

Two pieces of news:

1. **The degenerate-result rule is doing all the work.** It's high-precision
   (only 1 of 118 was baseline-correct) and the agent fixes 9.3% of them. The
   other rules either hurt or stay flat.
2. **Rule 4 (T=0.7 disagreement) is the largest single regression.** With 25
   baseline-correct cases caught in the net and only 3 fixes, it costs -7 net.
   The hypothesis that cross-temperature disagreement signals "the baseline is
   uncertain, send to agent" doesn't hold for a strong coder model — when the
   greedy answer is good but lies in a flat softmax region, T=0.7 just picks
   a different valid (or invalid) phrasing without indicating defect.
3. **Rule 1 (exec_error) regressed too.** v1 got 13 fixes out of 28 exec_error
   cases. v2 got 0 — the agent now keeps the baseline SQL on those (since the
   prompt anchors hard on it), which is by definition still an exec_error.

## Statistical significance of v2 vs v1

McNemar on the paired full-dev predictions (n=1534):
v2 vs greedy: chi2 = 0.038, p = 0.84, 95% CI on delta = [-0.65, +0.65]pp.
**v2 is statistically indistinguishable from greedy.** v1 still has p = 7.7e-6
vs greedy.

v2-vs-v1 directly is not informative (most of v2's predictions ARE v1's
baseline, by design). The right comparison is each-vs-greedy.

## Conclusion: keep v1 as the headline cell

`results/matrix.json` agentic_routed stays at v1's 62.65% (+2.02pp). The v2
experiment showed (a) `keep_baseline_sql` works for break-rate reduction but
the gain is dwarfed by the fix-rate collapse, and (b) the routing-set
expansion was net-negative because the new rules (especially T=0.7) are
low-precision.

A future v3 would: keep `keep_baseline_sql` (it's the right affordance) but
roll back to v1's routing set, or restrict v2's union to exec_error +
degenerate only. Predicted recovery: ~+2.5pp at a lower break rate than v1.
Not run today — out of budget.

## Files

- `bird/agentic.py` — added `keep_baseline_sql` tool, baseline-anchored prompt
- `modal_app_agentic.py` — Sampler class + `run_t07_baseline` + `run_t07_disagreement` + baseline plumbing in `run_agentic_routed`
- `scripts/build_routing_set.py` — rules 4 & 5, threshold raised to 0.75
- `scripts/mcnemar.py` — paired contingency + 95% CI utility
- `scripts/smoke_test_agentic.py` — extended for keep_baseline_sql + baseline prompt
- bird-results volume:
  - `baseline-qwen3-coder-30b-a3b-instruct-dev-t07.json` — T=0.7 sample, 1534
  - `t07-disagreement-qwen3-coder-30b-a3b-instruct-dev.json` — Rule-4 flags
  - `agentic-routed-q3cm-dev-v2.{traces,routed_only,merged}.json`

## Reproduce

```bash
# 1. T=0.7 sample on full dev (~3 min wall, ~67s of generation)
modal run modal_app_agentic.py::run_t07_baseline

# 2. Diff greedy and T=0.7 result-sets to build Rule-4 flags (~30s)
modal run modal_app_agentic.py::run_t07_disagreement

# 3. Rebuild routing set (local, pure-Python; takes <10s)
python scripts/build_routing_set.py --refresh

# 4. Run the agent on the new routed set + score (single command, ~25 min wall
#    incl. ~5 min model load)
modal run modal_app_agentic.py::run_agentic_routed --save-as agentic-routed-q3cm-dev-v2.json
```
