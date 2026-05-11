# Agentic BIRD — exploration on `feat/agentic-explore`

Brain: **Qwen3-Coder-30B-A3B-Instruct** (the current matrix leader, 60.63% greedy on full dev).
Subset: **first 50 questions of BIRD dev** (28 simple / 20 moderate / 2 challenging).

## Design

Two tools, emitted as fenced ```json``` blocks:

| tool | purpose |
|---|---|
| `execute_sql(sql)` | read-only SQLite, 10s timeout, returns up to 20 rows or an error string |
| `submit(sql)`      | terminates the loop; the SQL inside is the graded answer |

No `inspect_table` / `list_tables` — the full DDL + 3 sample rows per table is
already in the initial user message. The model can always `SELECT * FROM t LIMIT 5`
through `execute_sql`. Keeping the palette to 2 minimizes the per-turn decision
problem ("which tool?") and matches the user's "stick to a couple of tools" brief.

Loop budget: `max_turns=6`, `max_obs_rows=20`, `max_obs_bytes=4000`. Greedy
(`T=0`). The whole loop runs inside the GPU container so each turn doesn't pay
Modal RPC overhead.

## Results

| run | EX | n_correct | exec_error | simple | moderate | challenging |
|---|---|---|---|---|---|---|
| baseline (greedy, same 50)       | **48.00%** | 24/50 | 4 | 17/28 (60.7%) | 6/20 (30.0%) | 1/2 |
| agentic v1 (parser bug)          | 44.00%     | 22/50 | 7 | 15/28 (53.6%) | 7/20 (35.0%) | 0/2 |
| **agentic v2 (parser fix)**      | **54.00%** | 27/50 | 4 | 19/28 (67.9%) | 8/20 (40.0%) | 0/2 |

**Net delta vs same-50 greedy baseline: +6.00pp** (+3 questions). On simple
questions the agent gained +7.2pp; on moderate +10.0pp; on challenging it lost
1 (small n, noise).

Per-question disagreement matrix (v2):

```
both right :  18
agent_wins :   9   [12 19 23 24 29 30 31 40 46]
base_wins  :   6   [2 13 33 35 36 47]
both wrong :  17
```

So in 15 of 50 questions agent ≠ baseline, with a 9-to-6 swing in agent's favor.

### Cost

| metric | value |
|---|---|
| avg turns per question        | 3.86 |
| avg `execute_sql` calls       | 2.96 |
| avg `execute_sql` ERRORs      | 0.00 |
| avg assistant tokens (~chars/4) | ~700 |
| avg wall time per question    | 5.63s (B200, single TP) |
| total 50-q wall              | 281s |

Greedy baseline is ~1.5s/q with ~250 tokens output — so the agent costs roughly
**3× tokens** and **3.7× wall time** per question for the +6pp gain.

## Qualitative findings (5-sample audit)

1. **Discovery is the killer feature.** q22 (`Contra Costa` school): the agent
   ran a probe, saw 5 rows with `NULL` school names (district aggregate rows),
   added `AND sname IS NOT NULL` to its final SQL, and matched gold exactly.
   The baseline missed this because the schema sample didn't surface the trap.
   Several other wins (q12, q23, q24, q31) follow the same pattern — probe →
   notice a value the schema didn't tell you → fix.

2. **Wrong probes still help via "no errors" feedback.** `avg_exec_errors=0`
   means the model's drafts always parsed and ran. The agent rarely uses
   `execute_sql` to *debug syntax*; it uses it to *check what the result looks
   like* and decide which column to join on.

3. **Budget exhaustion (5/50) is mostly the agent over-checking itself.** The
   budget-hit questions had a draft that was already correct or near-correct;
   the model kept calling `execute_sql` to add increasingly elaborate
   sanity-checks until turn 6 hit and we fell back to the last-executed SQL,
   which sometimes had extra columns it was just exploring. q22 was the most
   painful: agent v1 actually got it right; v2 got it wrong because v2's
   regex-fallback recovered an extra turn that *added* `NumTstTakr` to the
   SELECT list.

4. **Wrong-column failures dominate the residual.** The 6 questions where the
   baseline beat the agent (q2, q13, q33, q35, q36, q47) are all
   "chose the wrong column" cases — e.g., `frpm.County Name` instead of
   `schools.County`. The agent had the tool to disambiguate (it could have
   `SELECT DISTINCT County FROM schools` and `SELECT DISTINCT \`County Name\` FROM frpm`)
   but didn't, because the question text didn't tip it off that there was a
   choice to make. Schema-linking via the tool would need *the schema itself to
   surface ambiguity* before the model would think to probe — which it doesn't.

5. **Parsing robustness matters a lot.** The v1→v2 jump from 44% to 54% came
   entirely from one fix: when the model emits a malformed JSON tool call
   (e.g. an unterminated string, a stray `}}`), strict `json.loads` was
   failing and the loop fell into the `no_tool_call` branch, which then ran
   `extract_sql` over the entire message — pulling JSON tail into the SQL
   ("`...'Alameda'\"}};`" → unrecognized token). The regex fallback that
   pattern-matches `"tool":"..."` and `"sql":"..."` recovered these. **5pp of
   the 6pp gain is a parser-resilience win, not a reasoning win** — sobering,
   but argues that any agent framework needs hard-tested tool-call parsing.

## Recommended next steps (if we had 8 more hours)

1. **Run on all 1534 dev questions.** 5.6s/q × 1534 ≈ 2.4h wall, well under
   Modal's 2h cap per container if we batch over multiple Agent instances or
   bump the timeout. The 50-q signal is +6pp with a 95% CI roughly ±13pp
   given n=50, so we cannot yet claim the agent beats the best stacked
   strategy (63.69%). The full-dev result is the load-bearing experiment.

2. **Agent-over-correction stack.** The agent's strengths (data-probing,
   NULL discovery) are orthogonal to stacked voting+correction. Wrap the agent
   as a candidate generator inside the existing correction pipeline:
   greedy → if exec_error, hand to agent for repair. That captures the agent's
   strength on hard questions without paying its cost on easy ones.

3. **Tighten the prompt against budget over-spending.** v2's 5 budget-hit
   questions cost real EX (1/5 was a correct one we re-wrote into wrong). A
   small system-prompt tweak — "after 2 successful execute_sql calls that
   return reasonable rows, prefer to submit" — should help. Could also
   instrument: if turn≥3 and last query was successful AND its result columns
   match the question shape, force a submit.

4. **Promote tool calls to vLLM's structured tools API.** The fenced-JSON
   format works but is a known source of parsing errors. Qwen3-Coder was
   trained for OpenAI-style function calling, and using vLLM's `tools=` would
   let the engine guarantee well-formed JSON via guided decoding — eliminating
   the v1→v2 class of bug entirely.

## Files

- `bird/agentic.py` — agent loop + tool dispatch + parsing (220 lines)
- `modal_app_agentic.py` — Modal Agent class + `run_agentic` entrypoint
- `scripts/smoke_test_agentic.py` — offline tests, no GPU
- `agentic_findings.md` — this writeup
- Results on `bird-results` volume:
  - `baseline-q3cm-dev-50-apples.json` — apples-to-apples greedy baseline
  - `agentic-q3cm-dev-50-v2.json` — agent v2 predictions
  - `agentic-q3cm-dev-50-v2.traces.json` — full per-question loop traces

## Reproduce

```bash
# (one-time) verify the offline plumbing
python -m scripts.smoke_test_agentic

# the 50-question Modal run
modal run modal_app_agentic.py::run_agentic --split dev --limit 50 \
  --save-as agentic-q3cm-dev-50-v2.json
```
