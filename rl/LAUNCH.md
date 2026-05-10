# Prime Intellect RL Training — BIRD Execution-Reward GRPO

Step-by-step runbook for kicking off a Prime Intellect GRPO run against BIRD with
execution-accuracy as the reward signal. The training reward is computed by
`bird/rl_env.py::compute_reward`, which calls the exact same `_execute` +
`_rows_equal` machinery as our offline EX evaluator (`bird/eval.py`). No drift
between training signal and the metric we report.

---

## 0. Background

| Item              | Value                                                                 |
|-------------------|-----------------------------------------------------------------------|
| Primary model     | `Qwen/Qwen3-Coder-30B-A3B-Instruct` (MoE, ~3B active / token)          |
| Fallback model    | `Qwen/Qwen2.5-Coder-32B-Instruct` (dense; exact Arctic-R1 base)        |
| Algorithm         | Online GRPO, group size 8                                             |
| Reward            | Binary: 1.0 if `_rows_equal(pred_rows, gold_rows)` else 0.0           |
| Train set         | BIRD train (~9,428 examples after dropping ones without gold SQL)      |
| Intra-run eval    | 200-q dev subset every 250 steps                                      |
| Full eval         | 1534-q BIRD dev via `scripts/eval_rl_checkpoint.py` after each ckpt    |
| Steps             | ~6,000 (Arctic-R1 recipe); bump to 10K if curve hasn't plateaued       |
| Checkpointing     | Every 500 steps                                                       |

---

## 1. Prereqs

1. **Prime Intellect account** with either:
   - **Hosted Training** access — for `prime rl ...` cloud submissions, or
   - A **reserved cluster** (Slurm) with at least 2 nodes × 8 GPUs (B200 or H200
     recommended; H100 also works but reduces step throughput).
   See [`app.primeintellect.ai/dashboard/quotes`](https://app.primeintellect.ai/dashboard/quotes).

2. **BIRD train + dev** on a node-accessible filesystem (NFS / shared volume).
   Layout:
   ```
   /shared/bird/train/train.json
   /shared/bird/train/train_databases/<db_id>/<db_id>.sqlite
   /shared/bird/dev/dev.json
   /shared/bird/dev/dev_databases/<db_id>/<db_id>.sqlite
   ```
   We already have these on our `bird-data` Modal volume. To stage them to a
   Prime-accessible filesystem, choose ONE of:
   - **NFS upload**: `rsync -avz /local/path/to/bird/ <node>:/shared/bird/`
   - **HF Hub mirror** (preferred for reproducibility): export `bird.data.load_split`
     output to a HF Dataset and push private, then update
     `bird/rl_env.py::build_dataset` to read from `datasets.load_dataset(...)`.
     (Not required for v1 — local filesystem is fine.)

3. **W&B and HF tokens** on the head node:
   ```bash
   export WANDB_API_KEY=...
   export HF_TOKEN=...        # required only if model gates access
   ```

4. **Budget**: see § 8 below for cost estimates. Plan for ~$200-500 for a full
   6K-step 30B-class run.

---

## 2. Install prime-rl + verifiers on the head/training node

```bash
# uv is the canonical installer
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install prime          # the `prime` CLI

# Clone prime-rl into your shared filesystem (so all nodes see the same checkout)
git clone https://github.com/PrimeIntellect-ai/prime-rl.git /shared/prime-rl
cd /shared/prime-rl
uv sync --all-extras

# Optional but recommended on Hopper/Blackwell: build FlashAttention-3
uv pip install "flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@main#subdirectory=hopper" --no-build-isolation

# Validate
uv run python -V                          # should be 3.12
uv run python -c "import flash_attn"
```

`verifiers` is pulled in automatically as a prime-rl dependency. If you want to
develop the env locally:
```bash
uv pip install verifiers
```

---

## 3. Stage this repo as a verifiers environment

Two registration paths — pick whichever is more convenient:

### Path A: Local installable environment package (recommended)

```bash
# From the bird-climb-action repo root:
cp -r . /shared/prime-rl/environments/bird_exec   # or symlink, your call
```

Create `/shared/prime-rl/environments/bird_exec/pyproject.toml` (already in
this repo if you copied the whole tree — `pyproject.toml` here declares the
`rl` optional dep group). Add a project-level `[project.entry-points]`
declaration so verifiers can discover `load_environment`:

```toml
[project.entry-points."verifiers.environments"]
"bird-exec" = "bird.rl_env:load_environment"
```

Then:
```bash
cd /shared/prime-rl/environments/bird_exec
uv pip install -e .[rl]
prime env install bird-exec                      # registers with the local prime tool
```

### Path B: Publish to the Environments Hub (if you want to share)

```bash
cd /shared/prime-rl/environments/bird_exec
prime env push --path .                          # pushes to app.primeintellect.ai
# Then in any rl.toml: id = "<your-org>/bird-exec"
```

For v1 we'll stay with Path A.

---

## 4. Smoke-test the environment LOCALLY before launching training

```bash
# In your bird-climb-action checkout:
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[rl]"
python -m scripts.smoke_test                    # baseline scaffolding tests
python -m scripts.smoke_test_rl_env             # exercises compute_reward
```

Both must pass before you spend any GPU money. The second script verifies:
- Correct SQL ⇒ reward 1.0
- Wrong-rows SQL ⇒ reward 0.0
- Syntax-error SQL ⇒ reward 0.0 (no crash)
- Runaway SQL (recursive CTE) ⇒ reward 0.0 after timeout (no hang)

---

## 5. Edit the config for your cluster

Open `rl/grpo_config.toml` and adjust:

| Setting                                  | What to set                                               |
|------------------------------------------|-----------------------------------------------------------|
| `[orchestrator.train.env.args] train_root` | Absolute path to BIRD train on the shared FS.            |
| `[orchestrator.train.env.args] eval_root`  | Absolute path to BIRD dev.                                |
| `[trainer.model] ep`                      | = number of GPUs per training node (8 for B200x8 box).    |
| `[inference.parallel] tp`                 | = number of GPUs per inference node.                      |
| `[wandb] name`                            | Distinguish runs (e.g. `qwen3-coder-30b-a3b-grpo-v1`).    |

For single-node debugging (1 node, 8 GPUs split between train+infer), add a
`[deployment]` block:
```toml
[deployment]
num_train_gpus = 4
num_infer_gpus = 4
```

For multi-node production (the typical configuration), add:
```toml
[deployment]
type             = "multi_node"
num_train_nodes  = 1
num_infer_nodes  = 1
```
…and a `[slurm]` block per `examples/qwen30b_math/rl.toml`.

---

## 6. Launch training

### Single-node (debug / small run):
```bash
cd /shared/prime-rl
uv run rl @ /shared/bird-climb-action/rl/grpo_config.toml \
   --output-dir /shared/outputs/bird-rl-v1
```

### Multi-node Slurm (production):
```bash
cd /shared/prime-rl

# Optional: open a tmux session with split-pane logs
bash scripts/tmux.sh bird-rl /shared/outputs/bird-rl-v1
# (then run the rl command in window 0; logs auto-tail in window 1)

uv run rl @ /shared/bird-climb-action/rl/grpo_config.toml \
   --output-dir /shared/outputs/bird-rl-v1
```

Logs land at:
```
/shared/outputs/bird-rl-v1/logs/trainer.log
/shared/outputs/bird-rl-v1/logs/orchestrator.log
/shared/outputs/bird-rl-v1/logs/inference.log
/shared/outputs/bird-rl-v1/logs/envs/train/bird-exec/*.log
```

Checkpoints land at:
```
/shared/outputs/bird-rl-v1/checkpoints/step-500/
/shared/outputs/bird-rl-v1/checkpoints/step-1000/
...
```

---

## 7. Evaluate each checkpoint on the full 1534-q BIRD dev

`prime-rl`'s intra-run eval uses a 200-q subset for speed. For the headline
EX number we need the full set:

```bash
# Local-path checkpoint (from the live run):
python -m scripts.eval_rl_checkpoint \
   /shared/outputs/bird-rl-v1/checkpoints/step-2000 \
   --split dev

# Or HF-hub path (e.g. baselines, Arctic-R1 itself):
python -m scripts.eval_rl_checkpoint Snowflake/Arctic-Text2SQL-R1-32B --split dev
```

This shells out to our existing Modal `Inference.run_baseline` entrypoint
under the hood — same scaffolding-free eval we used for the 22 baselines,
so numbers compare apples-to-apples. Results are saved on the `bird-results`
Modal volume under `rl-checkpoints/<run>/step-<N>.json`.

Recommended cadence:
- Eval `step-500`, `step-1000`, `step-2000`, `step-3000`, `step-4500`, `step-6000`.
- If `step-6000` is still climbing, extend `max_steps` to 10000 and continue training.

---

## 8. Cost / sizing notes

| Setup                    | Throughput        | 6K-step run       | Cost @ $1.50/GPU-hr |
|--------------------------|-------------------|-------------------|---------------------|
| 1×8 B200 (train+infer)   | ~5-8 steps/min    | ~12-20 hours      | $150-240            |
| 1×8 train + 1×8 infer (B200) | ~15-25 steps/min  | ~4-7 hours        | $100-170            |
| 2×8 train + 2×8 infer    | ~30-50 steps/min  | ~2-3.5 hours      | $100-170            |

(Throughput estimates extrapolated from the prime-rl `qwen30b_math` example
report — adjust as you measure.)

---

## 9. Troubleshooting

- **`trainer.model impl="custom"` errors with "unsupported architecture":**
  Qwen3-Coder-MoE may not register as `qwen3_moe` despite sharing the family.
  Try `impl = "hf"` first — slower but always works. Failing that, switch to
  `rl/grpo_config_qwen2.5_fallback.toml` (dense 32B, no MoE).

- **OOM during weight broadcast:** lower `[orchestrator] batch_size` (e.g.
  to 128) and/or raise `[trainer.model.ac_offloading] max_inflight_activations`.

- **Reward stuck at ~0.6 and not improving past step ~500:**
  Likely LR too low. Bump `[trainer.optim] lr` from `5e-7` to `1e-6` and
  resume from a recent ckpt. Or your `n_samples_schema` is too small —
  larger sampled-rows blocks give the model more grounding. Try 5.

- **Reward variance collapses (all rollouts 0 or all 1):** dataset is too
  easy or too hard. Confirm `[orchestrator.buffer]` thresholds are set
  (default config drops uniform-success and uniform-failure prompts).

- **Train rewards rise but dev EX (from `eval_rl_checkpoint.py`) doesn't:**
  schema-block format mismatch between train and eval. Confirm
  `n_samples_schema` matches what our 22-baseline runs used (default 3).

---

## 10. Post-run

1. Pick the best ckpt by full-dev EX.
2. Push to a private HF repo:
   ```bash
   hf upload <org>/bird-qwen3-coder-rl-stepNNNN /shared/outputs/bird-rl-v1/checkpoints/step-NNNN
   ```
3. Add the model's EX (+ delta over base) to `presentation.md`'s leaderboard table.
4. Save the W&B run URL into `results/runs.md` (or wherever we track runs).

Done.
