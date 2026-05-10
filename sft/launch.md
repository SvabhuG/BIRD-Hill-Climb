# SFT Launch Runbook

End-to-end recipe for fine-tuning Qwen3.6-27B with LoRA on BIRD train via
Modal + Unsloth, then evaluating against BIRD dev with the existing vLLM
`Inference` class.

## What this gets you

- **Baseline**: Qwen3.6-27B zero-shot on BIRD dev = 63.30%.
- **Best non-SFT scaffolding**: stacked x Qwen3.6 = 63.69% (+0.39pp).
- **SFT target**: 66-68% (+3-5pp). Anchor: OmniSQL reports +12pp with 2.5M
  synthetic training pairs; we have 9.4k real BIRD pairs, so expect a
  fraction of that delta (~+3-5pp is consistent with the data-volume gap).

## Prereqs

1. Modal authenticated to `action-svabhu` workspace.
2. BIRD train data already on the `bird-data` volume at
   `/data/bird/train/train.json` + `/data/bird/train/train_databases/`.
   Layout is fixed (see `modal_app.py::fix_train_layout`).
3. `Qwen/Qwen3.6-27B` weights already on the `hf-cache` volume (the baseline
   pipeline pulled them).

If you're missing (2) or (3):
```
MODAL_PROFILE=action-svabhu modal run modal_app.py::download_bird --splits train
MODAL_PROFILE=action-svabhu modal run modal_app.py::fix_train_layout
# Qwen3.6-27B is pulled lazily by vLLM the first time you load it.
```

## Step 1 — Smoke-test the formatter locally (no GPU, no Modal)

```
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m scripts.smoke_test                  # existing tests
.venv/bin/python -m scripts.smoke_test_sft_format       # new SFT format tests
```

Both must pass before launching the GPU run.

## Step 2 — Launch training

```
MODAL_PROFILE=action-svabhu modal run sft/train_unsloth.py::main
```

Optional flags (defaults shown):
- `--base-model Qwen/Qwen3.6-27B`
- `--lora-r 64 --lora-alpha 128 --lora-dropout 0.05`
- `--learning-rate 2e-4`
- `--max-seq-length 16384`
- `--epochs 3 --per-device-batch 1 --grad-accum 16`  (effective batch = 16)
- `--save-every 200`
- `--dataset-limit 0` (set e.g. 500 for a 30-min canary)

Expected wallclock: ~2-3h on 4x B200 with Unsloth's optimizations. Hard cap
in the function decorator is 6h30m so a slow run still fits the budget.

## Step 3 — Watch the logs

```
modal app logs bird-sft -f
```

Loss should drop from ~1.0-1.5 (random) to ~0.3-0.5 over the first ~500
steps, then plateau. If loss is stuck above 1.0 after 200 steps, suspect
tokenizer-template mismatch — the chat template `apply_chat_template` uses
must match the eval-time inference template (vLLM applies it too).

## Step 4 — Mid-run eval (optional)

The trainer checkpoints to `/checkpoints/qwen3.6-27b-bird-sft-v1/checkpoint-<N>/`
every 200 steps. To eval one:

```
.venv/bin/python -m scripts.eval_sft_checkpoint \
    --model /checkpoints/qwen3.6-27b-bird-sft-v1/checkpoint-400 \
    --modal-profile action-svabhu \
    --save-as sft-q3.6-step-400.json
```

NOTE: step checkpoints are LoRA-adapter-only. The current `Inference` class
loads merged models (vLLM, no `--enable-lora` plumbing). Two options:

- **Wait for the end-of-run merge.** `train_sft` saves a merged model to
  `<save_dir>/merged/` after training completes. This is the easy path.
- **Manually merge a mid-run checkpoint.** Run a small Modal CPU job that
  does `PeftModel.from_pretrained(adapter).merge_and_unload().save_pretrained(...)`.
  Not scripted yet — only do this if you need to ship before training finishes.

## Step 5 — Eval the final merged model

The `Inference` class currently mounts `hf-cache` and `bird-data` only. To
load `/checkpoints/.../merged`, edit `modal_app.py`'s `Inference.cls`:

```python
@app.cls(
    image=gpu_image,
    gpu="B200",
    volumes={
        HF_HOME: hf_cache,
        BIRD_ROOT: bird_data,
        "/checkpoints": modal.Volume.from_name("bird-sft-checkpoints"),  # ADD
    },
    ...
)
```

Then:
```
MODAL_PROFILE=action-svabhu modal run modal_app.py::run_baseline \
    --model /checkpoints/qwen3.6-27b-bird-sft-v1/merged \
    --max-tokens 8192 \
    --tensor-parallel-size 4 \
    --save-as sft-qwen3.6-27b-final.json
```

Or via the helper:
```
.venv/bin/python -m scripts.eval_sft_checkpoint \
    --model /checkpoints/qwen3.6-27b-bird-sft-v1/merged \
    --modal-profile action-svabhu \
    --tensor-parallel-size 4 \
    --save-as sft-qwen3.6-27b-final.json
```

## Cost & schedule budget

- 4x B200 on Modal ≈ $24/h (rule of thumb $6/B200-hour).
- Expected training time: ~2-3h. Wallclock budget: 6h.
- Implied cost: ~$50-75 for training + 1 final eval pass (~$10).
- Decision rule: if step-200 mid-eval shows < +1.5pp delta vs. baseline,
  abort and tune (LR, target_modules, n_samples). If > +2pp, keep going.

## Known unknowns

- Unsloth's Qwen3.6 specific path: docs we anchored on cover Qwen3 generally;
  the 3.6 minor uses the same tokenizer family, so the chat template + LoRA
  config should be drop-in. First training run will confirm.
- 4-bit base + LoRA: Unsloth's specialty, but quantizing the base reduces
  the maximum achievable accuracy. If we see < +2pp at converged loss,
  re-run with `--load-in-4bit=False` (LoRA on bf16 base; needs more GPU mem,
  possibly 8x B200).
- The Modal image pin `transformers==5.8.0` matches the eval image. If
  Unsloth refuses to install with that version (incompatible peer dep),
  drop the pin and accept whatever it brings — Qwen3.6 tokenizer support is
  stable across recent transformers releases.
