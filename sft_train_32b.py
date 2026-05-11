"""Full fine-tuning SFT of Qwen2.5-Coder-32B-Base on BIRD train.

Sharded across 8x B200 GPUs in a single Modal container via FSDP
(HuggingFace Accelerate's FullyShardedDataParallelPlugin).

Recipe is the exp18 recipe, ported onto this repo's infra:
  - Schema rendering: bird.schema.extract_schema + render_ddl_with_samples
  - Prompt format:    bird.prompts.build_messages + messages_to_raw_text
  - SFT target:       bird.sft_format.build_sft_completion (fenced SQL)
  - Volumes:          existing bird-data, hf-cache, bird-sft-checkpoints
  - At inference:     reuse modal_app.Inference with model_name pointed at
                      /checkpoints/sft_32b_base/final (vLLM accepts paths).
                      Same `messages_to_raw_text` flat format → bytewise-aligned.

Why FSDP, full-FT (no LoRA):
  A 32B bf16 model is ~64 GB params + ~64 GB grads + ~128 GB AdamW state ≈ 256
  GB just for optimizer/model state. Doesn't fit one GPU. FSDP FULL_SHARD
  across 8x B200 (180 GB each) divides params, grads, and optimizer state
  evenly → per-GPU footprint drops ~8x. LoRA on a 32B base leaves capacity on
  the table; the Arctic-Text2SQL recipe + earlier 7B SFT proved full-FT pays
  for itself when the substrate is the *Base* checkpoint.

The load-bearing knobs (with reasoning):
  - mp.spawn (NOT accelerate.launch / notebook_launcher)
        Modal containers init CUDA before our code runs. fork() can't re-init
        CUDA in children. spawn-style fresh-interpreter children sidesteps it.
  - torch.cuda.set_device(rank) FIRST in worker
        Without this, every rank's compute device defaults to cuda:0 → FSDP
        crashes with "Inconsistent compute device and device_id".
  - sharding_strategy=FULL_SHARD + auto_wrap on Qwen2DecoderLayer
        Per-block wrap → all-gather/reduce-scatter cost amortizes per layer.
        Wrong wrap (or default) → one giant FSDP unit → OOM at step 1.
  - mixed_precision_policy = bf16 for params/reduce/buffers
        B200 has hw bf16; reduce_dtype=bf16 cuts comm volume in half.
  - backward_prefetch = BACKWARD_PRE
        Overlaps next-layer all-gather with current-layer backward.
        Measured 1.2-1.4x speedup in exp18.
  - sync_module_states=True + init_empty_weights() on non-rank-0
        Only rank 0 reads from disk; others get meta-tensor stubs and receive
        weights via FSDP broadcast during prepare(). Caps CPU mem at ~64 GB
        instead of 8 * ~64 GB = 512 GB OOM.
  - HF gradient_checkpointing flag (NOT manual apply_activation_checkpointing)
        Setting the flag BEFORE prepare() lets HF insert checkpoint wrappers
        internally during forward, while FSDP's auto_wrap_policy still sees
        raw Qwen2DecoderLayer instances at wrap time. Manual wrapping hides
        the class from auto_wrap → silent FSDP degradation.
  - use_orig_params=True
        Required for FSDP + bf16 + AdamW with fused param groups.
  - lr_lambda step counter scaled by num_processes
        accelerate's AcceleratedScheduler ticks scheduler.step() once per
        process per global step. Schedule lengths are scaled by num_processes
        so warmup/total see correct fractional progress.
  - EOS-preserve truncation
        When the prompt+completion exceeds max_seq_tokens, we truncate the
        completion but force the trailing EOS to remain. Without this, long
        examples teach the model to never terminate.

Hyperparams:
  lr = 5e-6, cosine to 10% with 100-step warmup, 1 epoch, micro_batch=1,
  grad_accum=4, world=8 → effective_batch=32, max_seq=8192, bf16, AdamW
  betas=(0.9,0.95), weight_decay=0.01.

Expected runtime:
  ~5k examples / eff_batch 32 → ~156 steps. 8x B200 with grad checkpointing
  at seq_len 8k → ~10-15 s/step → 25-40 min wall clock for 1 epoch. Add
  ~5 min for rank-0 model load + final checkpoint save. Budget 1h with
  overhead.

Launch:
  modal run --detach sft_train_32b.py::main                        # full run
  modal run --detach sft_train_32b.py::main --dry-run              # 50-example smoke
"""
# NOTE: do NOT add `from __future__ import annotations` — Modal reads
# `modal.parameter` annotations at class-decoration time and needs real types.
import os
import time

import modal


# ============================================================
# Modal infra (reuses volumes already used by modal_app.py)
# ============================================================

APP_NAME = "bird-climb-sft-32b"

bird_data = modal.Volume.from_name("bird-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
sft_checkpoints = modal.Volume.from_name("bird-sft-checkpoints", create_if_missing=True)

BIRD_ROOT = "/data/bird"
HF_HOME = "/root/.cache/hf"
CKPT_ROOT = "/checkpoints"

# nvidia/cuda dev image matches exp18's known-good base. PyTorch wheels bundle
# their own CUDA runtime, so the dev tag is overkill for FSDP per se, but it
# guarantees nvcc + CUDA headers are present if we ever want flash-attn3 or
# any custom kernel build later.
sft_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("torch")
    .pip_install("transformers", "huggingface_hub")
    # accelerate>=0.34 has the FullyShardedDataParallelPlugin API used below.
    # Pin a recent stable rather than 'latest' to keep the image hash stable.
    .pip_install("accelerate>=0.34.0")
    .env({"HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install("hf_transfer")
    .add_local_python_source("bird")
)


app = modal.App(APP_NAME)


# Coder-32B BASE — the pre-RLHF, pre-instruct checkpoint. SFT pays off on the
# Base regime, not on Instruct (which has already saturated format alignment).
MODEL_ID = "Qwen/Qwen2.5-Coder-32B"


# ============================================================
# Distributed entrypoint
# ============================================================

@app.function(
    image=sft_image,
    volumes={
        BIRD_ROOT: bird_data,
        HF_HOME: hf_cache,
        CKPT_ROOT: sft_checkpoints,
    },
    # 8x B200 in a single Modal container. Modal NCCL-wires them inside the
    # container; we then spawn 8 worker processes via mp.spawn so each gets
    # its own GPU and torch.distributed initializes a world_size=8 group.
    gpu="B200:8",
    timeout=2 * 60 * 60,  # 2h ceiling — expect ~30-45 min actual
)
def train_sft_32b(
    num_examples: int = 0,             # 0 = use all (after length filter)
    max_prompt_tokens: int = 7168,
    max_seq_tokens: int = 8192,
    micro_batch_size: int = 1,         # per-GPU micro batch
    grad_accum_steps: int = 4,         # 1 * 4 * 8 GPUs = effective 32
    epochs: int = 1,
    lr: float = 5e-6,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    save_every: int = 500,
    output_dir: str = "/checkpoints/sft_32b_base",
    dry_run: bool = False,
):
    """Modal entrypoint: spawn 8 worker processes via torch.multiprocessing.spawn.

    Each worker runs `_train_worker_main` with the same args; the spawn
    helper assigns each its `rank`. The worker sets distributed env vars
    (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) so accelerate's
    Accelerator() picks them up on construction.
    """
    import torch.multiprocessing as mp

    args = dict(
        num_examples=num_examples,
        max_prompt_tokens=max_prompt_tokens,
        max_seq_tokens=max_seq_tokens,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        epochs=epochs,
        lr=lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        save_every=save_every,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    world_size = 8
    mp.spawn(
        _train_worker_main,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )


def _train_worker_main(rank: int, world_size: int, args: dict):
    """Per-rank training body. Runs once per worker process (8 total)."""
    # Distributed env vars must be set BEFORE any torch.distributed or CUDA
    # init. accelerate.Accelerator() picks these up automatically.
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    # Pin THIS process to its assigned GPU. mp.spawn children all see all 8
    # GPUs; without explicit set_device, every rank's "compute device"
    # defaults to cuda:0, then FSDP fails with:
    #   ValueError: Inconsistent compute device and `device_id` on rank N:
    #   cuda:0 vs cuda:N
    # Must be called before any tensor/CUDA op in this process.
    import torch
    torch.cuda.set_device(rank)

    num_examples = args["num_examples"]
    max_prompt_tokens = args["max_prompt_tokens"]
    max_seq_tokens = args["max_seq_tokens"]
    micro_batch_size = args["micro_batch_size"]
    grad_accum_steps = args["grad_accum_steps"]
    epochs = args["epochs"]
    lr = args["lr"]
    warmup_steps = args["warmup_steps"]
    weight_decay = args["weight_decay"]
    save_every = args["save_every"]
    output_dir = args["output_dir"]
    dry_run = args["dry_run"]

    import functools
    import json
    import math
    import random
    from collections import Counter
    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    from accelerate import Accelerator, init_empty_weights
    from accelerate.utils import FullyShardedDataParallelPlugin, set_seed

    # bird/* is shipped via add_local_python_source. Lazy-imported here so the
    # module can also be imported from CPU contexts without torch.
    from bird.data import BirdExample
    from bird.exp18_schema import profile_database
    from bird.sft_format import build_sft_completion, build_sft_prompt

    if dry_run:
        num_examples = 50
        epochs = 1
        save_every = 25

    set_seed(42)

    # ---- 0. Build FSDP plugin + Accelerator ----
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        mixed_precision_policy=mixed_precision_policy,
        limit_all_gathers=True,
        sync_module_states=True,  # rank-0 broadcasts to others during prepare()
        cpu_offload=False,        # B200 has 180GB; offload would just slow us down
        use_orig_params=True,     # required for FSDP + bf16 + fused AdamW
    )
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=grad_accum_steps,
        fsdp_plugin=fsdp_plugin,
    )

    is_main = accelerator.is_main_process

    def log(msg):
        if is_main:
            print(msg, flush=True)

    log("=" * 60)
    log("SFT: Qwen2.5-Coder-32B BASE on BIRD train (FSDP, 8x B200)")
    log("=" * 60)
    log(f"world_size={accelerator.num_processes} | rank={accelerator.process_index}")

    if is_main:
        sft_checkpoints.reload()

    # ---- 1. Load BIRD train ----
    train_root = Path(BIRD_ROOT) / "train"
    train_json = train_root / "train.json"
    train_db_dir = train_root / "train_databases"

    if not train_json.exists():
        raise FileNotFoundError(
            f"{train_json} missing — run "
            "`modal run modal_app.py::download_bird --splits train` and "
            "`modal run modal_app.py::fix_train_layout` first."
        )

    with open(train_json) as f:
        all_tasks = json.load(f)
    log(f"BIRD train: {len(all_tasks)} tasks")

    # ---- 2. Profile databases ----
    # Each rank profiles independently. profile_database is heavier than our
    # old schema extractor (it scans for distinct values on every low-cardinality
    # column), but the result is cached per db_id and BIRD only has ~70 dbs.
    db_counts = Counter(t["db_id"] for t in all_tasks)
    log(f"Profiling {len(db_counts)} databases...")
    profile_cache: dict[str, dict] = {}
    available_dbs: set[str] = set()
    for i, db_id in enumerate(db_counts):
        p = train_db_dir / db_id / f"{db_id}.sqlite"
        if p.exists():
            try:
                profile_cache[db_id] = profile_database(str(p))
                available_dbs.add(db_id)
            except Exception as e:
                log(f"  Skip {db_id}: {e}")
        if (i + 1) % 20 == 0:
            log(f"  Profiled {i+1}/{len(db_counts)} databases...")
    log(f"Profiled {len(available_dbs)} databases")

    tasks = [t for t in all_tasks if t["db_id"] in available_dbs]
    if dry_run:
        tasks = tasks[:num_examples]

    # ---- 3. Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, cache_dir=HF_HOME
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 4. Build (input_ids, labels) pairs ----
    # Prompt format = build_sft_prompt(ex, schema) — flat preamble + markdown
    # body, no chat-wrapping. Same function is used at eval time for bytewise
    # alignment. SFT target = build_sft_completion(gold_sql, eos_token) — raw
    # SQL + EOS, no fences. Loss is computed only on completion tokens (prompt
    # labeled -100).
    log("Building examples...")
    pairs = []
    skipped_long = 0
    for idx, task in enumerate(tasks):
        db_id = task["db_id"]
        profile = profile_cache[db_id]

        ex = BirdExample(
            question_id=task.get("question_id", idx),
            db_id=db_id,
            question=task["question"],
            evidence=task.get("evidence", task.get("hint", "")) or "",
            sql=task.get("SQL", ""),
            difficulty=task.get("difficulty"),
        )
        prompt_text = build_sft_prompt(ex, profile)
        completion_text = build_sft_completion(ex.sql, tokenizer.eos_token)

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)

        if len(prompt_ids) > max_prompt_tokens:
            skipped_long += 1
            continue
        total_len = len(prompt_ids) + len(completion_ids)
        if total_len > max_seq_tokens:
            keep = max_seq_tokens - len(prompt_ids)
            if keep < 8:
                skipped_long += 1
                continue
            # Truncate completion but preserve the trailing EOS so the model
            # always learns to terminate. Without this, slicing chops off the
            # EOS and the model never sees a stop signal on long examples.
            eos_id = tokenizer.eos_token_id
            if completion_ids[-1] == eos_id and len(completion_ids) > keep:
                completion_ids = completion_ids[: keep - 1] + [eos_id]
            else:
                completion_ids = completion_ids[:keep]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        pairs.append({
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": len(prompt_ids),
            "completion_len": len(completion_ids),
            "db_id": db_id,
        })

    log(f"Built {len(pairs)} examples ({skipped_long} skipped over-length)")
    if num_examples and len(pairs) > num_examples:
        random.seed(0)
        random.shuffle(pairs)
        pairs = pairs[:num_examples]
    # Deterministic shuffle so all ranks see the same ordering.
    random.seed(0)
    random.shuffle(pairs)
    log(f"Training on {len(pairs)} examples")

    # ---- 5. Load BASE model (lazy non-rank-0 init for FSDP) ----
    log(f"Loading base model {MODEL_ID} (rank 0 from disk, others meta-init)...")
    if accelerator.is_main_process:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=HF_HOME,
            use_cache=False,
        )
    else:
        # Pull config from cache (cheap), build empty model on meta device.
        # sync_module_states=True will populate from rank 0 during prepare().
        config = AutoConfig.from_pretrained(
            MODEL_ID, trust_remote_code=True, cache_dir=HF_HOME
        )
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        model.config.use_cache = False
    accelerator.wait_for_everyone()

    # HF gradient_checkpointing flag (BEFORE prepare so FSDP's auto_wrap sees
    # raw Qwen2DecoderLayer instances; HF inserts checkpoint wrappers internally).
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    # ---- 6. Optimizer + scheduler (BEFORE accelerator.prepare so FSDP can
    #         wrap optimizer state into its sharding plan) ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = max(
        1,
        len(pairs) // (micro_batch_size * grad_accum_steps * accelerator.num_processes),
    )
    total_steps = steps_per_epoch * epochs

    # accelerate's AcceleratedScheduler ticks scheduler.step() num_processes
    # times per global step (split_batches=False default). Scale schedule
    # lengths by num_processes so lr_lambda sees correct fractional progress.
    sched_warmup = warmup_steps * accelerator.num_processes
    sched_total = total_steps * accelerator.num_processes

    def lr_lambda(step):
        # Linear warmup → cosine decay to 10% of peak.
        if step < sched_warmup:
            return float(step) / float(max(1, sched_warmup))
        progress = (step - sched_warmup) / float(max(1, sched_total - sched_warmup))
        progress = min(1.0, max(0.0, progress))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---- 7. Dataloader ----
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for b in batch:
            seq = b["input_ids"]
            lab = b["labels"]
            pad_n = max_len - len(seq)
            input_ids.append(seq + [pad_id] * pad_n)
            attention_mask.append([1] * len(seq) + [0] * pad_n)
            labels.append(lab + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    dataloader = DataLoader(
        pairs,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,  # collate is a closure → can't pickle to subprocess
        drop_last=True,
    )

    # ---- 8. accelerator.prepare() ----
    # FSDP sharding actually happens here: model wrapped per Qwen2DecoderLayer,
    # optimizer state sharded, dataloader gets a DistributedSampler equivalent
    # inserted under the hood.
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    log(
        f"Training: {len(pairs)} pairs | micro_batch={micro_batch_size} | "
        f"grad_accum={grad_accum_steps} | world_size={accelerator.num_processes} | "
        f"effective_batch={micro_batch_size * grad_accum_steps * accelerator.num_processes} | "
        f"epochs={epochs} | total_steps={total_steps} | warmup={warmup_steps} | lr={lr}"
    )

    # ---- 9. Train loop ----
    def save_checkpoint(tag: str):
        ckpt_dir = os.path.join(output_dir, tag)
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        state_dict = accelerator.get_state_dict(model)
        if is_main:
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrapped.save_pretrained(
                ckpt_dir,
                is_main_process=True,
                save_function=accelerator.save,
                state_dict=state_dict,
            )
            tokenizer.save_pretrained(ckpt_dir)
            sft_checkpoints.commit()
            log(f"  Saved checkpoint: {ckpt_dir}")
        accelerator.wait_for_everyone()

    model.train()
    start_time = time.time()
    global_step = 0
    running_loss = 0.0
    running_count = 0

    for epoch in range(epochs):
        log(f"\n--- Epoch {epoch+1}/{epochs} ---")
        for batch in dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.detach().float().item()
            running_count += 1

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % 10 == 0 or global_step == 1:
                    avg = running_loss / max(running_count, 1)
                    elapsed = time.time() - start_time
                    eta = elapsed / global_step * (total_steps - global_step) if global_step else 0
                    cur_lr = scheduler.get_last_lr()[0]
                    log(
                        f"Step {global_step}/{total_steps} | loss={avg:.4f} | "
                        f"lr={cur_lr:.2e} | elapsed={elapsed:.0f}s | eta={eta:.0f}s"
                    )
                    running_loss = 0.0
                    running_count = 0

                if global_step % save_every == 0:
                    save_checkpoint(f"checkpoint-{global_step}")

    # ---- 10. Final save ----
    save_checkpoint("final")

    elapsed = time.time() - start_time
    log("=" * 60)
    log(f"Training complete: {global_step} steps in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log(f"Final checkpoint: {os.path.join(output_dir, 'final')}")
    log("=" * 60)

    return {
        "output_dir": output_dir,
        "steps": global_step,
        "elapsed": elapsed,
    }


@app.local_entrypoint()
def main(
    num_examples: int = 0,
    epochs: int = 1,
    lr: float = 5e-6,
    micro_batch_size: int = 1,
    grad_accum_steps: int = 4,
    save_every: int = 500,
    output_dir: str = "/checkpoints/sft_32b_base",
    dry_run: bool = False,
):
    """Spawn the FSDP training job. Thin entrypoint so `modal run --detach`
    truly detaches.
    """
    print("=== SFT: Qwen2.5-Coder-32B BASE on BIRD train (FSDP, 8x B200) ===")
    print(f"  num_examples={num_examples or 'all'}  epochs={epochs}  lr={lr}")
    print(f"  micro_batch={micro_batch_size}  grad_accum={grad_accum_steps}")
    print(f"  effective_batch={micro_batch_size * grad_accum_steps * 8} (across 8 GPUs)")
    print(f"  output_dir={output_dir}  dry_run={dry_run}")

    fc = train_sft_32b.spawn(
        num_examples=num_examples,
        epochs=epochs,
        lr=lr,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        save_every=save_every,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    print(f"\nSpawned function call: {fc.object_id}")
    print(f"Tail logs:        modal app logs {APP_NAME}")
    print(f"Or fetch result:  FunctionCall.from_id('{fc.object_id}').get()")
