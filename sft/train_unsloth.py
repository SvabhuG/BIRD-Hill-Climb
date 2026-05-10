"""Modal+Unsloth SFT entrypoint: Qwen3.6-27B + LoRA on BIRD train.

Design choices:

* Separate Modal app (`bird-sft`) from `modal_app.py`'s `bird-climb` to keep
  training and inference concerns isolated and let us iterate on the image
  without churning the eval path.
* Lazy imports of unsloth/trl/torch/datasets/transformers inside the Modal
  function — they aren't pip-installed locally, so the *module* must be
  importable on a laptop (for `modal run sft/train_unsloth.py::main`).
* `bird.sft_format` builds the dataset on Modal (CPU-cheap; reads bird-data
  volume directly).
* LoRA + 4-bit base load (Unsloth's specialty): ~6-8x memory reduction vs.
  full FT, fits Qwen3.6-27B on 4x B200 comfortably.
* Merge the adapter to a standalone model at the end so eval can load it via
  the existing vLLM Inference class without `--enable-lora`.

Reference: https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune
"""
# NOTE: do NOT `from __future__ import annotations` — Modal reads
# `modal.parameter` annotations at class-decoration time. We don't currently
# decorate a class here, but the convention matches modal_app.py.
import json
import os
import time
from pathlib import Path

import modal


APP_NAME = "bird-sft"

# ---------- Volumes (shared with modal_app.py) ----------

bird_data = modal.Volume.from_name("bird-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
sft_checkpoints = modal.Volume.from_name("bird-sft-checkpoints", create_if_missing=True)

BIRD_ROOT = "/data/bird"
HF_HOME = "/root/.cache/hf"
CKPT_ROOT = "/checkpoints"


# ---------- Image ----------
#
# Pins:
#   - torch 2.5.1 + cu124 is Unsloth's recommended baseline (Nov 2025+) and
#     supports Blackwell (B200) wheels.
#   - unsloth is left unpinned to pick up the newest Qwen3 path; if it ships
#     a breaking change we'll pin here.
#   - transformers 5.8.0 keeps wire-compat with the eval image (Qwen3.6
#     tokenizer surface). Unsloth occasionally pins a min version above this
#     — if so, drop the pin and let unsloth bring its own.
#   - flash-attn pre-built wheel; B200 needs >= 2.7.4.
#   - paged_adamw_8bit comes from bitsandbytes.

unsloth_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "wget", "curl")
    # CUDA-bundled torch + Unsloth's prebuilt wheel index
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        # Core Unsloth stack. We let unsloth pull its preferred pins for
        # transformers / trl / peft / accelerate to avoid version skew.
        "unsloth",
        "unsloth_zoo",
        "transformers==5.8.0",
        "trl",
        "peft",
        "accelerate",
        "datasets",
        "bitsandbytes",
        "xformers",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "hf_transfer",
        "tqdm",
        "pydantic>=2",
        "sqlglot>=25",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .env({
        "HF_HOME": HF_HOME,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Unsloth checks this to skip its own torch reinstall.
        "UNSLOTH_FORCE_FLOAT32": "0",
        # Reduce noise from datasets cache misses on the volume mount.
        "HF_DATASETS_DISABLE_PROGRESS_BARS": "1",
    })
    .add_local_python_source("bird")
)


app = modal.App(APP_NAME)


# ============================================================
# Dataset prep (runs on Modal so it can read the bird-data volume)
# ============================================================

@app.function(
    image=unsloth_image,
    volumes={BIRD_ROOT: bird_data, CKPT_ROOT: sft_checkpoints},
    cpu=8,
    memory=16 * 1024,
    timeout=60 * 30,
)
def prepare_dataset(
    split_name: str = "train",
    n_samples: int = 3,
    limit: int = 0,
    cache_jsonl: str = "/checkpoints/bird-sft-train.jsonl",
) -> dict:
    """Materialize the SFT JSONL on the checkpoints volume.

    Idempotent: if `cache_jsonl` exists and has the expected line count we
    short-circuit. Otherwise rebuild from `bird-data` and commit.
    """
    from bird.sft_format import build_sft_dataset, write_sft_jsonl

    cache_path = Path(cache_jsonl)
    if cache_path.exists() and limit == 0:
        n_lines = sum(1 for _ in cache_path.open())
        print(f"[prepare_dataset] reusing {cache_path} ({n_lines} lines)")
        return {"status": "cached", "path": str(cache_path), "n": n_lines}

    print(f"[prepare_dataset] building from {BIRD_ROOT}/{split_name} (limit={limit or 'all'})")
    t0 = time.time()
    examples, stats = build_sft_dataset(
        Path(BIRD_ROOT) / split_name,
        n_samples=n_samples,
        split_name=split_name,
        limit=limit or None,
    )
    print(f"[prepare_dataset] built {stats.n_emitted}/{stats.n_total} in {time.time()-t0:.1f}s")
    print(f"[prepare_dataset] missing_gold={stats.n_missing_gold} "
          f"missing_db={stats.n_missing_db} schema_errors={stats.n_schema_errors}")

    write_sft_jsonl(examples, cache_path)
    sft_checkpoints.commit()
    return {
        "status": "built",
        "path": str(cache_path),
        "n": stats.n_emitted,
        "stats": {
            "n_total": stats.n_total,
            "n_emitted": stats.n_emitted,
            "n_missing_gold": stats.n_missing_gold,
            "n_missing_db": stats.n_missing_db,
            "n_schema_errors": stats.n_schema_errors,
        },
    }


# ============================================================
# Training
# ============================================================

@app.function(
    image=unsloth_image,
    gpu="B200:4",  # 4x B200 — LoRA + 4-bit base fits comfortably.
    volumes={
        BIRD_ROOT: bird_data,
        HF_HOME: hf_cache,
        CKPT_ROOT: sft_checkpoints,
    },
    timeout=6 * 3600 + 30 * 60,  # 6h30m hard cap (mission budget)
    # Heavy hosts: give the container room to swap if 4-bit base spills.
    memory=64 * 1024,
)
def train_sft(
    base_model: str = "Qwen/Qwen3.6-27B",
    dataset_jsonl: str = "/checkpoints/bird-sft-train.jsonl",
    save_dir: str = "/checkpoints/qwen3.6-27b-bird-sft-v1",
    # LoRA hyperparams (defaults from Unsloth's Qwen3 fine-tune guide).
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    # Optim.
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine",
    weight_decay: float = 0.0,
    optim: str = "paged_adamw_8bit",
    # Schedule.
    max_seq_length: int = 16384,
    epochs: int = 3,
    max_steps: int = -1,  # -1 = use epochs
    per_device_batch: int = 1,
    grad_accum: int = 16,
    # Checkpointing / logging.
    save_every: int = 200,
    save_total_limit: int = 4,
    logging_steps: int = 50,
    # Misc.
    seed: int = 42,
    load_in_4bit: bool = True,
    merge_after_train: bool = True,
) -> dict:
    """Fine-tune Qwen3.6-27B with LoRA on BIRD train via Unsloth.

    Steps:
      1. Load `dataset_jsonl` via `datasets.load_dataset("json", ...)`.
      2. `FastLanguageModel.from_pretrained(..., load_in_4bit=True)`.
      3. `FastLanguageModel.get_peft_model(...)` to attach LoRA.
      4. Apply the model's chat template to every example.
      5. Run SFTTrainer (cosine LR, warmup 0.1, paged_adamw_8bit, bf16).
      6. Save the final adapter to `save_dir/adapter`.
      7. Merge LoRA into base weights, save to `save_dir/merged`.
      8. Commit the checkpoints volume.

    Returns: a small JSON summary (final loss, n_steps, save paths).
    """
    # ----- lazy imports (only resolved on the GPU container) -----
    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments

    # Unsloth must be imported BEFORE transformers in the same process for its
    # autograd patches to take effect — but we already imported transformers
    # above. To stay safe, prefer importing unsloth first in this function and
    # let it transparently patch. We re-import transformers below if Unsloth's
    # guard whines.
    from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: I001

    from trl import SFTTrainer

    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    print(f"[train_sft] base={base_model}  data={dataset_jsonl}  out={save_dir}")
    print(f"[train_sft] LoRA r={lora_r} alpha={lora_alpha} dropout={lora_dropout}")
    print(f"[train_sft] lr={learning_rate} epochs={epochs} bs={per_device_batch}x{grad_accum}")
    print(f"[train_sft] max_seq_len={max_seq_length}  load_in_4bit={load_in_4bit}")

    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    # ----- 1. dataset -----
    if not Path(dataset_jsonl).exists():
        raise FileNotFoundError(
            f"{dataset_jsonl} missing; run `prepare_dataset` first or pass --dataset-jsonl"
        )
    ds = load_dataset("json", data_files=dataset_jsonl, split="train")
    print(f"[train_sft] loaded {len(ds)} training examples")

    # ----- 2. model + tokenizer -----
    bf16 = is_bfloat16_supported()
    print(f"[train_sft] bf16_supported={bf16}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # let Unsloth pick (bf16 on Hopper/Blackwell)
        load_in_4bit=load_in_4bit,
        # Read from / write to the shared HF cache volume.
        cache_dir=HF_HOME,
    )

    # ----- 3. LoRA adapter -----
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized variant
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()

    # ----- 4. apply chat template to each example -----
    # Qwen3 models ship a chat_template; rely on it via tokenizer.apply_chat_template.
    def _format_row(row: dict) -> dict:
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,  # full conversation incl. assistant target
        )
        return {"text": text}

    ds_text = ds.map(_format_row, num_proc=4, remove_columns=ds.column_names)
    print(f"[train_sft] applied chat template; sample[0][:400]:")
    print(ds_text[0]["text"][:400])

    # Defensive: drop any example that exceeds max_seq_length after tokenization.
    # Tokenizing once up-front lets us prune the few overflowing rows BIRD has
    # (very-long schemas) instead of silently truncating them.
    def _len_ok(row: dict) -> bool:
        ids = tokenizer(row["text"], add_special_tokens=False, truncation=False)["input_ids"]
        return len(ids) <= max_seq_length

    n_before = len(ds_text)
    ds_text = ds_text.filter(_len_ok, num_proc=4)
    n_after = len(ds_text)
    if n_after < n_before:
        print(f"[train_sft] dropped {n_before - n_after} examples > {max_seq_length} tokens")

    # ----- 5. training args -----
    args = TrainingArguments(
        output_dir=str(save_root),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs if max_steps < 0 else 1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=weight_decay,
        optim=optim,
        bf16=bf16,
        fp16=not bf16,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_every,
        save_total_limit=save_total_limit,
        report_to="none",
        seed=seed,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        # Trainer can resume from the latest checkpoint if the run crashed.
        # Volume is persistent so this is the right default.
        # `resume_from_checkpoint=True` is read at trainer.train() time.
    )

    # ----- 6. trainer -----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_text,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,  # explicit pairs; packing breaks our per-example loss
        args=args,
    )

    print(f"[train_sft] starting training: {len(ds_text)} examples, "
          f"effective batch = {per_device_batch * grad_accum} per step")

    # Resume if a checkpoint is already in save_root (volume is persistent).
    resume = any((save_root / p.name).exists() for p in save_root.glob("checkpoint-*"))
    train_result = trainer.train(resume_from_checkpoint=resume)

    print(f"[train_sft] training complete; final loss={train_result.training_loss:.4f}")

    # ----- 7. save final adapter -----
    adapter_dir = save_root / "adapter"
    print(f"[train_sft] saving adapter to {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    summary = {
        "base_model": base_model,
        "save_dir": str(save_root),
        "adapter_dir": str(adapter_dir),
        "n_train_examples": len(ds_text),
        "global_step": int(train_result.global_step),
        "training_loss": float(train_result.training_loss),
        "lora": {
            "r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout,
            "target_modules": target_modules,
        },
        "optim": {
            "lr": learning_rate, "scheduler": lr_scheduler_type,
            "warmup_ratio": warmup_ratio, "optim": optim,
        },
        "schedule": {
            "epochs": epochs, "per_device_batch": per_device_batch,
            "grad_accum": grad_accum, "max_seq_length": max_seq_length,
        },
    }

    # ----- 8. merge LoRA into base for inference convenience -----
    if merge_after_train:
        merged_dir = save_root / "merged"
        print(f"[train_sft] merging LoRA into base; writing to {merged_dir}")
        try:
            # Unsloth ships `save_pretrained_merged`; falls back to PEFT's
            # `merge_and_unload` if the helper isn't present.
            if hasattr(model, "save_pretrained_merged"):
                model.save_pretrained_merged(
                    str(merged_dir),
                    tokenizer,
                    save_method="merged_16bit",  # bf16 weights for vLLM
                )
            else:
                merged = model.merge_and_unload()
                merged.save_pretrained(str(merged_dir), safe_serialization=True)
                tokenizer.save_pretrained(str(merged_dir))
            summary["merged_dir"] = str(merged_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[train_sft] merge failed: {e!r}; adapter is still saved")
            summary["merge_error"] = repr(e)

    # ----- 9. commit volume + write summary -----
    summary_path = save_root / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    sft_checkpoints.commit()
    print(f"[train_sft] done; summary at {summary_path}")
    return summary


# ============================================================
# Local entrypoint
# ============================================================

@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen3.6-27B",
    save_dir: str = "/checkpoints/qwen3.6-27b-bird-sft-v1",
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    max_seq_length: int = 16384,
    epochs: int = 3,
    per_device_batch: int = 1,
    grad_accum: int = 16,
    save_every: int = 200,
    n_samples: int = 3,
    dataset_limit: int = 0,
    skip_prepare: bool = False,
    skip_merge: bool = False,
):
    """Orchestrator: prepare dataset (idempotent), then train.

    Run from the worktree root:
        MODAL_PROFILE=action-svabhu modal run sft/train_unsloth.py::main
    """
    dataset_jsonl = "/checkpoints/bird-sft-train.jsonl"

    if not skip_prepare:
        print("[main] preparing dataset on Modal")
        prep = prepare_dataset.remote(
            split_name="train",
            n_samples=n_samples,
            limit=dataset_limit,
            cache_jsonl=dataset_jsonl,
        )
        print(json.dumps(prep, indent=2))

    print(f"[main] launching training on {base_model}")
    summary = train_sft.remote(
        base_model=base_model,
        dataset_jsonl=dataset_jsonl,
        save_dir=save_dir,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        epochs=epochs,
        per_device_batch=per_device_batch,
        grad_accum=grad_accum,
        save_every=save_every,
        merge_after_train=not skip_merge,
    )
    print("[main] training summary:")
    print(json.dumps(summary, indent=2))
