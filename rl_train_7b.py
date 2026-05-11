"""GRPO RL of Qwen2.5-Coder-7B-Instruct on BIRD with execution-accuracy reward.

Single B200, LoRA adapter, TRL's GRPOTrainer. Goal is to *demonstrate the
technique* — show a measurable lift over the 47.85% greedy baseline — not to
hit Arctic-R1's 73% (they used 6000 steps on 32B). We aim for +2-6pp in ~1h.

Recipe — designed to fit on one B200 (180GB) with comfortable margin:
  - Model: Qwen/Qwen2.5-Coder-7B-Instruct (we have its baseline EX in matrix)
  - LoRA r=32 alpha=64 on q/k/v/o; base frozen → ref model is the base itself
  - GRPO group size 4, lr 1e-5, max_steps 100
  - per_device_train_batch_size = 1, grad_accum 8 → 8 prompts * 4 gens = 32 rollouts/step
  - max_prompt_length 2048, max_completion_length 512, temperature 0.9
  - bf16

Reward = binary execution accuracy:
  - parse SQL via bird.prompts.extract_sql
  - execute predicted vs gold against per-question SQLite DB w/ 5s timeout
  - row-set equality (bird.eval._rows_equal); set-equality, type-normalized
  - GOLD QUERIES ARE CACHED per question_id across rollouts — gold execution
    is expensive and deterministic, no point re-running it per rollout

Smoke vs full:
  modal run rl_train_7b.py::main --smoke True              # 5 min, 2 steps, 20 qs
  modal run --detach rl_train_7b.py::main                  # full training
  modal run rl_train_7b.py::eval_rl                        # eval saved adapter on dev
"""
# NOTE: do NOT add `from __future__ import annotations` — Modal reads
# `modal.parameter` annotations at class-decoration time and needs real types.
import os
import time

import modal


# ============================================================
# Modal infra (reuses volumes already used by modal_app.py)
# ============================================================

APP_NAME = "bird-climb-rl-7b"

bird_data = modal.Volume.from_name("bird-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
sft_checkpoints = modal.Volume.from_name("bird-sft-checkpoints", create_if_missing=True)
results_vol = modal.Volume.from_name("bird-results", create_if_missing=True)

BIRD_ROOT = "/data/bird"
HF_HOME = "/root/.cache/hf"
CKPT_ROOT = "/checkpoints"
RESULTS_ROOT = "/results"

# nvidia/cuda dev image so nvcc + CUDA headers are present in case
# bitsandbytes / flash-attn build paths trigger. PyTorch wheels bundle their
# own runtime so this is belt-and-braces.
rl_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("torch")
    # TRL >= 0.16 has GRPOTrainer with the modern (prompts, completions, **kwargs)
    # reward signature; older versions had a different API.
    .pip_install(
        "transformers==4.57.0",
        "trl>=0.16",
        "peft>=0.13",
        "accelerate>=0.34",
        "datasets",
        "huggingface_hub",
    )
    .env({"HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install("hf_transfer")
    # For the eval pass we also want vLLM, kept in this same image so we don't
    # have to manage two images.
    .pip_install("vllm==0.11.0", "sqlglot>=25")
    .add_local_python_source("bird")
)


app = modal.App(APP_NAME)


MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
RL_OUTPUT_DIR = "/checkpoints/rl_7b_grpo"


# ============================================================
# Training
# ============================================================

@app.function(
    image=rl_image,
    volumes={
        BIRD_ROOT: bird_data,
        HF_HOME: hf_cache,
        CKPT_ROOT: sft_checkpoints,
    },
    gpu="B200",
    timeout=90 * 60,  # 90 min hard cap
)
def train_rl_7b(
    num_examples: int = 800,
    max_steps: int = 100,
    learning_rate: float = 1e-5,
    num_generations: int = 4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_prompt_length: int = 2048,
    max_completion_length: int = 512,
    temperature: float = 0.9,
    lora_r: int = 32,
    lora_alpha: int = 64,
    output_dir: str = RL_OUTPUT_DIR,
    smoke: bool = False,
):
    """Run GRPO training on a B200. Saves a LoRA adapter to `output_dir`/final."""
    import json
    import random
    import sqlite3
    import threading
    import time as _time
    from pathlib import Path

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from bird.data import BirdExample
    from bird.prompts import build_messages, extract_sql
    from bird.schema import extract_schema

    # ---- 0. Smoke override ----
    if smoke:
        num_examples = 20
        max_steps = 2
        # gen_batch_size = per_device_bs * world * grad_accum must be divisible
        # by num_generations. With per_device_bs=1, world=1, grad_accum=4 → 4,
        # divisible by num_generations=4.
        gradient_accumulation_steps = 4
        print("[smoke] num_examples=20, max_steps=2, grad_accum=4")

    sft_checkpoints.reload()
    bird_data.reload()

    # ---- 1. Tokenizer + chat template ----
    print(f"[init] loading tokenizer {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_HOME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. Build BIRD train dataset ----
    train_root = Path(BIRD_ROOT) / "train"
    with open(train_root / "train.json") as f:
        all_tasks = json.load(f)
    train_db_dir = train_root / "train_databases"

    print(f"[data] BIRD train: {len(all_tasks)} tasks; sampling {num_examples}")
    random.seed(42)
    # Deterministic sample so smoke and full runs are reproducible.
    random.shuffle(all_tasks)

    # Profile schemas as we build prompts. Cap by num_examples * a buffer
    # since some tasks may be filtered for prompt length.
    schema_cache: dict[str, object] = {}
    rows: list[dict] = []
    for idx, task in enumerate(all_tasks):
        if len(rows) >= num_examples:
            break
        db_id = task["db_id"]
        db_path = train_db_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            continue
        if db_id not in schema_cache:
            try:
                schema_cache[db_id] = extract_schema(db_path, db_id, n_samples=3)
            except Exception as e:
                print(f"[data] skip db_id={db_id}: {e}")
                continue
        ex = BirdExample(
            question_id=task.get("question_id", idx),
            db_id=db_id,
            question=task["question"],
            evidence=task.get("evidence", task.get("hint", "")) or "",
            sql=task.get("SQL", ""),
            difficulty=task.get("difficulty"),
        )
        msgs = build_messages(ex, schema_cache[db_id], n_samples=3)
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        # Filter overlong prompts. Cheap token-length sanity check.
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_ids) > max_prompt_length:
            continue
        rows.append({
            "prompt": prompt_text,
            "question_id": int(ex.question_id) if ex.question_id is not None else idx,
            "db_id": db_id,
            "gold_sql": ex.sql,
            "db_path": str(db_path),
        })

    print(f"[data] built {len(rows)} training examples")
    if len(rows) < 4:
        raise RuntimeError(f"Too few training examples ({len(rows)}); aborting.")

    train_ds = Dataset.from_list(rows)

    # ---- 3. Reward function (with gold-execution cache) ----
    # The gold cache lives in the closure. Each unique question_id's gold
    # output is computed once and reused for every rollout (num_generations
    # rollouts per prompt per step + revisits across epochs).
    gold_cache: dict[int, tuple[str, list[tuple]] | tuple[str, None]] = {}
    reward_stats = {"calls": 0, "positives": 0, "errors": 0, "empties": 0}

    def _execute(db_path: str, sql: str, timeout_s: float = 5.0) -> list[tuple]:
        """Same as bird.eval._execute but inlined to avoid a second import."""
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout_s)
        conn.text_factory = lambda b: (
            b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b
        )
        timer = threading.Timer(timeout_s, conn.interrupt)
        timer.daemon = True
        timer.start()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            return cur.fetchall()
        finally:
            timer.cancel()
            conn.close()

    def _rows_equal(a, b) -> bool:
        if len(a) != len(b):
            return False
        try:
            return set(a) == set(b)
        except TypeError:
            from collections import Counter
            return Counter(map(repr, a)) == Counter(map(repr, b))

    def _gold_rows(question_id: int, db_path: str, gold_sql: str):
        """Cached gold-row result for a question. Returns None if gold fails."""
        if question_id in gold_cache:
            return gold_cache[question_id][1]
        try:
            rows = _execute(db_path, gold_sql, timeout_s=5.0)
            gold_cache[question_id] = ("ok", rows)
            return rows
        except Exception as e:
            # Mark this question as un-rewardable (gold itself fails). Cache the
            # negative so we don't keep paying the SQL cost.
            gold_cache[question_id] = (f"gold_error: {e}", None)
            return None

    def reward_exec(prompts, completions, question_id=None, db_id=None,
                    gold_sql=None, db_path=None, **kwargs) -> list[float]:
        """Binary execution-accuracy reward.

        TRL passes prompts and completions as parallel lists, plus per-row
        dataset columns as keyword args (also as lists, one entry per
        completion). Returns one float per completion.
        """
        n = len(completions)
        rewards: list[float] = []
        for i in range(n):
            text = completions[i]
            # Completions can be either a raw string or a chat-list of dicts.
            if isinstance(text, list):
                text = "".join(seg.get("content", "") for seg in text)
            qid = question_id[i] if question_id is not None else -1
            dbp = db_path[i] if db_path is not None else ""
            gold = gold_sql[i] if gold_sql is not None else ""

            reward_stats["calls"] += 1
            sql = extract_sql(text)
            if not sql.strip() or sql.strip() == ";":
                reward_stats["empties"] += 1
                rewards.append(0.0)
                continue

            gold_rows = _gold_rows(qid, dbp, gold)
            if gold_rows is None:
                # Gold itself broke — neutral 0 reward (can't penalize).
                rewards.append(0.0)
                continue

            try:
                pred_rows = _execute(dbp, sql, timeout_s=5.0)
            except Exception:
                reward_stats["errors"] += 1
                rewards.append(0.0)
                continue

            if _rows_equal(pred_rows, gold_rows):
                reward_stats["positives"] += 1
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    # ---- 4. Load model + apply LoRA ----
    print(f"[model] loading {MODEL_ID} in bf16")
    t0 = _time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=HF_HOME,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    print(f"[model] loaded in {_time.time() - t0:.1f}s")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---- 5. GRPO config ----
    # NOTE: recent TRL (>=0.20) dropped `max_prompt_length` from GRPOConfig.
    # We enforce the prompt cap upstream when building the dataset (lines above).
    grpo_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        logging_steps=1,
        save_steps=max(max_steps // 4, 1),
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        temperature=temperature,
        top_p=1.0,
        beta=0.04,                  # KL coefficient against reference (the base)
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,  # IMPORTANT: keep question_id/db_path/gold_sql for reward fn
        report_to=[],                  # no wandb
        seed=42,
    )

    # ---- 6. Build trainer ----
    print("[trainer] building GRPOTrainer")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_ds,
        reward_funcs=reward_exec,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # ---- 7. Train ----
    print("[train] starting GRPO")
    t_start = _time.time()
    trainer.train()
    elapsed = _time.time() - t_start
    print(f"[train] done in {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # ---- 8. Save adapter ----
    final_dir = os.path.join(output_dir, "final")
    print(f"[save] saving LoRA adapter to {final_dir}")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    sft_checkpoints.commit()

    # ---- 9. Dump reward stats ----
    pos = reward_stats["positives"]
    calls = reward_stats["calls"]
    print("=" * 60)
    print(f"REWARD STATS: total_rollouts={calls}, positives={pos} "
          f"({100*pos/max(calls,1):.1f}%), exec_errors={reward_stats['errors']}, "
          f"empty_sql={reward_stats['empties']}")
    print(f"Training elapsed: {elapsed/60:.1f}m")
    print(f"Adapter saved to: {final_dir}")
    print("=" * 60)

    return {
        "output_dir": final_dir,
        "elapsed_s": elapsed,
        "max_steps": max_steps,
        "reward_calls": calls,
        "reward_positives": pos,
        "reward_positive_rate": pos / max(calls, 1),
    }


# ============================================================
# Eval (merge LoRA + vLLM)
# ============================================================

@app.function(
    image=rl_image,
    volumes={
        BIRD_ROOT: bird_data,
        HF_HOME: hf_cache,
        CKPT_ROOT: sft_checkpoints,
        RESULTS_ROOT: results_vol,
    },
    gpu="B200",
    timeout=60 * 60,
)
def eval_rl_adapter(
    adapter_path: str = "/checkpoints/rl_7b_grpo/final",
    split: str = "dev",
    limit: int = 0,
    n_samples: int = 3,
    max_tokens: int = 1024,
    save_as: str = "rl-qwen2.5-coder-7b-grpo-dev.json",
):
    """Merge the LoRA adapter into the base, serve via vLLM, eval on BIRD dev.

    Merging is one-shot (~30s) and lets vLLM treat the result as a single
    HF checkpoint — much simpler than runtime adapter switching.
    """
    import json
    import time as _time
    from pathlib import Path

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bird.data import load_split
    from bird.prompts import build_messages, extract_sql
    from bird.schema import extract_schema

    sft_checkpoints.reload()
    bird_data.reload()

    merged_dir = adapter_path.rstrip("/") + "_merged"
    if not Path(merged_dir).exists() or not (Path(merged_dir) / "config.json").exists():
        print(f"[merge] loading base + adapter from {adapter_path}, merging to {merged_dir}")
        t0 = _time.time()
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=HF_HOME,
        )
        peft_model = PeftModel.from_pretrained(base, adapter_path)
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_HOME)
        tok.save_pretrained(merged_dir)
        sft_checkpoints.commit()
        print(f"[merge] done in {_time.time() - t0:.1f}s")
        # Drop the HF model from GPU before bringing up vLLM.
        del merged, peft_model, base
        torch.cuda.empty_cache()
    else:
        print(f"[merge] reusing existing merged dir {merged_dir}")

    # Build prompts.
    from bird.inference import GenConfig, VLLMEngine
    sp = load_split(Path(BIRD_ROOT) / split, name=split)
    examples = sp.examples[:limit] if limit else sp.examples
    print(f"[eval] {len(examples)} {split} examples")

    schema_cache: dict[str, object] = {}
    convos: list[list[dict]] = []
    metas: list[dict] = []
    for ex in examples:
        if ex.db_id not in schema_cache:
            schema_cache[ex.db_id] = extract_schema(sp.db_path(ex.db_id), ex.db_id, n_samples=n_samples)
        convos.append(build_messages(ex, schema_cache[ex.db_id], n_samples=n_samples))
        metas.append({
            "question_id": ex.question_id,
            "db_id": ex.db_id,
            "difficulty": ex.difficulty,
            "gold_sql": ex.sql,
        })

    print(f"[eval] loading vLLM from merged dir {merged_dir}")
    engine = VLLMEngine(model=merged_dir, max_model_len=16384, download_dir=HF_HOME)
    outs = engine.chat(convos, GenConfig(n=1, temperature=0.0, max_tokens=max_tokens))

    predictions = []
    for meta, out in zip(metas, outs):
        text = out.texts[0] if out.texts else ""
        predictions.append({
            "question_id": meta["question_id"],
            "db_id": meta["db_id"],
            "difficulty": meta["difficulty"],
            "gold_sql": meta["gold_sql"],
            "predicted_sql": extract_sql(text),
            "raw_completion": text,
        })

    # Evaluate inline (don't use the modal_app eval function — different image).
    from bird.eval import evaluate_predictions as _eval_pred
    from bird.eval import format_summary, make_pred_item, summarize

    split_root = Path(BIRD_ROOT) / split
    db_dir = split_root / f"{split}_databases"
    items = [
        make_pred_item(
            question_id=p["question_id"],
            db_id=p["db_id"],
            db_path=db_dir / p["db_id"] / f'{p["db_id"]}.sqlite',
            predicted_sql=p["predicted_sql"],
            gold_sql=p["gold_sql"],
            difficulty=p.get("difficulty"),
            timeout_s=30.0,
        )
        for p in predictions
    ]
    t0 = _time.time()
    # Workers=1 inside Modal to avoid spawn-fork pool issues with vLLM-loaded process.
    results = _eval_pred(items, workers=1)
    summary = summarize(results)
    print(format_summary(summary))
    print(f"[eval] scored {len(results)} in {_time.time() - t0:.1f}s")

    payload = {
        "split": split,
        "n": summary.n,
        "n_correct": summary.n_correct,
        "ex": summary.ex,
        "by_status": summary.by_status,
        "by_difficulty": summary.by_difficulty,
        "results": [
            {
                **predictions[i],
                "question_id": r.question_id,
                "db_id": r.db_id,
                "difficulty": r.difficulty,
                "status": r.status.value,
                "error": r.error,
                "predicted_sql": r.predicted_sql,
                "gold_sql": r.gold_sql,
            }
            for i, r in enumerate(results)
        ],
    }

    out_path = Path(RESULTS_ROOT) / save_as
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    results_vol.commit()
    print(f"[eval] wrote {out_path}")

    return {
        "split": split,
        "n": summary.n,
        "n_correct": summary.n_correct,
        "ex": summary.ex,
        "saved": save_as,
    }


# ============================================================
# Local entrypoints
# ============================================================

@app.local_entrypoint()
def main(
    num_examples: int = 800,
    max_steps: int = 100,
    learning_rate: float = 1e-5,
    num_generations: int = 4,
    smoke: bool = False,
    output_dir: str = RL_OUTPUT_DIR,
):
    """Spawn GRPO training. `--smoke True` runs the 2-step / 20-q validation."""
    tag = "SMOKE" if smoke else "FULL"
    print(f"=== GRPO RL of {MODEL_ID} on BIRD ({tag}) ===")
    print(f"  num_examples={num_examples}  max_steps={max_steps}  lr={learning_rate}")
    print(f"  num_generations={num_generations}  output_dir={output_dir}")

    fc = train_rl_7b.spawn(
        num_examples=num_examples,
        max_steps=max_steps,
        learning_rate=learning_rate,
        num_generations=num_generations,
        output_dir=output_dir,
        smoke=smoke,
    )
    print(f"\nSpawned function call: {fc.object_id}")
    print(f"Tail logs:        modal app logs {APP_NAME}")
    print(f"Or fetch result:  modal.FunctionCall.from_id('{fc.object_id}').get()")


@app.local_entrypoint()
def eval_rl(
    adapter_path: str = "/checkpoints/rl_7b_grpo/final",
    split: str = "dev",
    limit: int = 0,
    save_as: str = "rl-qwen2.5-coder-7b-grpo-dev.json",
):
    """Eval a saved adapter on BIRD dev (blocking call)."""
    print(f"=== Eval RL adapter {adapter_path} on BIRD {split} ===")
    print(f"  limit={limit or 'all'}  save_as={save_as}")
    result = eval_rl_adapter.remote(
        adapter_path=adapter_path,
        split=split,
        limit=limit,
        save_as=save_as,
    )
    import json
    print(json.dumps(result, indent=2))
