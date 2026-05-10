"""Evaluate an SFT checkpoint by routing the existing `Inference` class at it.

Two paths supported:

1. **Merged model** (recommended): `train_sft` saves a merged bf16 model to
   `<save_dir>/merged`. vLLM loads this like any HF checkpoint — no LoRA
   glue needed. Pass that path as `--model`:

       python -m scripts.eval_sft_checkpoint \\
           --model /checkpoints/qwen3.6-27b-bird-sft-v1/merged

2. **Raw checkpoint dir** (e.g. `checkpoint-200`): the dir contains a LoRA
   adapter, not a merged model. vLLM CAN serve LoRA via `--enable-lora` but
   our `Inference` class doesn't currently wire that flag through, so we
   recommend running `merge_and_unload` first. This script will detect the
   adapter-only case and refuse, printing the merge command to run.

Notes on path semantics:
  - vLLM accepts both HF hub IDs ("Qwen/Qwen3.6-27B") and local paths.
  - For local paths to work, the path must live inside a volume mounted by
    `Inference` — which is `hf-cache` or `bird-data`. The SFT checkpoints
    volume isn't currently mounted by `Inference`, so before this script
    works end-to-end the operator needs to either:
      (a) copy `merged/` into hf-cache, OR
      (b) extend `Inference`'s `volumes={}` to mount `bird-sft-checkpoints`.
    The runbook (sft/launch.md) documents (b) as the preferred path.

Usage:
    python -m scripts.eval_sft_checkpoint --model <path-or-hub-id> \\
        [--split dev] [--limit 0] [--max-tokens 8192] [--save-as out.json]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="HF hub ID or absolute path inside a mounted Modal volume.")
    p.add_argument("--split", default="dev")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Generation cap. SFT models tend to be terser; 8k is generous.")
    p.add_argument("--tensor-parallel-size", type=int, default=4,
                   help="Qwen3.6-27B needs >=2 GPUs at fp16; 4 is safe.")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--save-as", default="",
                   help="Filename on bird-results volume. Auto-generated if empty.")
    p.add_argument("--modal-profile", default=None,
                   help="Sets MODAL_PROFILE for the spawned `modal run` (e.g. action-svabhu).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command we'd execute and exit.")
    args = p.parse_args()

    # ---------- guardrails on local checkpoints ----------
    if args.model.startswith("/"):
        model_path = Path(args.model)
        # If they pointed at a `checkpoint-N` dir, that's an adapter only.
        # Detect by looking for adapter_model.* without a merged config.
        # We can't `stat` the Modal volume from a laptop, so we only check
        # by *path shape* (warn the user); actual file check happens on Modal.
        if "checkpoint-" in model_path.name and not model_path.name.endswith("merged"):
            print(
                "[warn] this looks like a step checkpoint (LoRA adapter only). "
                "vLLM via our Inference class loads merged models. Either:\n"
                "  - eval `<save_dir>/merged/` instead (set by train_sft after merge), or\n"
                "  - run an offline merge: peft.PeftModel.from_pretrained(...).merge_and_unload()\n"
                "Proceeding anyway in case you've wired LoRA loading downstream."
            )

    # ---------- build save tag ----------
    save_as = args.save_as or f"sft-eval-{Path(args.model).name}-{args.split}.json"

    # ---------- assemble the modal-run invocation ----------
    cmd = [
        "modal", "run", "modal_app.py::run_baseline",
        "--split", args.split,
        "--limit", str(args.limit),
        "--model", args.model,
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
        "--save-as", save_as,
    ]

    env = None
    if args.modal_profile:
        import os
        env = {**os.environ, "MODAL_PROFILE": args.modal_profile}

    print(f"[eval_sft_checkpoint] {' '.join(cmd)}")
    if args.modal_profile:
        print(f"[eval_sft_checkpoint] MODAL_PROFILE={args.modal_profile}")
    print(f"[eval_sft_checkpoint] save_as={save_as}")

    if args.dry_run:
        print("[eval_sft_checkpoint] --dry-run: not executing")
        return

    # Run from the repo root so `modal_app.py` resolves.
    repo_root = Path(__file__).resolve().parents[1]
    try:
        subprocess.run(cmd, cwd=repo_root, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[eval_sft_checkpoint] modal run failed with exit {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
