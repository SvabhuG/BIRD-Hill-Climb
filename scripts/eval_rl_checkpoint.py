"""Eval an arbitrary HF model path or local checkpoint dir on BIRD dev.

Wraps our existing `modal_app.py::run_baseline` entrypoint so RL checkpoints
get scored under exactly the same prompting + extraction + execution-accuracy
pipeline as the 22 baselines on the leaderboard. No prompt drift, no parser
drift — directly comparable EX numbers.

Usage
-----
    # Hugging Face hub:
    python -m scripts.eval_rl_checkpoint Snowflake/Arctic-Text2SQL-R1-32B

    # Local path (e.g. a fresh prime-rl checkpoint on a shared filesystem
    # that Modal can pull from after a quick `hf upload`):
    python -m scripts.eval_rl_checkpoint <org>/bird-qwen3-coder-step-3000 --split dev

    # With explicit sizing for a 30B-class model:
    python -m scripts.eval_rl_checkpoint <path> --tensor-parallel-size 4 --max-model-len 16384

Implementation note
-------------------
We invoke `modal run modal_app.py::run_baseline` as a subprocess so this
script stays a thin wrapper around the already-working pipeline. The
alternative — duplicating the orchestration in-process — would create a
second source of truth for "how we eval", which is the bug we're trying
to avoid.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    """Repo root = the directory containing modal_app.py."""
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "modal_app.py").exists():
            return p
    raise SystemExit(
        f"Could not locate modal_app.py from {here}; run this script from the repo root."
    )


def _looks_like_local_path(s: str) -> bool:
    # Heuristic: starts with '/' or './', or exists on disk.
    return s.startswith("/") or s.startswith("./") or Path(s).exists()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "model",
        help="HuggingFace repo id (e.g. Snowflake/Arctic-Text2SQL-R1-32B) "
        "or local checkpoint directory (must be reachable from the GPU container).",
    )
    p.add_argument("--split", default="dev", choices=["dev", "train"], help="BIRD split to eval on.")
    p.add_argument("--limit", type=int, default=0, help="Only eval first N examples (0 = all).")
    p.add_argument("--n-samples", type=int, default=3, help="Sample rows per table in the schema prompt.")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--base-model", action="store_true", help="Use raw-completion path (no chat template).")
    p.add_argument(
        "--save-as",
        default="",
        help="Filename on the bird-results volume (default auto-generated).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the modal command without invoking it.",
    )
    args = p.parse_args(argv)

    repo = _repo_root()
    is_local = _looks_like_local_path(args.model)
    if is_local:
        local = Path(args.model).resolve()
        if not local.exists():
            print(f"[err] local checkpoint path does not exist: {local}", file=sys.stderr)
            return 2
        # Heads-up: Modal can only see paths inside its containers / volumes.
        # The caller has to have already uploaded the ckpt to a place vLLM
        # will load from. Two supported patterns:
        #   1. Push to HF Hub first (`hf upload <repo> <local-path>`) — then
        #      re-run this script with the hub repo id.
        #   2. Mount the ckpt onto an HF cache volume Modal already mounts at
        #      $HF_HOME. The path needs to be visible inside the container.
        print(
            f"[warn] local path passed: {local}\n"
            f"       Modal containers only see paths inside mounted volumes.\n"
            f"       If this isn't on the hf-cache volume, push to HF Hub first.",
            file=sys.stderr,
        )

    # Default save filename encodes model + split + timestamp so checkpoint
    # learning-curve scans are easy to grep.
    save_as = args.save_as or f"rl-eval-{Path(args.model).name}-{args.split}-{int(time.time())}.json"

    cmd = [
        "modal", "run", str(repo / "modal_app.py") + "::run_baseline",
        "--split", args.split,
        "--limit", str(args.limit),
        "--model", args.model,
        "--n-samples", str(args.n_samples),
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--save-as", save_as,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
    ]
    if args.base_model:
        cmd.append("--base-model")

    print(f"[eval_rl_checkpoint] launching: {' '.join(cmd)}")
    if args.dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=repo)
    if proc.returncode != 0:
        print(f"[eval_rl_checkpoint] modal run exited non-zero ({proc.returncode})", file=sys.stderr)
        return proc.returncode
    print(f"[eval_rl_checkpoint] results saved as {save_as} on the bird-results volume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
