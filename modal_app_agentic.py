"""Modal app: agentic BIRD solver (feat/agentic-explore).

Reuses the bird-data, hf-cache, bird-results volumes from the main app.
Adds an `Agent` GPU class that runs a per-question tool-using loop on Qwen3-Coder-MoE.

Usage:
    modal run modal_app_agentic.py::run_agentic --split dev --limit 50
"""
import json
import time
from pathlib import Path

import modal

APP_NAME = "bird-climb-agentic"

bird_data = modal.Volume.from_name("bird-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("bird-results", create_if_missing=True)

BIRD_ROOT = "/data/bird"
HF_HOME = "/root/.cache/hf"
RESULTS_ROOT = "/results"

_PY = "3.11"

cpu_image = (
    modal.Image.debian_slim(python_version=_PY)
    .apt_install("unzip", "wget", "curl")
    .pip_install("tqdm", "pydantic>=2", "requests", "sqlglot>=25")
    .add_local_python_source("bird")
)

gpu_image = (
    modal.Image.debian_slim(python_version=_PY)
    .apt_install("git")
    .pip_install(
        "vllm==0.11.0",
        "transformers==4.57.0",
        "tqdm",
        "pydantic>=2",
        "sqlglot>=25",
        "huggingface_hub",
    )
    .env({"HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install("hf_transfer")
    .add_local_python_source("bird")
)

# Q3.6-27B has a hybrid Gated DeltaNet architecture (model type `qwen3_5`) that
# the standard image's transformers==4.57.0 doesn't recognize. The known-working
# pin is vllm 0.20.2 + transformers 5.8.0 + FLASH_ATTN backend, with DeepGEMM
# auto-selection disabled at runtime via env var.
#
# Tried vllm 0.19.0 (pre-DeepGEMM auto-selection) but it pins transformers<5
# and its huggingface_hub 1.14.0 removes `is_offline_mode` which transformers
# 5.8.0 still imports — irreconcilable.
#
# Three-part fix for vllm 0.20.2's DeepGEMM crash on B200/Qwen3.6:
#   1. drop the `deep_gemm` pip dep (source-only, can't pip-build)
#   2. set VLLM_USE_DEEP_GEMM=0 + VLLM_ATTENTION_BACKEND=FLASH_ATTN env vars
#   3. pass max_num_batched_tokens=2096 at the LLM ctor — required for GDN
#      cache alignment; vLLM's default 8192 silently breaks Q3.5/3.6.
#
# `qwen3_5`'s Gated DeltaNet requires nvcc at runtime for CUDAGraph capture, so
# we base on the CUDA-devel image (which ships /usr/local/cuda + nvcc) instead
# of debian_slim.
gpu_image_q36 = (
    modal.Image.from_registry(
        # CUDA 12.8 for B200/Blackwell (compute_100a). 12.4 fails with
        # "Unsupported gpu architecture 'compute_100a'" when flashinfer
        # JIT-compiles fmha kernels at warmup.
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python=_PY,
    )
    .apt_install("git")
    # vllm 0.19.0 pins transformers<5, but transformers 5.8.0 is the only
    # version recognizing the `qwen3_5` model_type for Q3.6. The two libs'
    # huggingface_hub expectations diverge (vllm 0.19 + hf_hub 1.x removes
    # `is_offline_mode`, which transformers 5.8.0 still imports). So we
    # use vllm==0.20.2 here (compatible w/ transformers 5.8.0) and instead
    # disable DeepGEMM at *runtime* via env vars + force FLASH_ATTN, plus
    # set max_num_batched_tokens=2096 at the LLM ctor for GDN alignment.
    .pip_install(
        "vllm==0.20.2",
        "transformers==5.8.0",
        "tqdm",
        "pydantic>=2",
        "sqlglot>=25",
        "huggingface_hub",
    )
    .env({
        "HF_HOME": HF_HOME,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Disable DeepGEMM auto-selection + force FLASH_ATTN backend for Q3.6.
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    })
    .pip_install("hf_transfer")
    .add_local_python_source("bird")
)

app = modal.App(APP_NAME)


@app.cls(
    image=gpu_image,
    gpu="B200",
    volumes={HF_HOME: hf_cache, BIRD_ROOT: bird_data, RESULTS_ROOT: results_vol},
    timeout=2 * 60 * 60,
    scaledown_window=300,
)
class Agent:
    """vLLM engine + per-question agent driver. The whole loop lives on the GPU
    container so each LLM call doesn't pay Modal RPC overhead and SQLite reads
    happen against the locally-mounted bird-data volume.
    """

    model_name: str = modal.parameter(default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    tensor_parallel_size: int = modal.parameter(default=1)
    max_model_len: int = modal.parameter(default=16384)
    max_turns: int = modal.parameter(default=6)
    # NOTE: modal.parameter only allows int|str|bytes|bool — encode floats as
    # *_x100 ints and divide on read.
    tool_timeout_cs: int = modal.parameter(default=1000)   # centiseconds; 1000 = 10.0s
    n_samples: int = modal.parameter(default=3)
    max_tokens: int = modal.parameter(default=1024)
    temperature_x100: int = modal.parameter(default=0)     # 0 == greedy
    # "v1" -> 2-tool / no-baseline-anchor recipe; "v2" -> 3-tool w/ keep_baseline_sql.
    agentic_version: str = modal.parameter(default="v2")

    @modal.enter()
    def _load(self):
        from bird.inference import VLLMEngine

        t0 = time.time()
        self.engine = VLLMEngine(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            download_dir=HF_HOME,
        )
        print(f"[agent] loaded {self.model_name} in {time.time() - t0:.1f}s")

    @modal.method()
    def solve_batch(
        self,
        split: str,
        limit: int,
        save_as: str,
        question_ids: list[int] | None = None,
        baseline_sql_by_qid: dict[str, str] | None = None,
    ) -> dict:
        """Run the agent on `limit` BIRD examples and return predictions + traces.

        If `question_ids` is provided, run on exactly that set (ignores `limit`).
        `baseline_sql_by_qid` maps str(qid) -> baseline greedy SQL; passed through
        to the agent so it can anchor on / fall back to the baseline.
        """
        from bird.agentic import build_initial_messages, step_agent
        from bird.data import load_split
        from bird.inference import GenConfig
        from bird.schema import extract_schema

        sp = load_split(Path(BIRD_ROOT) / split, name=split)
        if question_ids is not None:
            wanted = set(int(q) for q in question_ids)
            examples = [ex for ex in sp.examples if ex.question_id in wanted]
            missing = wanted - {ex.question_id for ex in examples}
            if missing:
                print(f"[agent] WARNING: {len(missing)} requested qids not in split (sample: {sorted(missing)[:5]})")
            print(f"[agent] subset mode: routing {len(examples)}/{len(wanted)} questions")
        else:
            examples = sp.examples[:limit] if limit else sp.examples

        temperature = self.temperature_x100 / 100.0
        tool_timeout_s = self.tool_timeout_cs / 100.0
        cfg = GenConfig(n=1, temperature=temperature, top_p=1.0, max_tokens=self.max_tokens)
        baseline_lookup: dict[int, str] = {}
        if baseline_sql_by_qid:
            for k, v in baseline_sql_by_qid.items():
                try:
                    baseline_lookup[int(k)] = v or ""
                except (TypeError, ValueError):
                    continue
        if baseline_lookup:
            print(f"[agent] baseline SQL available for {len(baseline_lookup)} qids")

        def chat_fn(messages):
            outs = self.engine.chat([messages], cfg)
            return outs[0].texts[0] if outs and outs[0].texts else ""

        schema_cache: dict = {}
        predictions: list[dict] = []
        traces: list[dict] = []

        for i, ex in enumerate(examples):
            if ex.db_id not in schema_cache:
                schema_cache[ex.db_id] = extract_schema(sp.db_path(ex.db_id), ex.db_id, n_samples=self.n_samples)
            schema = schema_cache[ex.db_id]
            db_path = sp.db_path(ex.db_id)
            baseline_sql = baseline_lookup.get(ex.question_id, "")

            t0 = time.time()
            msgs = build_initial_messages(
                ex, schema, n_samples=self.n_samples, baseline_sql=baseline_sql,
                agentic_version=self.agentic_version,
            )
            try:
                trace = step_agent(
                    msgs,
                    db_path=db_path,
                    max_turns=self.max_turns,
                    tool_timeout_s=tool_timeout_s,
                    chat_fn=chat_fn,
                    baseline_sql=baseline_sql,
                    agentic_version=self.agentic_version,
                )
            except Exception as e:
                print(f"[agent] q{ex.question_id}: trace crashed: {e!r}")
                trace = None

            elapsed = time.time() - t0
            if trace is None:
                predictions.append({
                    "question_id": ex.question_id,
                    "db_id": ex.db_id,
                    "difficulty": ex.difficulty,
                    "gold_sql": ex.sql,
                    "predicted_sql": "",
                    "raw_completion": "",
                })
                traces.append({
                    "question_id": ex.question_id, "turns": 0, "n_tool_calls": 0,
                    "n_exec_errors": 0, "completion_chars": 0, "finish_reason": "crash",
                    "elapsed_s": elapsed,
                })
                continue

            predictions.append({
                "question_id": ex.question_id,
                "db_id": ex.db_id,
                "difficulty": ex.difficulty,
                "gold_sql": ex.sql,
                "predicted_sql": trace.final_sql,
                "raw_completion": "",  # available in trace.history if needed
            })
            traces.append({
                "question_id": ex.question_id,
                "db_id": ex.db_id,
                "difficulty": ex.difficulty,
                "turns": trace.turns,
                "n_tool_calls": trace.n_tool_calls,
                "n_exec_errors": trace.n_exec_errors,
                "completion_chars": trace.completion_chars,
                "finish_reason": trace.finish_reason,
                "elapsed_s": elapsed,
                "history": trace.history,
            })
            if (i + 1) % 10 == 0 or i == len(examples) - 1:
                print(f"[agent] {i + 1}/{len(examples)}  q{ex.question_id} "
                      f"turns={trace.turns} calls={trace.n_tool_calls} "
                      f"finish={trace.finish_reason} t={elapsed:.1f}s")

        # Persist traces alongside predictions for later inspection.
        if save_as:
            out = Path(RESULTS_ROOT) / save_as.replace(".json", ".traces.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(traces, indent=2, default=str))
            results_vol.commit()
            print(f"[agent] wrote traces to {out}")

        # Aggregate stats from traces.
        n = len(traces)
        avg_turns = sum(t["turns"] for t in traces) / max(n, 1)
        avg_calls = sum(t["n_tool_calls"] for t in traces) / max(n, 1)
        avg_errors = sum(t["n_exec_errors"] for t in traces) / max(n, 1)
        avg_chars = sum(t["completion_chars"] for t in traces) / max(n, 1)
        finish_counts: dict[str, int] = {}
        for t in traces:
            finish_counts[t["finish_reason"]] = finish_counts.get(t["finish_reason"], 0) + 1

        return {
            "predictions": predictions,
            "stats": {
                "n": n,
                "avg_turns": avg_turns,
                "avg_tool_calls": avg_calls,
                "avg_exec_errors": avg_errors,
                "avg_completion_chars": avg_chars,
                "finish_reason_counts": finish_counts,
            },
        }


@app.function(
    image=cpu_image,
    volumes={BIRD_ROOT: bird_data, RESULTS_ROOT: results_vol},
    cpu=8,
    timeout=60 * 30,
)
def evaluate_predictions(
    split: str,
    predictions: list[dict],
    timeout_s: float = 30.0,
    workers: int = 8,
    save_as: str | None = None,
) -> dict:
    """Same as the main app's evaluator — duplicated here so this app is self-contained."""
    from bird.eval import (
        evaluate_predictions as _eval_pred,
        format_summary,
        make_pred_item,
        summarize,
    )

    split_root = Path(BIRD_ROOT) / split
    db_dir = split_root / f"{split}_databases"

    items = [
        make_pred_item(
            question_id=p["question_id"],
            db_id=p["db_id"],
            db_path=db_dir / p["db_id"] / f'{p["db_id"]}.sqlite',
            predicted_sql=p.get("predicted_sql", ""),
            gold_sql=p.get("gold_sql", ""),
            difficulty=p.get("difficulty"),
            timeout_s=timeout_s,
        )
        for p in predictions
    ]

    t0 = time.time()
    results = _eval_pred(items, workers=workers)
    summary = summarize(results)
    print(format_summary(summary))
    print(f"[eval] scored {len(results)} in {time.time() - t0:.1f}s")

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

    if save_as:
        out = Path(RESULTS_ROOT) / save_as
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        results_vol.commit()
        print(f"[eval] wrote {out}")

    return {
        "split": split, "n": summary.n, "n_correct": summary.n_correct,
        "ex": summary.ex, "by_status": summary.by_status,
        "by_difficulty": summary.by_difficulty, "saved": save_as,
    }


@app.function(
    image=cpu_image,
    volumes={BIRD_ROOT: bird_data, RESULTS_ROOT: results_vol},
    cpu=8,
    timeout=60 * 30,
)
def classify_degenerate(
    split: str,
    baseline_filename: str,
    save_as: str,
    timeout_s: float = 10.0,
) -> dict:
    """Execute every baseline predicted SQL against its DB and classify whether
    the result is "degenerate" (empty, all-NULL, or single zero/NULL row).

    Returns and saves `{question_id: bool}` plus per-question status. Skips
    exec_error/timeout rows — those are caught by rule 1.
    """
    import sqlite3
    import threading

    split_root = Path(BIRD_ROOT) / split
    db_dir = split_root / f"{split}_databases"
    baseline_path = Path(RESULTS_ROOT) / baseline_filename
    baseline = json.loads(baseline_path.read_text())

    def _exec_ro(db_path, sql, t_s):
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=t_s)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b
        timer = threading.Timer(t_s, conn.interrupt)
        timer.daemon = True
        timer.start()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            return cur.fetchmany(8)  # we only need to inspect the first few rows
        finally:
            timer.cancel()
            conn.close()

    def _is_degen(rows):
        if not rows:
            return True
        if all(all(c is None for c in row) for row in rows):
            return True
        if len(rows) == 1:
            row = rows[0]
            ok = True
            for c in row:
                if c is None:
                    continue
                if isinstance(c, (int, float)) and c == 0:
                    continue
                if isinstance(c, str) and c.strip() == "":
                    continue
                ok = False
                break
            if ok:
                return True
        return False

    flags: dict[int, bool] = {}
    n_checked = n_degen = n_skipped = 0
    for r in baseline["results"]:
        qid = int(r["question_id"])
        status = r.get("status")
        if status in ("exec_error", "timeout", "empty"):
            # Caught by rule 1 already; don't waste cycles re-executing.
            n_skipped += 1
            continue
        sql = r.get("predicted_sql") or ""
        if not sql.strip():
            n_skipped += 1
            continue
        db_path = db_dir / r["db_id"] / f'{r["db_id"]}.sqlite'
        try:
            rows = _exec_ro(db_path, sql, timeout_s)
        except Exception:
            n_skipped += 1
            continue
        is_d = _is_degen(rows)
        flags[qid] = is_d
        n_checked += 1
        if is_d:
            n_degen += 1
        if n_checked % 200 == 0:
            print(f"[degen] {n_checked} checked, {n_degen} degenerate so far")

    payload = {str(k): v for k, v in flags.items()}
    out = Path(RESULTS_ROOT) / save_as
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    results_vol.commit()
    print(f"[degen] DONE: {n_checked} executed, {n_degen} degenerate, {n_skipped} skipped")
    print(f"[degen] wrote {out}")
    return {"n_checked": n_checked, "n_degenerate": n_degen, "n_skipped": n_skipped, "saved": save_as}


@app.local_entrypoint()
def run_classify_degenerate(
    split: str = "dev",
    baseline_filename: str = "baseline-qwen3-coder-30b-a3b-instruct-dev-full.json",
    save_as: str = "degenerate-flags-qwen3-coder-30b-a3b-instruct-dev.json",
):
    """One-shot helper to write a {qid: bool} degenerate-flag file to bird-results."""
    summary = classify_degenerate.remote(
        split=split, baseline_filename=baseline_filename, save_as=save_as,
    )
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def run_agentic_routed(
    split: str = "dev",
    routing_from: str = "results/routing_set.json",
    baseline_filename: str = "baseline-qwen3-coder-30b-a3b-instruct-dev-full.json",
    model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    max_turns: int = 6,
    tool_timeout_s: float = 10.0,
    n_samples: int = 3,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    save_as: str = "",
    tensor_parallel_size: int = 1,
    max_model_len: int = 16384,
    agentic_version: str = "v2",
):
    """Run the agent ONLY on `routing_from`'s question_ids, then merge with
    `baseline_filename`'s predictions for the unrouted questions and score
    the merged full-dev predictions end-to-end.
    """
    routing_payload = json.loads(Path(routing_from).read_text())
    qids: list[int] = list(routing_payload["question_ids"])
    if not qids:
        raise SystemExit("routing set is empty; nothing to do")
    if len(qids) < 50:
        raise SystemExit(f"routing set too small ({len(qids)} < 50); aborting per spec")

    tag = save_as or f"agentic-routed-{Path(model).name}-{split}-{len(qids)}-{int(time.time())}.json"
    print(f"[routed] running agent on {len(qids)} questions; tag={tag}")

    # Fetch baseline up front so we can (1) supply baseline_sql to the agent for
    # the keep_baseline_sql affordance, and (2) merge with non-routed predictions later.
    print(f"[routed] fetching baseline from {baseline_filename}")
    baseline_payload = _fetch_baseline_via_volume.remote(baseline_filename)
    baseline_sql_by_qid: dict[str, str] = {
        str(int(r["question_id"])): (r.get("predicted_sql") or "")
        for r in baseline_payload["results"]
    }

    agent = Agent(
        model_name=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_turns=max_turns,
        tool_timeout_cs=int(round(tool_timeout_s * 100)),
        n_samples=n_samples,
        max_tokens=max_tokens,
        temperature_x100=int(round(temperature * 100)),
        agentic_version=agentic_version,
    )
    result = agent.solve_batch.remote(
        split=split, limit=0, save_as=tag, question_ids=qids,
        baseline_sql_by_qid=baseline_sql_by_qid,
    )
    agent_preds = result["predictions"]
    agent_stats = result["stats"]

    print("[routed] agent stats:")
    print(json.dumps(agent_stats, indent=2))

    # Score the agent's predictions alone first (so we know agent EX on the routed set).
    routed_only_tag = tag.replace(".json", ".routed_only.json")
    if not routed_only_tag.endswith(".json"):
        routed_only_tag += ".routed_only.json"
    print(f"[routed] scoring agent-only predictions on the routed subset → {routed_only_tag}")
    routed_summary = evaluate_predictions.remote(
        split=split, predictions=agent_preds, save_as=routed_only_tag,
    )
    print("[routed] agent-on-routed:")
    print(json.dumps(routed_summary, indent=2))

    # Merge: baseline elsewhere + agent on routed.
    print(f"[routed] merging with baseline predictions from {baseline_filename}")
    routed_set = set(int(q) for q in qids)
    agent_by_qid = {int(p["question_id"]): p for p in agent_preds}

    merged_preds: list[dict] = []
    for r in baseline_payload["results"]:
        qid = int(r["question_id"])
        if qid in routed_set and qid in agent_by_qid:
            ap = agent_by_qid[qid]
            merged_preds.append({
                "question_id": qid,
                "db_id": r["db_id"],
                "difficulty": r.get("difficulty"),
                "gold_sql": r.get("gold_sql", ""),
                "predicted_sql": ap.get("predicted_sql", ""),
                "raw_completion": "",
                "source": "agent",
            })
        else:
            merged_preds.append({
                "question_id": qid,
                "db_id": r["db_id"],
                "difficulty": r.get("difficulty"),
                "gold_sql": r.get("gold_sql", ""),
                "predicted_sql": r.get("predicted_sql", ""),
                "raw_completion": "",
                "source": "baseline",
            })

    merged_tag = tag.replace(".json", ".merged.json")
    print(f"[routed] scoring merged full-dev predictions → {merged_tag}")
    merged_summary = evaluate_predictions.remote(
        split=split, predictions=merged_preds, save_as=merged_tag,
    )
    print("[routed] merged full-dev:")
    print(json.dumps(merged_summary, indent=2))

    # Final compact summary
    print()
    print("=" * 60)
    print("[routed] FINAL SUMMARY")
    print("=" * 60)
    print(f"  routed-set size           : {len(qids)}")
    print(f"  baseline EX on routed     : {routing_payload.get('baseline_routed_ex'):.4f}")
    print(f"  agent EX on routed        : {routed_summary['ex']:.4f}")
    print(f"  merged full-dev EX        : {merged_summary['ex']:.4f}")
    print(f"  baseline full-dev EX      : {baseline_payload['ex']:.4f}")
    print(f"  saved agent-only          : {routed_only_tag}")
    print(f"  saved merged              : {merged_tag}")


@app.function(
    image=cpu_image,
    volumes={RESULTS_ROOT: results_vol},
    cpu=2,
    timeout=60 * 5,
)
def _fetch_baseline_via_volume(filename: str) -> dict:
    """Read a JSON file off the bird-results volume and return its contents.

    Used by `run_agentic_routed` so the local entrypoint can merge predictions
    without needing a local copy of the baseline file.
    """
    return json.loads((Path(RESULTS_ROOT) / filename).read_text())


@app.cls(
    image=gpu_image_q36,
    gpu="B200",
    volumes={HF_HOME: hf_cache, BIRD_ROOT: bird_data, RESULTS_ROOT: results_vol},
    timeout=2 * 60 * 60,
    scaledown_window=300,
)
class AgentQ36:
    """Qwen3.6-27B-aware variant of Agent, using vllm 0.20.2 + transformers 5.8.0."""

    model_name: str = modal.parameter(default="Qwen/Qwen3.6-27B")
    tensor_parallel_size: int = modal.parameter(default=1)
    max_model_len: int = modal.parameter(default=16384)
    max_turns: int = modal.parameter(default=6)
    tool_timeout_cs: int = modal.parameter(default=1000)
    n_samples: int = modal.parameter(default=3)
    max_tokens: int = modal.parameter(default=1024)
    temperature_x100: int = modal.parameter(default=0)
    agentic_version: str = modal.parameter(default="v1")

    @modal.enter()
    def _load(self):
        from bird.inference import VLLMEngine

        t0 = time.time()
        # Q3.6 GDN requires max_num_batched_tokens=2096 + FLASH_ATTN.
        self.engine = VLLMEngine(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            download_dir=HF_HOME,
            max_num_batched_tokens=2096,
            attention_backend="FLASH_ATTN",
        )
        print(f"[agent-q36] loaded {self.model_name} in {time.time() - t0:.1f}s")

    @modal.method()
    def solve_batch(
        self,
        split: str,
        limit: int,
        save_as: str,
        question_ids: list[int] | None = None,
        baseline_sql_by_qid: dict[str, str] | None = None,
    ) -> dict:
        from bird.agentic import build_initial_messages, step_agent
        from bird.data import load_split
        from bird.inference import GenConfig
        from bird.schema import extract_schema

        sp = load_split(Path(BIRD_ROOT) / split, name=split)
        if question_ids is not None:
            wanted = set(int(q) for q in question_ids)
            examples = [ex for ex in sp.examples if ex.question_id in wanted]
            missing = wanted - {ex.question_id for ex in examples}
            if missing:
                print(f"[agent-q36] WARNING: {len(missing)} requested qids not in split "
                      f"(sample: {sorted(missing)[:5]})")
            print(f"[agent-q36] subset mode: routing {len(examples)}/{len(wanted)} questions")
        else:
            examples = sp.examples[:limit] if limit else sp.examples

        temperature = self.temperature_x100 / 100.0
        tool_timeout_s = self.tool_timeout_cs / 100.0
        cfg = GenConfig(n=1, temperature=temperature, top_p=1.0, max_tokens=self.max_tokens)
        baseline_lookup: dict[int, str] = {}
        if baseline_sql_by_qid:
            for k, v in baseline_sql_by_qid.items():
                try:
                    baseline_lookup[int(k)] = v or ""
                except (TypeError, ValueError):
                    continue
        if baseline_lookup:
            print(f"[agent-q36] baseline SQL available for {len(baseline_lookup)} qids")

        def chat_fn(messages):
            outs = self.engine.chat([messages], cfg)
            return outs[0].texts[0] if outs and outs[0].texts else ""

        schema_cache: dict = {}
        predictions: list[dict] = []
        traces: list[dict] = []

        for i, ex in enumerate(examples):
            if ex.db_id not in schema_cache:
                schema_cache[ex.db_id] = extract_schema(
                    sp.db_path(ex.db_id), ex.db_id, n_samples=self.n_samples,
                )
            schema = schema_cache[ex.db_id]
            db_path = sp.db_path(ex.db_id)
            baseline_sql = baseline_lookup.get(ex.question_id, "")

            t0 = time.time()
            msgs = build_initial_messages(
                ex, schema, n_samples=self.n_samples, baseline_sql=baseline_sql,
                agentic_version=self.agentic_version,
            )
            try:
                trace = step_agent(
                    msgs,
                    db_path=db_path,
                    max_turns=self.max_turns,
                    tool_timeout_s=tool_timeout_s,
                    chat_fn=chat_fn,
                    baseline_sql=baseline_sql,
                    agentic_version=self.agentic_version,
                )
            except Exception as e:
                print(f"[agent-q36] q{ex.question_id}: trace crashed: {e!r}")
                trace = None

            elapsed = time.time() - t0
            if trace is None:
                predictions.append({
                    "question_id": ex.question_id, "db_id": ex.db_id,
                    "difficulty": ex.difficulty, "gold_sql": ex.sql,
                    "predicted_sql": "", "raw_completion": "",
                })
                traces.append({
                    "question_id": ex.question_id, "turns": 0, "n_tool_calls": 0,
                    "n_exec_errors": 0, "completion_chars": 0, "finish_reason": "crash",
                    "elapsed_s": elapsed,
                })
                continue

            predictions.append({
                "question_id": ex.question_id, "db_id": ex.db_id,
                "difficulty": ex.difficulty, "gold_sql": ex.sql,
                "predicted_sql": trace.final_sql, "raw_completion": "",
            })
            traces.append({
                "question_id": ex.question_id, "db_id": ex.db_id,
                "difficulty": ex.difficulty,
                "turns": trace.turns, "n_tool_calls": trace.n_tool_calls,
                "n_exec_errors": trace.n_exec_errors,
                "completion_chars": trace.completion_chars,
                "finish_reason": trace.finish_reason, "elapsed_s": elapsed,
                "history": trace.history,
            })
            if (i + 1) % 10 == 0 or i == len(examples) - 1:
                print(f"[agent-q36] {i + 1}/{len(examples)}  q{ex.question_id} "
                      f"turns={trace.turns} calls={trace.n_tool_calls} "
                      f"finish={trace.finish_reason} t={elapsed:.1f}s")

        if save_as:
            out = Path(RESULTS_ROOT) / save_as.replace(".json", ".traces.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(traces, indent=2, default=str))
            results_vol.commit()
            print(f"[agent-q36] wrote traces to {out}")

        n = len(traces)
        finish_counts: dict[str, int] = {}
        for t in traces:
            finish_counts[t["finish_reason"]] = finish_counts.get(t["finish_reason"], 0) + 1
        return {
            "predictions": predictions,
            "stats": {
                "n": n,
                "avg_turns": sum(t["turns"] for t in traces) / max(n, 1),
                "avg_tool_calls": sum(t["n_tool_calls"] for t in traces) / max(n, 1),
                "avg_exec_errors": sum(t["n_exec_errors"] for t in traces) / max(n, 1),
                "avg_completion_chars": sum(t["completion_chars"] for t in traces) / max(n, 1),
                "finish_reason_counts": finish_counts,
            },
        }


@app.cls(
    image=gpu_image_q36,
    gpu="B200",
    volumes={HF_HOME: hf_cache, BIRD_ROOT: bird_data, RESULTS_ROOT: results_vol},
    timeout=2 * 60 * 60,
    scaledown_window=300,
)
class SamplerQ36:
    """Qwen3.6-27B-aware Sampler, mirroring Sampler but on the q36 image."""

    model_name: str = modal.parameter(default="Qwen/Qwen3.6-27B")
    tensor_parallel_size: int = modal.parameter(default=1)
    max_model_len: int = modal.parameter(default=16384)

    @modal.enter()
    def _load(self):
        from bird.inference import VLLMEngine

        t0 = time.time()
        # Q3.6 GDN requires max_num_batched_tokens=2096 + FLASH_ATTN.
        self.engine = VLLMEngine(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            download_dir=HF_HOME,
            max_num_batched_tokens=2096,
            attention_backend="FLASH_ATTN",
        )
        print(f"[sampler-q36] loaded {self.model_name} in {time.time() - t0:.1f}s")

    @modal.method()
    def vote_full(
        self,
        split: str,
        n_votes: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        save_as: str,
        vote_timeout_s: float = 15.0,
        limit: int = 0,
    ) -> dict:
        from bird.data import load_split
        from bird.inference import GenConfig
        from bird.prompts import build_messages, extract_sql
        from bird.schema import extract_schema
        from bird.voting import vote as _vote

        sp = load_split(Path(BIRD_ROOT) / split, name=split)
        examples = sp.examples[:limit] if limit else sp.examples
        print(f"[voting-q36] preparing prompts for {len(examples)} questions"
              + (f" (limit={limit})" if limit else ""))

        schema_cache: dict = {}
        convos: list[list[dict]] = []
        metas: list[dict] = []
        for ex in examples:
            if ex.db_id not in schema_cache:
                schema_cache[ex.db_id] = extract_schema(
                    sp.db_path(ex.db_id), ex.db_id, n_samples=3,
                )
            convos.append(build_messages(ex, schema_cache[ex.db_id], n_samples=3))
            metas.append({
                "question_id": ex.question_id, "db_id": ex.db_id,
                "difficulty": ex.difficulty, "gold_sql": ex.sql,
            })

        cfg = GenConfig(n=n_votes, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        print(f"[voting-q36] generating n={n_votes} per question at T={temperature} top_p={top_p}")
        t0 = time.time()
        outs = self.engine.chat(convos, cfg)
        print(f"[voting-q36] generated samples in {time.time() - t0:.1f}s")

        split_root = Path(BIRD_ROOT) / split
        db_dir = split_root / f"{split}_databases"

        predictions: list[dict] = []
        t_vote = time.time()
        for meta, out in zip(metas, outs):
            cands = [extract_sql(t) for t in (out.texts or [])]
            db_path = str(db_dir / meta["db_id"] / f'{meta["db_id"]}.sqlite')
            outcome = _vote(cands, db_path, timeout_s=vote_timeout_s)
            predictions.append({
                "question_id": meta["question_id"], "db_id": meta["db_id"],
                "difficulty": meta["difficulty"], "gold_sql": meta["gold_sql"],
                "predicted_sql": outcome["winner_sql"],
                "candidate_sqls": cands,
                "voting_metadata": {
                    "winner_count": outcome["winner_count"],
                    "n_candidates": outcome["n_candidates"],
                    "n_executable": outcome["n_executable"],
                    "n_distinct_results": outcome["n_distinct_results"],
                    "fallback_used": outcome["fallback_used"],
                },
            })
        print(f"[voting-q36] vote done in {time.time() - t_vote:.1f}s")

        if save_as:
            out_path = Path(RESULTS_ROOT) / save_as
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "split": split, "n": len(predictions), "n_votes": n_votes,
                "temperature": temperature, "top_p": top_p,
                "results": predictions,
            }, indent=2))
            results_vol.commit()
            print(f"[voting-q36] wrote {out_path}")

        return {"split": split, "n": len(predictions), "saved": save_as}


@app.local_entrypoint()
def run_voting_q36(
    split: str = "dev",
    model: str = "Qwen/Qwen3.6-27B",
    n_votes: int = 8,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 1024,
    vote_timeout_s: float = 15.0,
    save_as: str = "voting-qwen3.6-27b-dev-full.json",
    tensor_parallel_size: int = 1,
    max_model_len: int = 16384,
    limit: int = 0,
):
    """Q3.6-specific voting entrypoint (uses gpu_image_q36).

    Pass `--limit N` for a smoke test (N questions, not a full split).
    """
    print(f"[voting-q36] {model} split={split} n_votes={n_votes} T={temperature} limit={limit}")
    sampler = SamplerQ36(
        model_name=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )
    summary = sampler.vote_full.remote(
        split=split, n_votes=n_votes, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p, save_as=save_as,
        vote_timeout_s=vote_timeout_s, limit=limit,
    )
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def run_agentic_routed_q36(
    split: str = "dev",
    routing_from: str = "results/routing_set.json",
    baseline_filename: str = "baseline-qwen3.6-27b-dev-full.json",
    model: str = "Qwen/Qwen3.6-27B",
    max_turns: int = 6,
    tool_timeout_s: float = 10.0,
    n_samples: int = 3,
    max_tokens: int = 2048,  # bumped from 1024 to kill the 12% mid-JSON truncation cases observed in v1
    temperature: float = 0.0,
    save_as: str = "",
    tensor_parallel_size: int = 1,
    max_model_len: int = 16384,
    agentic_version: str = "v1",
):
    """Q3.6-specific routed-agentic entrypoint.

    Same pipeline as `run_agentic_routed` but uses AgentQ36 on the q36 image
    (vllm 0.20.2 + transformers 5.8.0). Default agentic_version is "v1".
    """
    routing_payload = json.loads(Path(routing_from).read_text())
    qids: list[int] = list(routing_payload["question_ids"])
    if not qids:
        raise SystemExit("routing set is empty; nothing to do")
    if len(qids) < 50:
        raise SystemExit(f"routing set too small ({len(qids)} < 50); aborting per spec")

    tag = save_as or f"agentic-routed-q36-{split}-{len(qids)}-{agentic_version}-{int(time.time())}.json"
    print(f"[routed-q36] running agent ({agentic_version}) on {len(qids)} questions; tag={tag}")

    print(f"[routed-q36] fetching baseline from {baseline_filename}")
    baseline_payload = _fetch_baseline_via_volume.remote(baseline_filename)
    baseline_sql_by_qid: dict[str, str] = {
        str(int(r["question_id"])): (r.get("predicted_sql") or "")
        for r in baseline_payload["results"]
    }

    agent = AgentQ36(
        model_name=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_turns=max_turns,
        tool_timeout_cs=int(round(tool_timeout_s * 100)),
        n_samples=n_samples,
        max_tokens=max_tokens,
        temperature_x100=int(round(temperature * 100)),
        agentic_version=agentic_version,
    )
    result = agent.solve_batch.remote(
        split=split, limit=0, save_as=tag, question_ids=qids,
        baseline_sql_by_qid=baseline_sql_by_qid,
    )
    agent_preds = result["predictions"]
    agent_stats = result["stats"]

    print("[routed-q36] agent stats:")
    print(json.dumps(agent_stats, indent=2))

    routed_only_tag = tag.replace(".json", ".routed_only.json")
    if not routed_only_tag.endswith(".json"):
        routed_only_tag += ".routed_only.json"
    print(f"[routed-q36] scoring agent-only predictions → {routed_only_tag}")
    routed_summary = evaluate_predictions.remote(
        split=split, predictions=agent_preds, save_as=routed_only_tag,
    )
    print("[routed-q36] agent-on-routed:")
    print(json.dumps(routed_summary, indent=2))

    print(f"[routed-q36] merging with baseline predictions from {baseline_filename}")
    routed_set = set(int(q) for q in qids)
    agent_by_qid = {int(p["question_id"]): p for p in agent_preds}

    merged_preds: list[dict] = []
    for r in baseline_payload["results"]:
        qid = int(r["question_id"])
        if qid in routed_set and qid in agent_by_qid:
            ap = agent_by_qid[qid]
            merged_preds.append({
                "question_id": qid, "db_id": r["db_id"],
                "difficulty": r.get("difficulty"),
                "gold_sql": r.get("gold_sql", ""),
                "predicted_sql": ap.get("predicted_sql", ""),
                "raw_completion": "", "source": "agent",
            })
        else:
            merged_preds.append({
                "question_id": qid, "db_id": r["db_id"],
                "difficulty": r.get("difficulty"),
                "gold_sql": r.get("gold_sql", ""),
                "predicted_sql": r.get("predicted_sql", ""),
                "raw_completion": "", "source": "baseline",
            })

    merged_tag = tag.replace(".json", ".merged.json")
    print(f"[routed-q36] scoring merged full-dev predictions → {merged_tag}")
    merged_summary = evaluate_predictions.remote(
        split=split, predictions=merged_preds, save_as=merged_tag,
    )
    print("[routed-q36] merged full-dev:")
    print(json.dumps(merged_summary, indent=2))

    print()
    print("=" * 60)
    print("[routed-q36] FINAL SUMMARY")
    print("=" * 60)
    print(f"  routed-set size           : {len(qids)}")
    print(f"  baseline EX on routed     : {routing_payload.get('baseline_routed_ex'):.4f}")
    print(f"  agent EX on routed        : {routed_summary['ex']:.4f}")
    print(f"  merged full-dev EX        : {merged_summary['ex']:.4f}")
    print(f"  baseline full-dev EX      : {baseline_payload['ex']:.4f}")
    print(f"  saved agent-only          : {routed_only_tag}")
    print(f"  saved merged              : {merged_tag}")


@app.cls(
    image=gpu_image,
    gpu="B200",
    volumes={HF_HOME: hf_cache, BIRD_ROOT: bird_data, RESULTS_ROOT: results_vol},
    timeout=2 * 60 * 60,
    scaledown_window=300,
)
class Sampler:
    """Single-pass non-greedy sampler used to generate the T=0.7 sidecar
    predictions for cross-temperature disagreement routing (Rule 4).
    """

    model_name: str = modal.parameter(default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    tensor_parallel_size: int = modal.parameter(default=1)
    max_model_len: int = modal.parameter(default=16384)

    @modal.enter()
    def _load(self):
        from bird.inference import VLLMEngine

        t0 = time.time()
        self.engine = VLLMEngine(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            download_dir=HF_HOME,
        )
        print(f"[sampler] loaded {self.model_name} in {time.time() - t0:.1f}s")

    @modal.method()
    def sample_full(
        self,
        split: str,
        n_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        save_as: str,
    ) -> dict:
        """Generate ONE non-greedy sample per question on the full split.

        Returns predictions in the same shape as `evaluate_predictions` expects.
        """
        from bird.data import load_split
        from bird.inference import GenConfig
        from bird.prompts import build_messages, extract_sql
        from bird.schema import extract_schema

        sp = load_split(Path(BIRD_ROOT) / split, name=split)
        examples = sp.examples
        print(f"[sampler] preparing prompts for {len(examples)} questions")

        schema_cache: dict = {}
        convos: list[list[dict]] = []
        metas: list[dict] = []
        for ex in examples:
            if ex.db_id not in schema_cache:
                schema_cache[ex.db_id] = extract_schema(
                    sp.db_path(ex.db_id), ex.db_id, n_samples=n_samples,
                )
            convos.append(build_messages(ex, schema_cache[ex.db_id], n_samples=n_samples))
            metas.append({
                "question_id": ex.question_id,
                "db_id": ex.db_id,
                "difficulty": ex.difficulty,
                "gold_sql": ex.sql,
            })

        cfg = GenConfig(
            n=1, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        )
        print(f"[sampler] generating at T={temperature} top_p={top_p} n=1")
        t0 = time.time()
        outs = self.engine.chat(convos, cfg)
        print(f"[sampler] generated {len(outs)} samples in {time.time() - t0:.1f}s")

        predictions: list[dict] = []
        for meta, out in zip(metas, outs):
            text = out.texts[0] if out.texts else ""
            sql = extract_sql(text)
            predictions.append({
                "question_id": meta["question_id"],
                "db_id": meta["db_id"],
                "difficulty": meta["difficulty"],
                "gold_sql": meta["gold_sql"],
                "predicted_sql": sql,
                "raw_completion": text,
            })

        if save_as:
            out_path = Path(RESULTS_ROOT) / save_as
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"split": split, "n": len(predictions),
                                            "temperature": temperature, "top_p": top_p,
                                            "results": predictions}, indent=2))
            results_vol.commit()
            print(f"[sampler] wrote {out_path}")

        return {"split": split, "n": len(predictions), "saved": save_as}

    @modal.method()
    def vote_full(
        self,
        split: str,
        n_votes: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        save_as: str,
        vote_timeout_s: float = 15.0,
    ) -> dict:
        """Generate n_votes candidates per question and majority-vote on execution.

        Mirrors `run_with_voting` from feat/voting on the main app, packaged
        inside the agentic worktree so we don't have to switch images.
        Writes a file shaped like `voting-*-dev-full.json` with per-result
        `voting_metadata` so the routing builder's rule 3 can consume it.
        """
        from bird.data import load_split
        from bird.inference import GenConfig
        from bird.prompts import build_messages, extract_sql
        from bird.schema import extract_schema
        from bird.voting import vote as _vote

        sp = load_split(Path(BIRD_ROOT) / split, name=split)
        examples = sp.examples
        print(f"[voting] preparing prompts for {len(examples)} questions")

        schema_cache: dict = {}
        convos: list[list[dict]] = []
        metas: list[dict] = []
        for ex in examples:
            if ex.db_id not in schema_cache:
                schema_cache[ex.db_id] = extract_schema(
                    sp.db_path(ex.db_id), ex.db_id, n_samples=3,
                )
            convos.append(build_messages(ex, schema_cache[ex.db_id], n_samples=3))
            metas.append({
                "question_id": ex.question_id,
                "db_id": ex.db_id,
                "difficulty": ex.difficulty,
                "gold_sql": ex.sql,
            })

        cfg = GenConfig(
            n=n_votes, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        )
        print(f"[voting] generating n={n_votes} per question at T={temperature} top_p={top_p}")
        t0 = time.time()
        outs = self.engine.chat(convos, cfg)
        print(f"[voting] generated samples in {time.time() - t0:.1f}s")

        split_root = Path(BIRD_ROOT) / split
        db_dir = split_root / f"{split}_databases"

        predictions: list[dict] = []
        t_vote = time.time()
        for meta, out in zip(metas, outs):
            cands = [extract_sql(t) for t in (out.texts or [])]
            db_path = str(db_dir / meta["db_id"] / f'{meta["db_id"]}.sqlite')
            outcome = _vote(cands, db_path, timeout_s=vote_timeout_s)
            predictions.append({
                "question_id": meta["question_id"],
                "db_id": meta["db_id"],
                "difficulty": meta["difficulty"],
                "gold_sql": meta["gold_sql"],
                "predicted_sql": outcome["winner_sql"],
                "candidate_sqls": cands,
                "voting_metadata": {
                    "winner_count": outcome["winner_count"],
                    "n_candidates": outcome["n_candidates"],
                    "n_executable": outcome["n_executable"],
                    "n_distinct_results": outcome["n_distinct_results"],
                    "fallback_used": outcome["fallback_used"],
                },
            })
        print(f"[voting] vote done in {time.time() - t_vote:.1f}s")

        if save_as:
            out_path = Path(RESULTS_ROOT) / save_as
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "split": split, "n": len(predictions), "n_votes": n_votes,
                "temperature": temperature, "top_p": top_p,
                "results": predictions,
            }, indent=2))
            results_vol.commit()
            print(f"[voting] wrote {out_path}")

        return {"split": split, "n": len(predictions), "saved": save_as}


@app.local_entrypoint()
def run_voting(
    split: str = "dev",
    model: str = "Qwen/Qwen3.6-27B",
    n_votes: int = 8,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 1024,
    vote_timeout_s: float = 15.0,
    save_as: str = "voting-qwen3.6-27b-dev-full.json",
    tensor_parallel_size: int = 1,
    max_model_len: int = 16384,
):
    """Generate n=8 voting candidates per question + majority-vote on execution.

    Output schema matches `voting-*-dev-full.json` on the bird-results volume so
    `scripts/build_routing_set.py` can ingest it for rule 3 (low vote-share).
    """
    print(f"[voting] {model} split={split} n_votes={n_votes} T={temperature}")
    sampler = Sampler(
        model_name=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )
    summary = sampler.vote_full.remote(
        split=split,
        n_votes=n_votes,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        save_as=save_as,
        vote_timeout_s=vote_timeout_s,
    )
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def run_t07_baseline(
    split: str = "dev",
    model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    n_samples: int = 3,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    save_as: str = "baseline-qwen3-coder-30b-a3b-instruct-dev-t07.json",
    tensor_parallel_size: int = 1,
    max_model_len: int = 16384,
):
    """One non-greedy sample per question on the full split. Used as the T=0.7
    sidecar for cross-temperature-disagreement routing (Rule 4).
    """
    print(f"[t07] sampling {model} on split={split} at T={temperature} top_p={top_p}")
    sampler = Sampler(
        model_name=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )
    summary = sampler.sample_full.remote(
        split=split,
        n_samples=n_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        save_as=save_as,
    )
    print(json.dumps(summary, indent=2))


@app.function(
    image=cpu_image,
    volumes={BIRD_ROOT: bird_data, RESULTS_ROOT: results_vol},
    cpu=8,
    timeout=60 * 30,
)
def compute_t07_disagreement(
    split: str,
    baseline_filename: str,
    t07_filename: str,
    save_as: str,
    timeout_s: float = 10.0,
) -> dict:
    """Execute greedy SQL and T=0.7 SQL side-by-side; flag questions where the
    result-sets differ. Output is `{qid: bool}`. Skips when either SQL errors
    or the rows match by canonical hash.
    """
    import hashlib
    import sqlite3
    import threading

    split_root = Path(BIRD_ROOT) / split
    db_dir = split_root / f"{split}_databases"

    baseline = json.loads((Path(RESULTS_ROOT) / baseline_filename).read_text())
    t07 = json.loads((Path(RESULTS_ROOT) / t07_filename).read_text())

    base_by_qid = {int(r["question_id"]): r for r in baseline["results"]}
    t07_by_qid = {int(r["question_id"]): r for r in t07["results"]}

    def _exec_ro(db_path, sql, t_s):
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=t_s)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b
        timer = threading.Timer(t_s, conn.interrupt)
        timer.daemon = True
        timer.start()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            return cur.fetchall()
        finally:
            timer.cancel()
            conn.close()

    def _canonical(rows):
        try:
            normed = sorted([tuple(repr(c) for c in r) for r in rows])
        except Exception:
            normed = sorted(repr(r) for r in rows)
        return hashlib.sha1(repr(normed).encode("utf-8")).hexdigest()

    flags: dict[int, bool] = {}
    n_checked = n_disagree = n_skipped = 0
    for qid, base in base_by_qid.items():
        if qid not in t07_by_qid:
            n_skipped += 1
            continue
        base_sql = (base.get("predicted_sql") or "").strip()
        t07_sql = (t07_by_qid[qid].get("predicted_sql") or "").strip()
        if not base_sql or not t07_sql:
            n_skipped += 1
            continue
        db_path = db_dir / base["db_id"] / f'{base["db_id"]}.sqlite'
        try:
            base_rows = _exec_ro(db_path, base_sql, timeout_s)
        except Exception:
            # Baseline errors are already routed via rule 1; this signal is moot.
            flags[qid] = False
            n_skipped += 1
            continue
        try:
            t07_rows = _exec_ro(db_path, t07_sql, timeout_s)
        except Exception:
            # T=0.7 errored — strong disagreement signal (modal sampling unstable).
            flags[qid] = True
            n_checked += 1
            n_disagree += 1
            continue
        same = _canonical(base_rows) == _canonical(t07_rows)
        flags[qid] = not same
        n_checked += 1
        if not same:
            n_disagree += 1
        if n_checked % 200 == 0:
            print(f"[t07-diff] {n_checked} checked, {n_disagree} disagree so far")

    payload = {str(k): v for k, v in flags.items()}
    out = Path(RESULTS_ROOT) / save_as
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    results_vol.commit()
    print(f"[t07-diff] DONE: {n_checked} executed, {n_disagree} disagree, {n_skipped} skipped")
    print(f"[t07-diff] wrote {out}")
    return {"n_checked": n_checked, "n_disagree": n_disagree, "n_skipped": n_skipped, "saved": save_as}


@app.local_entrypoint()
def run_t07_disagreement(
    split: str = "dev",
    baseline_filename: str = "baseline-qwen3-coder-30b-a3b-instruct-dev-full.json",
    t07_filename: str = "baseline-qwen3-coder-30b-a3b-instruct-dev-t07.json",
    save_as: str = "t07-disagreement-qwen3-coder-30b-a3b-instruct-dev.json",
):
    summary = compute_t07_disagreement.remote(
        split=split,
        baseline_filename=baseline_filename,
        t07_filename=t07_filename,
        save_as=save_as,
    )
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def run_agentic(
    split: str = "dev",
    limit: int = 50,
    model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    max_turns: int = 6,
    tool_timeout_s: float = 10.0,
    n_samples: int = 3,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    save_as: str = "",
    tensor_parallel_size: int = 1,
    max_model_len: int = 16384,
):
    """Run the agentic loop on `limit` BIRD examples and evaluate."""
    save_tag = save_as or f"agentic-{Path(model).name}-{split}-{limit}-{int(time.time())}.json"
    print(f"[orchestrator] agentic run on {model}, split={split}, limit={limit}, "
          f"max_turns={max_turns}")

    agent = Agent(
        model_name=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_turns=max_turns,
        tool_timeout_cs=int(round(tool_timeout_s * 100)),
        n_samples=n_samples,
        max_tokens=max_tokens,
        temperature_x100=int(round(temperature * 100)),
    )
    result = agent.solve_batch.remote(split=split, limit=limit, save_as=save_tag)
    predictions = result["predictions"]
    stats = result["stats"]

    print("[agent] aggregate stats:")
    print(json.dumps(stats, indent=2))

    print("[orchestrator] running execution evaluator")
    summary = evaluate_predictions.remote(
        split=split, predictions=predictions, save_as=save_tag,
    )
    summary["agent_stats"] = stats
    print(json.dumps(summary, indent=2))
    print(f"[done] saved as {save_tag} on bird-results volume")
