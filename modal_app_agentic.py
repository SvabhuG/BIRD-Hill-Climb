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
    ) -> dict:
        """Run the agent on `limit` BIRD examples and return predictions + traces.

        If `question_ids` is provided, run on exactly that set (ignores `limit`).
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

            t0 = time.time()
            msgs = build_initial_messages(ex, schema, n_samples=self.n_samples)
            try:
                trace = step_agent(
                    msgs,
                    db_path=db_path,
                    max_turns=self.max_turns,
                    tool_timeout_s=tool_timeout_s,
                    chat_fn=chat_fn,
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
    result = agent.solve_batch.remote(
        split=split, limit=0, save_as=tag, question_ids=qids,
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
    baseline_payload = json.loads((Path(RESULTS_ROOT) / baseline_filename).read_text()) if Path(RESULTS_ROOT).exists() else None
    # The local entrypoint runs locally, so RESULTS_ROOT won't exist; pull baseline via Modal:
    if baseline_payload is None:
        baseline_payload = _fetch_baseline_via_volume.remote(baseline_filename)

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
