"""Modal app: BIRD hill-climbing.

Pieces:
  * Two persistent Volumes — one for the BIRD corpus, one for HF model weights.
    Both survive across runs so we never re-download.
  * `download_bird` — one-shot CPU function that populates the bird-data volume.
    Idempotent: if dev.json already exists it's a no-op.
  * `Inference` — vLLM engine on a B200; persistent class so the model loads once
    per container and serves many calls. Configure model + parallelism via env.
  * `evaluate_predictions` — CPU function that runs SQLite execution-accuracy
    in parallel processes against the bird-data volume.
  * `run_baseline` — local entrypoint that orchestrates the loop.

Usage:
    modal run modal_app.py::download_bird
    modal run modal_app.py::run_baseline --split dev --limit 50
"""
# NOTE: do NOT add `from __future__ import annotations` here — Modal reads
# `modal.parameter` annotations at class-decoration time and needs real types,
# not stringified ones.
import json
import time
from pathlib import Path

import modal

# Top-level imports kept light so `modal run` works even when the local CLI's
# Python doesn't have all our deps (e.g. sqlglot). Linker-specific imports are
# pulled in inside `run_with_linking`, which is the only entrypoint that needs them.
from bird.prompts import extract_sql, messages_to_raw_text


APP_NAME = "bird-climb"

# ---------- Volumes (persistent, shared across runs) ----------

bird_data = modal.Volume.from_name("bird-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("bird-results", create_if_missing=True)

BIRD_ROOT = "/data/bird"          # mount path for bird-data
HF_HOME = "/root/.cache/hf"       # mount path for hf-cache
RESULTS_ROOT = "/results"         # mount path for results_vol

# ---------- Images ----------

_PY = "3.11"

# CPU image used by data-prep + eval functions.
cpu_image = (
    modal.Image.debian_slim(python_version=_PY)
    .apt_install("unzip", "wget", "curl")
    .pip_install("tqdm", "pydantic>=2", "requests", "sqlglot>=25")
    .add_local_python_source("bird")
)

# GPU image used by the vLLM engine.
gpu_image = (
    modal.Image.debian_slim(python_version=_PY)
    .apt_install("git")
    # Known-working combo for the Qwen2Tokenizer.all_special_tokens_extended
    # incompatibility (verl-project/verl#4337, QwenLM/Qwen3-VL#2058,
    # rllm-org/rllm#388). Older vllm + newer transformers, OR newer vllm + older
    # transformers, both break — only this paired version works.
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


# ============================================================
# Data download
# ============================================================

DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
TRAIN_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"


@app.function(
    image=cpu_image,
    volumes={BIRD_ROOT: bird_data},
    timeout=60 * 60,
)
def download_bird(splits: str = "dev") -> dict:
    """Download and unzip BIRD splits into the bird-data volume.

    `splits` is a comma-separated string: "dev", "train", or "dev,train".

    Volume layout after a successful dev+train run:
        /data/bird/dev/dev.json
        /data/bird/dev/dev_databases/<db_id>/<db_id>.sqlite
        /data/bird/train/train.json
        /data/bird/train/train_databases/<db_id>/<db_id>.sqlite
    """
    import subprocess
    import zipfile

    split_list = [s.strip() for s in splits.split(",") if s.strip()]
    root = Path(BIRD_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for split in split_list:
        url = {"dev": DEV_URL, "train": TRAIN_URL}[split]
        target_dir = root / split
        marker = target_dir / f"{split}.json"
        if marker.exists():
            print(f"[skip] {split}: {marker} already present")
            summary[split] = {"status": "skipped", "path": str(target_dir)}
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        zip_path = root / f"{split}.zip"
        print(f"[download] {url} -> {zip_path}")
        subprocess.run(["curl", "-L", "-o", str(zip_path), url], check=True)

        print(f"[unzip] {zip_path} -> {target_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)

        # The release zip wraps everything in a dated folder like `dev_20240627/`.
        # Flatten so callers can find <split>.json at a stable path.
        nested = [p for p in target_dir.iterdir() if p.is_dir()]
        if marker.exists() is False and len(nested) == 1 and (nested[0] / f"{split}.json").exists():
            inner = nested[0]
            for child in inner.iterdir():
                child.rename(target_dir / child.name)
            inner.rmdir()

        # Inner DB zips (e.g. dev_databases.zip) sometimes need a second unzip pass.
        inner_zip = target_dir / f"{split}_databases.zip"
        if inner_zip.exists():
            print(f"[unzip] {inner_zip}")
            with zipfile.ZipFile(inner_zip) as zf:
                zf.extractall(target_dir)
            inner_zip.unlink()

        zip_path.unlink(missing_ok=True)

        with marker.open() as f:
            n = len(json.load(f))
        summary[split] = {"status": "downloaded", "n_examples": n, "path": str(target_dir)}
        print(f"[done] {split}: {n} examples at {target_dir}")

    bird_data.commit()
    return summary


# ============================================================
# Inference (GPU)
# ============================================================

@app.cls(
    image=gpu_image,
    gpu="B200",
    volumes={HF_HOME: hf_cache, BIRD_ROOT: bird_data},
    timeout=60 * 60,
    scaledown_window=300,
)
class Inference:
    model_name: str = modal.parameter(default="Qwen/Qwen2.5-Coder-7B-Instruct")
    tensor_parallel_size: int = modal.parameter(default=1)
    max_model_len: int = modal.parameter(default=16384)

    @modal.enter()
    def _load(self):
        # Lazy import so this file stays importable from a laptop.
        from bird.inference import VLLMEngine

        t0 = time.time()
        self.engine = VLLMEngine(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            download_dir=HF_HOME,
        )
        print(f"[inference] loaded {self.model_name} in {time.time() - t0:.1f}s")

    @modal.method()
    def chat(
        self,
        conversations: list[list[dict]],
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        from bird.inference import GenConfig

        cfg = GenConfig(
            n=n, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, stop=stop or [],
        )
        outs = self.engine.chat(conversations, cfg)
        return [o.texts for o in outs]

    @modal.method()
    def complete(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        """Raw-completion path for base models (no chat template)."""
        from bird.inference import GenConfig

        cfg = GenConfig(
            n=n, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, stop=stop or [],
        )
        outs = self.engine.complete(prompts, cfg)
        return [o.texts for o in outs]


# ============================================================
# Evaluation (CPU, parallel processes)
# ============================================================

@app.function(
    image=cpu_image,
    volumes={BIRD_ROOT: bird_data, RESULTS_ROOT: results_vol},
    cpu=8,
    timeout=60 * 30,
)
def evaluate_predictions(
    split: str,
    predictions: list[dict],   # [{question_id, db_id, predicted_sql, gold_sql, difficulty}]
    timeout_s: float = 30.0,
    workers: int = 8,
    save_as: str | None = None,
) -> dict:
    """Run execution-accuracy on a list of predictions, return summary + per-row results."""
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
            # Pass caller-attached fields through (raw_completion, linker_selection,
            # linking_recall, etc.), then overwrite with the authoritative eval status.
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

    # Return a slim summary; full results are on the volume if save_as was set.
    return {
        "split": split, "n": summary.n, "n_correct": summary.n_correct,
        "ex": summary.ex, "by_status": summary.by_status,
        "by_difficulty": summary.by_difficulty, "saved": save_as,
    }


# ============================================================
# Voting (CPU, parallel processes)
# ============================================================

@app.function(
    image=cpu_image,
    volumes={BIRD_ROOT: bird_data},
    cpu=8,
    timeout=60 * 30,
)
def vote_predictions(
    split: str,
    candidate_lists: list[list[str]],
    metas: list[dict],
    timeout_s: float = 15.0,
    workers: int = 8,
) -> list[dict]:
    """Majority-vote-on-execution per question.

    For each question, executes its candidate SQLs against the question's SQLite DB,
    hashes each result-set, and returns the SQL whose result is the most common.
    Returns a list aligned with `metas`: [{question_id, winner_sql, voting_metadata}].
    """
    import multiprocessing as mp

    split_root = Path(BIRD_ROOT) / split
    db_dir = split_root / f"{split}_databases"

    # Pack work items so we can fan out to a process pool. We carry the question_id
    # so results can be reassembled in caller order.
    jobs: list[tuple[int, str, list[str], str, float]] = []
    for cand_list, meta in zip(candidate_lists, metas):
        db_path = str(db_dir / meta["db_id"] / f'{meta["db_id"]}.sqlite')
        jobs.append((meta["question_id"], meta["db_id"], list(cand_list), db_path, timeout_s))

    t0 = time.time()
    if workers <= 1:
        results = [_vote_worker(j) for j in jobs]
    else:
        # `vote` is CPU-bound (SQLite execution + hashing) — use processes for true
        # parallelism, mirroring `evaluate_predictions`.
        with mp.get_context("spawn").Pool(workers) as pool:
            results = list(pool.imap(_vote_worker, jobs, chunksize=4))
    print(f"[vote] voted on {len(results)} questions in {time.time() - t0:.1f}s")
    return results


def _vote_worker(job):
    """Module-level worker so the spawn-context pool can pickle it."""
    from bird.voting import vote
    qid, db_id, cands, db_path, t = job
    outcome = vote(cands, db_path, timeout_s=t)
    return {
        "question_id": qid,
        "db_id": db_id,
        "winner_sql": outcome["winner_sql"],
        "voting_metadata": {
            "winner_count": outcome["winner_count"],
            "n_candidates": outcome["n_candidates"],
            "n_executable": outcome["n_executable"],
            "n_distinct_results": outcome["n_distinct_results"],
            "fallback_used": outcome["fallback_used"],
        },
    }


# ============================================================
# Local orchestrator
# ============================================================

@app.local_entrypoint()
def run_baseline(
    split: str = "dev",
    limit: int = 0,
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    n_samples: int = 3,           # rows-per-table samples in the schema block
    max_tokens: int = 1024,
    temperature: float = 0.0,
    save_as: str = "",
    base_model: bool = False,     # True for base models w/o chat template
    tensor_parallel_size: int = 1,
    max_model_len: int = 16384,
):
    """Greedy zero-shot baseline. The number-on-the-board run."""
    print(f"[orchestrator] preparing prompts for split={split}, limit={limit or 'all'}")
    convos, examples_meta = _prepare_local.remote(split, limit, n_samples)

    print(f"[orchestrator] {len(convos)} prompts prepared; running inference on {model} "
          f"(base_model={base_model}, tp={tensor_parallel_size})")
    inf = Inference(
        model_name=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )
    save_tag = save_as or f"baseline-{Path(model).name}-{split}-{int(time.time())}.json"
    if base_model:
        raw_prompts = [messages_to_raw_text(c) for c in convos]
        raw = inf.complete.remote(raw_prompts, n=1, temperature=temperature, max_tokens=max_tokens)
    else:
        raw = inf.chat.remote(convos, n=1, temperature=temperature, max_tokens=max_tokens)

    predictions = []
    for meta, gens in zip(examples_meta, raw):
        text = gens[0] if gens else ""
        sql = extract_sql(text)
        predictions.append({
            "question_id": meta["question_id"],
            "db_id": meta["db_id"],
            "difficulty": meta["difficulty"],
            "gold_sql": meta["gold_sql"],
            "predicted_sql": sql,
            "raw_completion": text,
        })

    print(f"[orchestrator] running execution evaluator")
    summary = evaluate_predictions.remote(
        split=split, predictions=predictions, save_as=save_tag,
    )
    print(json.dumps(summary, indent=2))
    print(f"[done] saved as {save_tag} on bird-results volume")


@app.function(
    image=cpu_image,
    volumes={BIRD_ROOT: bird_data},
    timeout=60 * 10,
)
def _prepare_local(split: str, limit: int, n_samples: int):
    """Build (messages, meta) pairs from the BIRD volume.

    Runs on Modal so it can read the bird-data volume; the orchestrator just
    receives Python objects back. Schema extraction is the slow bit (per-DB SQLite
    introspection); we cache by db_id within this single call.
    """
    from bird.data import load_split
    from bird.prompts import build_messages
    from bird.schema import extract_schema

    sp = load_split(Path(BIRD_ROOT) / split, name=split)
    examples = sp.examples[:limit] if limit else sp.examples

    schema_cache: dict[str, object] = {}
    convos: list[list[dict]] = []
    metas: list[dict] = []
    for ex in examples:
        if ex.db_id not in schema_cache:
            schema_cache[ex.db_id] = extract_schema(sp.db_path(ex.db_id), ex.db_id, n_samples=n_samples)
        msgs = build_messages(ex, schema_cache[ex.db_id], n_samples=n_samples)
        convos.append(msgs)
        metas.append({
            "question_id": ex.question_id,
            "db_id": ex.db_id,
            "difficulty": ex.difficulty,
            "gold_sql": ex.sql,
        })
    return convos, metas


@app.function(
    image=cpu_image,
    volumes={BIRD_ROOT: bird_data},
    timeout=60 * 10,
)
def _prepare_for_linking(split: str, limit: int, n_samples: int):
    """Build linker prompts + return per-DB schemas + per-question metadata.

    Schemas are returned as a unique-by-db_id dict so we don't ship 1500x copies
    of the same schema back to the orchestrator. Locally we use these schemas to:
      - parse linker outputs into Selections (validated against the schema)
      - augment with lexical_link
      - ensure_keys (PK/FK closure)
      - measure linking-recall against gold_columns(gold_sql, schema)
      - restrict_schema before building the SQL-gen prompt
    """
    from bird.data import load_split
    from bird.linking import build_linker_messages
    from bird.schema import extract_schema

    sp = load_split(Path(BIRD_ROOT) / split, name=split)
    examples = sp.examples[:limit] if limit else sp.examples

    schemas: dict[str, object] = {}
    linker_msgs: list[list[dict]] = []
    metas: list[dict] = []
    for ex in examples:
        if ex.db_id not in schemas:
            schemas[ex.db_id] = extract_schema(sp.db_path(ex.db_id), ex.db_id, n_samples=n_samples)
        linker_msgs.append(build_linker_messages(ex, schemas[ex.db_id]))
        metas.append({
            "question_id": ex.question_id,
            "db_id": ex.db_id,
            "question": ex.question,
            "evidence": ex.evidence,
            "difficulty": ex.difficulty,
            "gold_sql": ex.sql,
        })
    return linker_msgs, schemas, metas


@app.local_entrypoint()
def run_with_linking(
    split: str = "dev",
    limit: int = 0,
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    n_samples: int = 3,
    use_lexical: bool = True,
    save_as: str = "",
):
    """Schema-linker pass + filtered SQL gen + eval. Reports linking-recall alongside EX.

    Two inference passes per question:
      1. Linker: compact-schema prompt -> JSON of relevant tables/columns
      2. Generator: filtered DDL prompt -> SQL

    Linking recall is computed locally against the gold SQL (via sqlglot) so we
    can attribute any EX gap to the linking stage vs. the generation stage.
    """
    # Linker-only deps; imported here so `run_baseline` doesn't drag sqlglot in
    # via the local CLI's Python.
    from bird.data import BirdExample
    from bird.linking import (
        Selection,
        ensure_keys,
        lexical_link,
        merge,
        parse_linker_output,
        restrict_schema,
    )
    from bird.linking_eval import gold_columns, linking_metrics
    from bird.prompts import build_messages

    print(f"[orchestrator] preparing linker materials for split={split}, limit={limit or 'all'}")
    linker_msgs, schemas, metas = _prepare_for_linking.remote(split, limit, n_samples)

    print(f"[orchestrator] {len(linker_msgs)} questions; linker pass on {model}")
    inf = Inference(model_name=model)
    linker_outs = inf.chat.remote(linker_msgs, n=1, temperature=0.0, max_tokens=512)

    print("[orchestrator] parsing linker outputs + lexical augmentation + ensure_keys")
    selections: list[Selection] = []
    recalls: list[float] = []
    for raw, meta in zip(linker_outs, metas):
        text = raw[0] if raw else ""
        schema = schemas[meta["db_id"]]
        sel = parse_linker_output(text, schema)
        if use_lexical:
            ex_for_lex = BirdExample(
                question_id=meta["question_id"], db_id=meta["db_id"],
                question=meta["question"], evidence=meta["evidence"],
                sql="", difficulty=meta["difficulty"],
            )
            sel = merge(sel, lexical_link(ex_for_lex, schema))
        sel = ensure_keys(sel, schema)
        selections.append(sel)

        gold, _status = gold_columns(meta["gold_sql"], schema)
        recalls.append(linking_metrics(sel.columns, gold)["recall"])

    avg_recall = sum(recalls) / max(len(recalls), 1)
    perfect = sum(1 for r in recalls if r == 1.0)
    print(f"[linking] avg_recall = {avg_recall:.4f}  ({perfect}/{len(recalls)} perfect)")

    print("[orchestrator] building filtered-schema SQL prompts")
    gen_msgs: list[list[dict]] = []
    for meta, sel in zip(metas, selections):
        ex = BirdExample(
            question_id=meta["question_id"], db_id=meta["db_id"],
            question=meta["question"], evidence=meta["evidence"],
            sql="", difficulty=meta["difficulty"],
        )
        narrow = restrict_schema(schemas[meta["db_id"]], sel)
        gen_msgs.append(build_messages(ex, narrow, n_samples=n_samples))

    print(f"[orchestrator] {len(gen_msgs)} gen prompts; SQL generation pass")
    gen_outs = inf.chat.remote(gen_msgs, n=1, temperature=0.0, max_tokens=1024)

    predictions = []
    for meta, raw, sel, recall in zip(metas, gen_outs, selections, recalls):
        text = raw[0] if raw else ""
        sql = extract_sql(text)
        predictions.append({
            "question_id": meta["question_id"],
            "db_id": meta["db_id"],
            "difficulty": meta["difficulty"],
            "gold_sql": meta["gold_sql"],
            "predicted_sql": sql,
            "raw_completion": text,
            "linker_selection": sorted([list(p) for p in sel.columns]),
            "linking_recall": recall,
        })

    save_tag = save_as or f"linked-{Path(model).name}-{split}-{int(time.time())}.json"
    print("[orchestrator] running execution evaluator")
    summary = evaluate_predictions.remote(split=split, predictions=predictions, save_as=save_tag)
    summary["linking_recall_avg"] = avg_recall
    summary["linking_recall_perfect"] = perfect
    print(json.dumps(summary, indent=2))
    print(f"[done] saved as {save_tag} on bird-results volume")


@app.local_entrypoint()
def run_with_voting(
    split: str = "dev",
    limit: int = 0,
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    n_samples_schema: int = 3,
    n_votes: int = 8,
    temperature: float = 0.6,
    max_tokens: int = 1024,
    save_as: str = "",
    tensor_parallel_size: int = 1,
    max_model_len: int = 16384,
    vote_timeout_s: float = 15.0,
    eval_timeout_s: float = 30.0,
):
    """Self-consistency by majority-vote-on-execution.

    Per question, sample n_votes candidates at temperature>0, execute each on the
    SQLite DB, group by canonical result-set hash, and pick the SQL whose result
    is the most common among executable candidates. Falls back to the first
    candidate if every sample fails to execute.

    Tradeoffs vs. greedy baseline: ~n_votes x more inference cost; expected gain
    on Qwen2.5-Coder-7B in the 2-4 EX-point range, with the bulk coming from
    questions that are right-once-out-of-eight in the EXEC_ERROR/WRONG buckets.
    """
    print(f"[orchestrator] preparing prompts for split={split}, limit={limit or 'all'}")
    convos, examples_meta = _prepare_local.remote(split, limit, n_samples_schema)

    print(f"[orchestrator] {len(convos)} prompts; sampling n={n_votes} @ T={temperature} on {model}")
    inf = Inference(
        model_name=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )
    raw = inf.chat.remote(
        convos, n=n_votes, temperature=temperature, max_tokens=max_tokens,
    )

    # Extract SQL from each completion -> per-question candidate lists.
    candidate_lists: list[list[str]] = []
    raw_completions: list[list[str]] = []
    for gens in raw:
        gens = gens or []
        raw_completions.append(gens)
        candidate_lists.append([extract_sql(g) for g in gens])

    print(f"[orchestrator] running per-question vote (timeout={vote_timeout_s}s)")
    voted = vote_predictions.remote(
        split=split,
        candidate_lists=candidate_lists,
        metas=examples_meta,
        timeout_s=vote_timeout_s,
    )

    # Build the predictions list, attaching voting_metadata as a passthrough field.
    by_qid = {v["question_id"]: v for v in voted}
    predictions = []
    n_fallback = 0
    for meta, gens, cands in zip(examples_meta, raw_completions, candidate_lists):
        v = by_qid[meta["question_id"]]
        if v["voting_metadata"]["fallback_used"]:
            n_fallback += 1
        predictions.append({
            "question_id": meta["question_id"],
            "db_id": meta["db_id"],
            "difficulty": meta["difficulty"],
            "gold_sql": meta["gold_sql"],
            "predicted_sql": v["winner_sql"],
            "raw_completions": gens,
            "candidate_sqls": cands,
            "voting_metadata": v["voting_metadata"],
        })

    avg_executable = (
        sum(p["voting_metadata"]["n_executable"] for p in predictions)
        / max(len(predictions), 1)
    )
    avg_winner_count = (
        sum(p["voting_metadata"]["winner_count"] for p in predictions)
        / max(len(predictions), 1)
    )
    print(
        f"[vote] avg_executable={avg_executable:.2f}/{n_votes}  "
        f"avg_winner_count={avg_winner_count:.2f}  fallback={n_fallback}/{len(predictions)}"
    )

    save_tag = save_as or f"voting-{Path(model).name}-{split}-n{n_votes}-{int(time.time())}.json"
    print("[orchestrator] running execution evaluator")
    summary = evaluate_predictions.remote(
        split=split, predictions=predictions, save_as=save_tag, timeout_s=eval_timeout_s,
    )
    summary["voting"] = {
        "n_votes": n_votes,
        "temperature": temperature,
        "avg_executable": avg_executable,
        "avg_winner_count": avg_winner_count,
        "n_fallback": n_fallback,
    }
    print(json.dumps(summary, indent=2))
    print(f"[done] saved as {save_tag} on bird-results volume")
