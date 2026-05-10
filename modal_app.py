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


@app.function(
    image=cpu_image,
    volumes={BIRD_ROOT: bird_data},
    timeout=60 * 30,
)
def fix_train_layout() -> dict:
    """One-shot repair: BIRD train.zip extracted as `train/train/...` with an
    inner train_databases.zip and a __MACOSX dir. Flatten + extract.

    Idempotent: if `train/train.json` already exists at the top, no-op.
    """
    import shutil
    import zipfile

    root = Path(BIRD_ROOT) / "train"
    if (root / "train.json").exists():
        return {"status": "already_flat", "path": str(root)}

    inner = root / "train"
    if not (inner / "train.json").exists():
        return {"status": "no_inner_train_dir", "path": str(root), "items": [p.name for p in root.iterdir()]}

    # Move every file/dir from inner up to root.
    for child in inner.iterdir():
        target = root / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))
    inner.rmdir()

    # Drop macOS metadata.
    macosx = root / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx)

    # Unzip train_databases.zip if still zipped.
    inner_zip = root / "train_databases.zip"
    if inner_zip.exists():
        print(f"[fix_train_layout] extracting {inner_zip}")
        with zipfile.ZipFile(inner_zip) as zf:
            zf.extractall(root)
        inner_zip.unlink()

    bird_data.commit()
    n_dbs = len(list((root / "train_databases").iterdir())) if (root / "train_databases").exists() else 0
    with (root / "train.json").open() as f:
        n_examples = len(json.load(f))
    return {"status": "fixed", "path": str(root), "n_examples": n_examples, "n_databases": n_dbs}


# ============================================================
# Inference (GPU)
# ============================================================

@app.cls(
    image=gpu_image,
    gpu="B200",
    volumes={HF_HOME: hf_cache, BIRD_ROOT: bird_data},
    timeout=2 * 60 * 60,  # 2h: voting (n=8) on thinking models can exceed 1h
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
    max_tokens: int = 1024,
    linker_max_tokens: int = 512,
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
    linker_outs = inf.chat.remote(linker_msgs, n=1, temperature=0.0, max_tokens=linker_max_tokens)

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
    gen_outs = inf.chat.remote(gen_msgs, n=1, temperature=0.0, max_tokens=max_tokens)

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


# ============================================================
# Candidate execution (CPU, parallel processes) — voting+correction
# ============================================================

# Cap on rows we round-trip back from Pass-2 to the orchestrator. The hash is
# already-fixed-size (SHA-1 of canonical-sorted rows), so for voting we only
# need enough raw rows to (a) compute the hash and (b) decide is_degenerate.
# 200 rows × 8 candidates × 1534 questions ~= 2.5M tuples, well under any RPC
# limit even with wide schemas. Hashing is done on the worker, but we keep the
# (capped) raw rows on the wire so the orchestrator can re-hash after the
# correction-retry replaces a candidate (no second execute round-trip needed).
_RESULT_ROW_CAP = 200


def _exec_candidate_worker(job):
    """Module-level worker: execute one SQL on one DB.

    Returns {sql, results, error} where:
      - results is a row-capped projection of the raw rows (hashing is
        sort-based and stable under projection only when rows are LIMIT'd in
        SQL; here we project AFTER fetch so the hash WILL differ from a true
        full result. We accept that — the cap is generous enough that real
        BIRD answers fit, and questions whose result-set exceeds 200 rows are
        rare. Voting groups capped-vs-capped; correctness is judged elsewhere
        by `evaluate_predictions` with the FULL gold execution.)
    """
    import sqlite3
    from bird.eval import _execute

    sql, db_path, timeout_s, row_cap = job
    if not sql or not sql.strip():
        return {"sql": sql, "results": [], "error": "empty_sql"}
    try:
        rows = _execute(db_path, sql, timeout_s=timeout_s)
    except sqlite3.OperationalError as e:
        msg = str(e)
        # Distinguish timeout from other operational errors so the corrector
        # knows whether re-running with a different shape might help at all.
        kind = "timeout" if "interrupt" in msg.lower() else "operational"
        return {"sql": sql, "results": [], "error": f"{kind}: {msg}"}
    except Exception as e:
        return {"sql": sql, "results": [], "error": repr(e)}

    if row_cap and len(rows) > row_cap:
        rows = rows[:row_cap]
    # Convert each row to a tuple of primitives for JSON-serializability.
    return {"sql": sql, "results": [list(r) for r in rows], "error": None}


@app.function(
    image=cpu_image,
    volumes={BIRD_ROOT: bird_data},
    cpu=8,
    timeout=60 * 30,
)
def _execute_candidates(
    split: str,
    candidate_lists: list[list[str]],
    metas: list[dict],
    timeout_s: float = 15.0,
    workers: int = 8,
    row_cap: int = _RESULT_ROW_CAP,
) -> list[list[dict]]:
    """Execute every candidate on its question's DB; return per-task lists.

    Output shape: aligned with `candidate_lists`. Each inner list has the same
    length as the corresponding candidate list, with each entry a dict
    {sql, results, error}. Raw `results` are projected to `row_cap` rows.
    """
    import multiprocessing as mp

    split_root = Path(BIRD_ROOT) / split
    db_dir = split_root / f"{split}_databases"

    # Flatten with (task_idx, cand_idx) bookkeeping so we can scatter to a pool
    # and gather back into the per-task shape.
    flat_jobs: list[tuple[str, str, float, int]] = []
    flat_addr: list[tuple[int, int]] = []
    for ti, (cand_list, meta) in enumerate(zip(candidate_lists, metas)):
        db_path = str(db_dir / meta["db_id"] / f'{meta["db_id"]}.sqlite')
        for ci, sql in enumerate(cand_list):
            flat_jobs.append((sql, db_path, timeout_s, row_cap))
            flat_addr.append((ti, ci))

    t0 = time.time()
    if workers <= 1:
        flat_out = [_exec_candidate_worker(j) for j in flat_jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            flat_out = list(pool.imap(_exec_candidate_worker, flat_jobs, chunksize=8))
    print(f"[exec-cand] executed {len(flat_out)} candidates in {time.time() - t0:.1f}s")

    # Re-shape into per-task lists.
    by_task: list[list[dict]] = [[None] * len(c) for c in candidate_lists]  # type: ignore[list-item]
    for (ti, ci), out in zip(flat_addr, flat_out):
        by_task[ti][ci] = out
    return by_task


# ============================================================
# Voting + within-vote correction (orchestrator)
# ============================================================

@app.local_entrypoint()
def run_with_voting_correction(
    split: str = "dev",
    limit: int = 0,
    model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    n_samples_schema: int = 3,
    n_votes: int = 8,
    temperature: float = 0.7,
    retry_temperature: float = 0.0,
    max_tokens: int = 1024,
    exec_timeout_s: float = 15.0,
    save_as: str = "",
    tensor_parallel_size: int = 1,
    max_model_len: int = 16384,
    eval_timeout_s: float = 30.0,
):
    """Composed strategy: maj@k voting + within-vote self-correction retry.

    Four passes per question:

      1. Sample n_votes candidates at temperature>0.
      2. Execute every candidate on its SQLite DB (CPU pool, 15s timeout).
      3. For each (task, candidate) whose error is not None, build a *short*
         self-correct prompt (original user content + failed SQL + error;
         no full schema re-render) and retry at retry_temperature. Replace
         the failed candidate in place with the corrected one (if its execution
         is now successful OR if we couldn't get any better).
      4. Vote per task with `pick_winner` (degenerate-aware refinement: drop
         degenerate result-sets only if non-degenerate are the strict majority).

    Defaults target Qwen3-Coder-30B-A3B-Instruct (60.63% greedy baseline).
    Expected delta: roughly +1.0pp over voting alone, +1.5pp over correction
    alone (the strategies overlap on different failure modes).
    """
    # Local-only import: the orchestrator needs these to reshape candidates and
    # build retry messages. modal CLI's venv has these (they ship with `bird`).
    from bird.voting_correction import (
        build_self_correct_user_prompt, pick_winner,
    )

    print(f"[orchestrator] preparing prompts for split={split}, limit={limit or 'all'}")
    convos, examples_meta = _prepare_local.remote(split, limit, n_samples_schema)
    n_tasks = len(convos)

    print(f"[orchestrator] {n_tasks} prompts; sampling n={n_votes} @ T={temperature} on {model}")
    inf = Inference(
        model_name=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )

    # ---- Pass 1: sample n_votes candidates per question ----
    raw = inf.chat.remote(
        convos, n=n_votes, temperature=temperature, max_tokens=max_tokens,
    )
    raw_completions: list[list[str]] = [list(g or []) for g in raw]
    candidate_lists: list[list[str]] = [
        [extract_sql(g) for g in gens] for gens in raw_completions
    ]

    # ---- Pass 2: execute every candidate, get rows + error per cell ----
    print(f"[orchestrator] executing {n_tasks * n_votes} candidates "
          f"(timeout={exec_timeout_s}s)")
    all_candidates: list[list[dict]] = _execute_candidates.remote(
        split=split,
        candidate_lists=candidate_lists,
        metas=examples_meta,
        timeout_s=exec_timeout_s,
    )

    n_executable_pre = sum(
        1 for cands in all_candidates for c in cands if c["error"] is None
    )
    print(f"[exec-cand] {n_executable_pre}/{n_tasks * n_votes} candidates executable pre-retry")

    # ---- Pass 3: build retry prompts for failed cells; flatten for one chat call ----
    # Original user content is the LAST message of each convo (system+user only).
    user_contents = [c[-1]["content"] for c in convos]

    retry_addrs: list[tuple[int, int]] = []  # (task_idx, cand_idx)
    retry_msgs: list[list[dict]] = []
    for ti, cands in enumerate(all_candidates):
        for ci, c in enumerate(cands):
            if c["error"] is None:
                continue
            # Reuse the original system prompt; rebuild user with the failure context.
            sys_msg = convos[ti][0]
            new_user = build_self_correct_user_prompt(
                user_contents[ti], failed_sql=c["sql"], error=c["error"],
            )
            retry_msgs.append([
                sys_msg,
                {"role": "user", "content": new_user},
            ])
            retry_addrs.append((ti, ci))

    n_retried = len(retry_addrs)
    print(f"[orchestrator] {n_retried} candidate cells to retry @ T={retry_temperature}")

    n_retried_rescued = 0
    if n_retried:
        retry_raw = inf.chat.remote(
            retry_msgs, n=1, temperature=retry_temperature, max_tokens=max_tokens,
        )
        # Build per-task retry candidate lists for re-execution.
        retry_sql_per_task: dict[int, list[tuple[int, str]]] = {}
        for (ti, ci), gens in zip(retry_addrs, retry_raw):
            text = gens[0] if gens else ""
            new_sql = extract_sql(text)
            retry_sql_per_task.setdefault(ti, []).append((ci, new_sql))

        # Reassemble candidate_lists for re-execution: only the retried cells.
        retry_task_idxs = sorted(retry_sql_per_task.keys())
        retry_cand_lists: list[list[str]] = []
        retry_metas: list[dict] = []
        # Track which (cand_idx_in_subset, ti, ci_orig) so we can splice back.
        retry_addrs_by_subset: list[list[tuple[int, int]]] = []
        for ti in retry_task_idxs:
            pairs = retry_sql_per_task[ti]
            retry_cand_lists.append([sql for _, sql in pairs])
            retry_metas.append(examples_meta[ti])
            retry_addrs_by_subset.append([(ti, ci_orig) for ci_orig, _ in pairs])

        retried_exec = _execute_candidates.remote(
            split=split,
            candidate_lists=retry_cand_lists,
            metas=retry_metas,
            timeout_s=exec_timeout_s,
        )

        # Replace-in-place: only swap if the retry actually executes; otherwise
        # leave the original failed candidate so its error still feeds telemetry.
        for sub_list, addr_list in zip(retried_exec, retry_addrs_by_subset):
            for new_cand, (ti, ci) in zip(sub_list, addr_list):
                if new_cand["error"] is None:
                    all_candidates[ti][ci] = new_cand
                    n_retried_rescued += 1
                else:
                    # Keep original failure but stamp the retry SQL for audit.
                    # (We deliberately don't replace — a fresh failure isn't an
                    # improvement, and it could mask an originally-recoverable
                    # error if the retry produced an unrelated syntax error.)
                    pass

    n_executable_post = sum(
        1 for cands in all_candidates for c in cands if c["error"] is None
    )
    print(f"[correct] rescued {n_retried_rescued}/{n_retried} retried candidates")
    print(f"[exec-cand] {n_executable_post}/{n_tasks * n_votes} candidates executable post-retry")

    # ---- Pass 4: vote per task ----
    from collections import Counter as _Counter
    predictions: list[dict] = []
    outcome_counts: _Counter = _Counter()
    for meta, cands in zip(examples_meta, all_candidates):
        outcome = pick_winner(cands)
        outcome_counts[outcome["vote_outcome"]] += 1
        n_deg = outcome["n_degenerate"]
        n_exec_pre_q = sum(1 for c in cands if c["error"] is None)  # post-retry, but per-question
        predictions.append({
            "question_id": meta["question_id"],
            "db_id": meta["db_id"],
            "difficulty": meta["difficulty"],
            "gold_sql": meta["gold_sql"],
            "predicted_sql": outcome["winner_sql"],
            # Per-question telemetry as a passthrough field.
            "voting_correction_metadata": {
                "n_candidates": n_votes,
                "n_executable_pre_retry": None,  # filled below
                "n_executable_post_retry": n_exec_pre_q,
                "n_degenerate": n_deg,
                "n_retried": None,               # filled below
                "n_retried_rescued": None,       # filled below
                "vote_outcome": outcome["vote_outcome"],
                "winner_count": outcome["winner_count"],
                "n_distinct_results": outcome["n_distinct_results"],
                "fallback_used": outcome["fallback_used"],
                "degenerate_filter_applied": outcome["degenerate_filter_applied"],
            },
        })

    # Backfill the per-question pre-retry / retry counts. We tracked these
    # globally; recompute per-question against the original candidate states.
    # (We need to re-derive from candidate_lists vs all_candidates since
    # all_candidates has been replaced-in-place.)
    # Build the pre-retry exec map from a separate count pass on candidate_lists
    # would re-execute everything; instead, derive from the retry_addrs we stored.
    retried_per_task: dict[int, int] = {}
    rescued_per_task: dict[int, int] = {}
    if n_retried:
        # Need the same scope as the retry block above.
        for (ti, _ci) in retry_addrs:
            retried_per_task[ti] = retried_per_task.get(ti, 0) + 1
        # Recount rescued by checking for each retried (ti, ci): is it now error-free?
        # We don't have the retry subsets in scope here; inline a quick pass.
        # all_candidates[ti][ci].error is None AND (ti, ci) was in retry_addrs => rescued.
        retry_set = set(retry_addrs)
        for ti, ci in retry_set:
            if all_candidates[ti][ci]["error"] is None:
                rescued_per_task[ti] = rescued_per_task.get(ti, 0) + 1

    for ti, p in enumerate(predictions):
        meta = p["voting_correction_metadata"]
        post = meta["n_executable_post_retry"]
        n_ret_q = retried_per_task.get(ti, 0)
        n_resc_q = rescued_per_task.get(ti, 0)
        # Pre-retry executable = post-retry executable - rescued.
        meta["n_retried"] = n_ret_q
        meta["n_retried_rescued"] = n_resc_q
        meta["n_executable_pre_retry"] = post - n_resc_q

    print(f"[vote] outcomes: {dict(outcome_counts)}")

    save_tag = save_as or (
        f"voting-correction-{Path(model).name}-{split}-n{n_votes}-{int(time.time())}.json"
    )
    print("[orchestrator] running execution evaluator")
    summary = evaluate_predictions.remote(
        split=split, predictions=predictions, save_as=save_tag,
        timeout_s=eval_timeout_s,
    )
    summary["voting_correction"] = {
        "n_votes": n_votes,
        "temperature": temperature,
        "retry_temperature": retry_temperature,
        "n_executable_pre_retry": n_executable_pre,
        "n_executable_post_retry": n_executable_post,
        "n_retried": n_retried,
        "n_retried_rescued": n_retried_rescued,
        "vote_outcomes": dict(outcome_counts),
    }
    print(json.dumps(summary, indent=2))
    print(f"[done] saved as {save_tag} on bird-results volume")
