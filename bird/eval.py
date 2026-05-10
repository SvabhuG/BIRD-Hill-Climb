"""Execution-accuracy evaluator for BIRD predictions.

BIRD's primary metric (EX) executes both predicted and gold SQL on the same SQLite
database and checks set-equality of the result rows. We mirror that, but also tag
each failure so per-bucket analysis is possible later (timeouts vs exec errors vs
wrong answers — the wedge between these tells you which scaffolding to add next).

Timeouts are enforced via `sqlite3.Connection.interrupt()` triggered by a Timer
thread; that's the only thread-safe way to abort an in-flight SQLite query and
cleanly handles runaway CTEs / cartesian explosions.
"""
from __future__ import annotations

import multiprocessing as mp
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


class EvalStatus(str, Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    EXEC_ERROR = "exec_error"   # SQL ran but raised (syntax, missing column, etc.)
    GOLD_ERROR = "gold_error"   # gold itself failed — corpus problem, count separately
    TIMEOUT = "timeout"
    EMPTY = "empty"             # no SQL produced at all


@dataclass
class EvalResult:
    question_id: int
    db_id: str
    difficulty: str | None
    status: EvalStatus
    error: str | None = None  # short error string for EXEC_ERROR/GOLD_ERROR/TIMEOUT
    predicted_sql: str = ""
    gold_sql: str = ""

    @property
    def correct(self) -> bool:
        return self.status is EvalStatus.CORRECT


@dataclass
class EvalSummary:
    n: int
    n_correct: int
    by_status: dict[str, int]
    by_difficulty: dict[str, dict[str, int]]  # difficulty -> {n, correct}

    @property
    def ex(self) -> float:
        return self.n_correct / self.n if self.n else 0.0


def _execute(db_path: str | Path, sql: str, timeout_s: float) -> list[tuple]:
    """Run `sql` against `db_path`, aborting after `timeout_s` wall seconds."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout_s)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b
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


def _rows_equal(a: Sequence[tuple], b: Sequence[tuple]) -> bool:
    """Set-equality on rows. Tuples of SQLite primitives are hashable."""
    if len(a) != len(b):
        return False
    try:
        return set(a) == set(b)
    except TypeError:
        # Defensive: BLOB or unusual type; fall back to multiset on repr.
        return Counter(map(repr, a)) == Counter(map(repr, b))


def evaluate_one(
    *,
    question_id: int,
    db_id: str,
    db_path: str | Path,
    predicted_sql: str,
    gold_sql: str,
    difficulty: str | None = None,
    timeout_s: float = 30.0,
) -> EvalResult:
    if not predicted_sql or not predicted_sql.strip():
        return EvalResult(
            question_id=question_id, db_id=db_id, difficulty=difficulty,
            status=EvalStatus.EMPTY, predicted_sql=predicted_sql, gold_sql=gold_sql,
        )

    # Gold first — if gold doesn't run, mark gold_error so it doesn't punish the model.
    try:
        gold_rows = _execute(db_path, gold_sql, timeout_s=timeout_s)
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        status = EvalStatus.TIMEOUT if "interrupt" in msg else EvalStatus.GOLD_ERROR
        return EvalResult(
            question_id=question_id, db_id=db_id, difficulty=difficulty,
            status=status, error=f"gold: {e}", predicted_sql=predicted_sql, gold_sql=gold_sql,
        )
    except Exception as e:  # pragma: no cover - unexpected DB issues
        return EvalResult(
            question_id=question_id, db_id=db_id, difficulty=difficulty,
            status=EvalStatus.GOLD_ERROR, error=f"gold: {e!r}",
            predicted_sql=predicted_sql, gold_sql=gold_sql,
        )

    try:
        pred_rows = _execute(db_path, predicted_sql, timeout_s=timeout_s)
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        status = EvalStatus.TIMEOUT if "interrupt" in msg else EvalStatus.EXEC_ERROR
        return EvalResult(
            question_id=question_id, db_id=db_id, difficulty=difficulty,
            status=status, error=str(e), predicted_sql=predicted_sql, gold_sql=gold_sql,
        )
    except Exception as e:
        return EvalResult(
            question_id=question_id, db_id=db_id, difficulty=difficulty,
            status=EvalStatus.EXEC_ERROR, error=repr(e),
            predicted_sql=predicted_sql, gold_sql=gold_sql,
        )

    status = EvalStatus.CORRECT if _rows_equal(pred_rows, gold_rows) else EvalStatus.WRONG
    return EvalResult(
        question_id=question_id, db_id=db_id, difficulty=difficulty,
        status=status, predicted_sql=predicted_sql, gold_sql=gold_sql,
    )


@dataclass
class _PredItem:
    question_id: int
    db_id: str
    db_path: str
    predicted_sql: str
    gold_sql: str
    difficulty: str | None
    timeout_s: float


def _worker(item: _PredItem) -> EvalResult:
    return evaluate_one(
        question_id=item.question_id,
        db_id=item.db_id,
        db_path=item.db_path,
        predicted_sql=item.predicted_sql,
        gold_sql=item.gold_sql,
        difficulty=item.difficulty,
        timeout_s=item.timeout_s,
    )


def evaluate_predictions(
    items: Iterable[_PredItem],
    workers: int = 8,
) -> list[EvalResult]:
    """Evaluate many predictions in parallel processes.

    Process pool (not threads) because SQLite holds the GIL during CPU-bound execution
    of the SQL plan, and we want true parallelism across cores when scoring 1500+ queries.
    """
    items = list(items)
    if workers <= 1:
        return [_worker(it) for it in items]
    with mp.get_context("spawn").Pool(workers) as pool:
        return list(pool.imap(_worker, items, chunksize=4))


def make_pred_item(
    *,
    question_id: int,
    db_id: str,
    db_path: str | Path,
    predicted_sql: str,
    gold_sql: str,
    difficulty: str | None = None,
    timeout_s: float = 30.0,
) -> _PredItem:
    return _PredItem(
        question_id=question_id, db_id=db_id, db_path=str(db_path),
        predicted_sql=predicted_sql, gold_sql=gold_sql, difficulty=difficulty,
        timeout_s=timeout_s,
    )


def summarize(results: Iterable[EvalResult]) -> EvalSummary:
    results = list(results)
    by_status: Counter[str] = Counter()
    by_diff: dict[str, dict[str, int]] = {}
    n_correct = 0
    for r in results:
        by_status[r.status.value] += 1
        if r.correct:
            n_correct += 1
        d = r.difficulty or "unknown"
        bucket = by_diff.setdefault(d, {"n": 0, "correct": 0})
        bucket["n"] += 1
        if r.correct:
            bucket["correct"] += 1
    return EvalSummary(
        n=len(results), n_correct=n_correct,
        by_status=dict(by_status), by_difficulty=by_diff,
    )


def format_summary(s: EvalSummary) -> str:
    lines = [f"EX = {s.ex:.4f}  ({s.n_correct}/{s.n})", ""]
    lines.append("by status:")
    for k, v in sorted(s.by_status.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:<12s} {v:>5d}  ({v / max(s.n, 1):.2%})")
    lines.append("")
    lines.append("by difficulty:")
    for d, b in sorted(s.by_difficulty.items()):
        ex = b["correct"] / b["n"] if b["n"] else 0.0
        lines.append(f"  {d:<12s} {b['correct']:>4d}/{b['n']:<4d}  ({ex:.2%})")
    return "\n".join(lines)
