"""Few-shot retrieval from BIRD train.

At prompt-build time we pull `k` solved (question -> SQL) examples from the train
split and slot them into the user message before the actual question. The intent
is to give the model concrete demonstrations of:

  * how questions phrased over this DB tend to map to SQL,
  * BIRD's quirks (case-sensitive string matching, evidence consumption, etc.).

Retrieval is intentionally simple — token-overlap (Jaccard) on the lowercased
question. Same-`db_id` shots are preferred; we only fall back to cross-db when
the in-domain pool is empty (those typically share zero schema with the dev
question and are mostly useful as style examples).

Why lexical instead of dense embeddings:
  * No GPU dependency outside the inference container; this runs on the CPU
    prep function.
  * BIRD questions overlap heavily on entity/column tokens — Jaccard is
    competitive with sentence-transformers in our quick eyeball tests.
  * Reproducibility: deterministic, no model drift.

If we later want a dense retriever, swap `_score` for an embedding-cosine and
keep the rest of the API.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .data import BirdExample, load_split


# Tiny stopword set — we want to keep schema-related nouns and verbs, only strip
# function words that drown out signal in Jaccard overlap. NOT a linguistic
# stopword list; deliberately small.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "into",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "have", "has", "had",
    "and", "or", "but", "not",
    "this", "that", "these", "those",
    "you", "your", "i", "we", "he", "she", "it", "they",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "many", "much", "most", "more", "less", "least",
    "any", "all", "each", "some",
    "there", "their", "them",
    "s",  # possessive remnant after tokenization
})


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Lowercase, split on non-alphanumeric, drop tiny stopwords.

    Returns a set (not a list) — Jaccard is a set operation and we don't care
    about token frequency for this retrieval.
    """
    if not text:
        return set()
    raw = _TOKEN_RE.findall(text.lower())
    return {t for t in raw if t and t not in _STOPWORDS}


def _score(query_toks: set[str], doc_toks: set[str]) -> float:
    """Jaccard overlap. Returns 0.0 on empty intersection or empty doc."""
    if not query_toks or not doc_toks:
        return 0.0
    inter = len(query_toks & doc_toks)
    if inter == 0:
        return 0.0
    union = len(query_toks | doc_toks)
    return inter / union


@dataclass(frozen=True)
class TrainIndex:
    """Precomputed per-example tokens + db_id buckets for retrieval."""

    examples: list[BirdExample]
    by_db_id: dict[str, list[BirdExample]] = field(default_factory=dict)
    # Parallel to `examples`: tokenized question for each example.
    tokens: list[set[str]] = field(default_factory=list)
    # qid -> position in `examples`. Question IDs are unique within a split.
    _qid_to_idx: dict[int, int] = field(default_factory=dict)

    def tokens_for(self, ex: BirdExample) -> set[str]:
        idx = self._qid_to_idx.get(ex.question_id)
        if idx is None:
            return _tokenize(ex.question)
        return self.tokens[idx]


def load_train_index(train_root: str | Path) -> TrainIndex:
    """Load BIRD train split and build a retrieval index.

    `train_root` should be the directory containing `train.json` and
    `train_databases/` (mirrors `bird.data.load_split` semantics).
    """
    sp = load_split(train_root, name="train")
    return build_train_index(sp.examples)


def build_train_index(examples: list[BirdExample]) -> TrainIndex:
    """Build a TrainIndex from an in-memory list. Useful for tests."""
    by_db: dict[str, list[BirdExample]] = {}
    toks: list[set[str]] = []
    qid_to_idx: dict[int, int] = {}
    for i, ex in enumerate(examples):
        by_db.setdefault(ex.db_id, []).append(ex)
        toks.append(_tokenize(ex.question))
        qid_to_idx[ex.question_id] = i
    return TrainIndex(
        examples=list(examples),
        by_db_id=by_db,
        tokens=toks,
        _qid_to_idx=qid_to_idx,
    )


def _rank(
    query_toks: set[str],
    candidates: list[BirdExample],
    train: TrainIndex,
    exclude: set[int],
) -> list[BirdExample]:
    """Return candidates sorted by (score desc, question_id asc), excluding qids in `exclude`."""
    scored: list[tuple[float, int, BirdExample]] = []
    for ex in candidates:
        if ex.question_id in exclude:
            continue
        s = _score(query_toks, train.tokens_for(ex))
        scored.append((s, ex.question_id, ex))
    # Stable order: highest score first, qid ascending on ties (deterministic).
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [ex for _s, _q, ex in scored]


def retrieve(
    question: str,
    db_id: str,
    train: TrainIndex,
    k: int = 4,
) -> list[BirdExample]:
    """Pick up to `k` few-shot demos for a (question, db_id) pair.

    Strategy:
      1. Same-`db_id` pool, ranked by Jaccard overlap of the question; take top-k.
      2. If still short of k, fill from cross-db pool (excluding already chosen).

    Returns up to k examples; may return fewer if the train split is small or
    Jaccard scores are all zero (we still include zero-score shots — at worst
    they're stylistically useful).
    """
    if k <= 0 or not train.examples:
        return []

    qtoks = _tokenize(question)
    chosen: list[BirdExample] = []
    chosen_qids: set[int] = set()

    in_db = train.by_db_id.get(db_id, [])
    if in_db:
        ranked = _rank(qtoks, in_db, train, exclude=chosen_qids)
        for ex in ranked[:k]:
            chosen.append(ex)
            chosen_qids.add(ex.question_id)

    if len(chosen) < k:
        # Cross-db fallback. Skip same-db (already considered) for efficiency.
        cross_pool = [e for e in train.examples if e.db_id != db_id]
        if cross_pool:
            ranked = _rank(qtoks, cross_pool, train, exclude=chosen_qids)
            for ex in ranked:
                if len(chosen) >= k:
                    break
                chosen.append(ex)
                chosen_qids.add(ex.question_id)

    return chosen
