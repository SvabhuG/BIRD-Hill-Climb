"""Load BIRD examples and locate their SQLite databases.

The BIRD dev release lays out as:

    <root>/dev.json                       # list[BirdExample]
    <root>/dev_databases/<db_id>/<db_id>.sqlite
    <root>/dev_tables.json                # schema descriptions

Train follows the same shape under a `train/` prefix.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BirdExample:
    question_id: int
    db_id: str
    question: str
    evidence: str
    sql: str  # gold; absent on test split
    difficulty: str | None  # simple/moderate/challenging on dev; None on test

    @classmethod
    def from_json(cls, obj: dict) -> "BirdExample":
        return cls(
            question_id=int(obj["question_id"]),
            db_id=obj["db_id"],
            question=obj["question"],
            evidence=obj.get("evidence", "") or "",
            sql=obj.get("SQL", "") or "",
            difficulty=obj.get("difficulty"),
        )


@dataclass(frozen=True)
class BirdSplit:
    """A loaded split with paths resolved against a single root directory."""

    name: str  # "dev" | "train"
    root: Path
    examples: list[BirdExample]
    db_dir: Path  # contains <db_id>/<db_id>.sqlite

    def db_path(self, db_id: str) -> Path:
        return self.db_dir / db_id / f"{db_id}.sqlite"


def load_split(root: str | Path, name: str = "dev") -> BirdSplit:
    """Load a BIRD split from its unzipped directory.

    `root` should point at the directory containing `<name>.json` and
    `<name>_databases/`. We don't try to autodetect the BIRD release-dated
    folder name (e.g. `dev_20240627`) — the caller is expected to point at it.
    """
    root = Path(root)
    questions_path = root / f"{name}.json"
    if not questions_path.exists():
        raise FileNotFoundError(f"{questions_path} not found — point root at the unzipped {name} dir")

    with questions_path.open() as f:
        raw = json.load(f)

    examples = [BirdExample.from_json(o) for o in raw]
    db_dir = root / f"{name}_databases"
    if not db_dir.exists():
        raise FileNotFoundError(f"{db_dir} not found — expected `{name}_databases/` next to {name}.json")

    return BirdSplit(name=name, root=root, examples=examples, db_dir=db_dir)
