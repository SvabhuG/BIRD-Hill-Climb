"""Render a SQLite database's schema for prompting.

Two surfaces:
  - `extract_schema` returns a structured `DatabaseSchema` (tables + columns + FKs + samples)
  - `render_ddl_with_samples` formats it into a model-friendly string

Phase-1 prompting just uses CREATE TABLE statements + a few sample rows per table.
Future scaffolding can extend this (M-Schema format, value indices, FK chains, etc.)
without touching prompts.py.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    notnull: bool
    pk: int  # 0 if not part of pk; 1+ for pk position
    default: str | None


@dataclass(frozen=True)
class ForeignKey:
    from_column: str
    to_table: str
    to_column: str


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: list[ColumnInfo]
    primary_key: list[str]
    foreign_keys: list[ForeignKey]
    create_sql: str  # the original CREATE TABLE from sqlite_master
    sample_rows: list[tuple] = field(default_factory=list)


@dataclass(frozen=True)
class DatabaseSchema:
    db_id: str
    tables: list[TableSchema]


def extract_schema(db_path: str | Path, db_id: str | None = None, n_samples: int = 3) -> DatabaseSchema:
    """Read tables/columns/FKs from a SQLite file.

    `n_samples=0` skips the row sampling pass, useful when callers only need DDL.
    """
    db_path = Path(db_path)
    db_id = db_id or db_path.stem
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        table_rows = cur.fetchall()

        tables: list[TableSchema] = []
        for tname, create_sql in table_rows:
            cols_info = cur.execute(f'PRAGMA table_info("{tname}")').fetchall()
            columns = [
                ColumnInfo(
                    name=row[1],
                    type=row[2] or "",
                    notnull=bool(row[3]),
                    pk=int(row[5]),
                    default=row[4],
                )
                for row in cols_info
            ]
            pk = [c.name for c in sorted([c for c in columns if c.pk > 0], key=lambda c: c.pk)]

            fk_info = cur.execute(f'PRAGMA foreign_key_list("{tname}")').fetchall()
            fks = [
                ForeignKey(from_column=row[3], to_table=row[2], to_column=row[4])
                for row in fk_info
            ]

            samples: list[tuple] = []
            if n_samples > 0:
                try:
                    samples = cur.execute(f'SELECT * FROM "{tname}" LIMIT {int(n_samples)}').fetchall()
                except sqlite3.Error:
                    samples = []

            tables.append(
                TableSchema(
                    name=tname,
                    columns=columns,
                    primary_key=pk,
                    foreign_keys=fks,
                    create_sql=(create_sql or "").strip(),
                    sample_rows=samples,
                )
            )
        return DatabaseSchema(db_id=db_id, tables=tables)
    finally:
        conn.close()


def _truncate(value: object, maxlen: int = 60) -> str:
    s = "NULL" if value is None else str(value)
    if len(s) > maxlen:
        return s[: maxlen - 1] + "…"
    return s


def render_ddl_with_samples(schema: DatabaseSchema, n_samples: int = 3) -> str:
    """Render schema as DDL blocks plus a tiny sample-rows table per relation."""
    parts: list[str] = []
    for t in schema.tables:
        parts.append(t.create_sql.rstrip(";") + ";")
        if n_samples and t.sample_rows:
            header = " | ".join(c.name for c in t.columns)
            rows_to_show = t.sample_rows[:n_samples]
            row_strs = [" | ".join(_truncate(v) for v in r) for r in rows_to_show]
            block = "\n".join([f"/* {len(rows_to_show)} sample rows from {t.name}:", header, *row_strs, "*/"])
            parts.append(block)
        parts.append("")  # blank line between tables
    return "\n".join(parts).rstrip() + "\n"
