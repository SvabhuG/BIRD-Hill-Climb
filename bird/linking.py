"""Schema linking — narrow the schema to the parts a question actually needs.

Three approaches, all producing a `Selection` (a set of (table, column) tuples,
case-normalized, validated against the real schema):

  * `lexical_link`  — deterministic, free at inference time. Tokens of the
                      question/evidence are matched against table & column names
                      and against any sampled cell values. High recall by design.

  * `build_linker_messages` + `parse_linker_output` — LLM-based linker. Compact
                      schema in, JSON `{"table": ["col", ...]}` out. One forward
                      pass per question; we pair this with the SQL-gen pass.

  * `merge` + `ensure_keys` — fuse multiple Selections (e.g. lexical ∪ LLM) and
                      force-include PK/FK columns so joins remain valid.

`restrict_schema` then takes a Selection and returns a filtered `DatabaseSchema`
that the existing `render_ddl_with_samples` knows how to render.

Linking recall (the *real* metric for this stage) lives in `linking_eval.py`.
This module is just about producing Selections; that one is about scoring them.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Iterable

from .data import BirdExample
from .schema import (
    ColumnInfo,
    DatabaseSchema,
    ForeignKey,
    TableSchema,
)


# ---------- Core data type ----------

@dataclass(frozen=True)
class Selection:
    """A (table, column) selection produced by a linker. Case-normalized."""
    columns: frozenset[tuple[str, str]]

    @property
    def tables(self) -> frozenset[str]:
        return frozenset(t for t, _ in self.columns)

    def __or__(self, other: "Selection") -> "Selection":
        return Selection(columns=self.columns | other.columns)

    @classmethod
    def empty(cls) -> "Selection":
        return cls(columns=frozenset())

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]]) -> "Selection":
        return cls(columns=frozenset((t.lower(), c.lower()) for t, c in pairs))


# ---------- Compact schema rendering for the linker prompt ----------

def compact_schema_for_linking(schema: DatabaseSchema) -> str:
    """One line per table: `table_name (col1, col2, col3 [FK->other], ...)`.

    Cheap on tokens — we don't include DDL, types, or sample rows here, because
    the linker only needs structural overview to decide what's relevant. The
    full DDL goes into the *generator* prompt downstream.
    """
    lines: list[str] = []
    for t in schema.tables:
        fk_by_col = {fk.from_column.lower(): fk for fk in t.foreign_keys}
        cols: list[str] = []
        for c in t.columns:
            tag = ""
            if c.pk:
                tag += " [PK]"
            if c.name.lower() in fk_by_col:
                fk = fk_by_col[c.name.lower()]
                tag += f" [FK->{fk.to_table}.{fk.to_column}]"
            cols.append(f"{c.name}{tag}")
        lines.append(f"{t.name} ({', '.join(cols)})")
    return "\n".join(lines)


# ---------- LLM-based linker ----------

LINKER_SYSTEM = (
    "You are a database expert. Given a database schema and a user question, "
    "you identify which tables and columns are needed to answer the question with SQL. "
    "Include foreign-key columns required for joins, even when they aren't named in the question. "
    "Be inclusive — over-selecting wastes tokens, but missing a column makes the query impossible."
)

LINKER_USER_TEMPLATE = """\
### Database tables and columns
{compact_schema}

### Hint / external knowledge
{evidence}

### Question
{question}

### Output
Output a single JSON object on one line, mapping each relevant table to a list of \
relevant column names. Use the EXACT names shown in the schema above. Example:
{{"users": ["id", "email"], "orders": ["user_id", "total"]}}
Output only the JSON, nothing else."""


def build_linker_messages(example: BirdExample, schema: DatabaseSchema) -> list[dict]:
    user = LINKER_USER_TEMPLATE.format(
        compact_schema=compact_schema_for_linking(schema),
        evidence=example.evidence.strip() or "(none provided)",
        question=example.question.strip(),
    )
    return [
        {"role": "system", "content": LINKER_SYSTEM},
        {"role": "user", "content": user},
    ]


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_DOTTED_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")


def parse_linker_output(text: str, schema: DatabaseSchema) -> Selection:
    """Parse the LLM's response into a Selection, validated against the schema.

    Tries (in order):
      1. JSON object {table: [cols]}
      2. Loose `table.column` pattern matches
      3. Bare column-name tokens that uniquely belong to one table
    Returns a Selection containing only (table, column) pairs that exist in the
    real schema. Unknown names are silently dropped — the linker may hallucinate;
    we don't propagate hallucinations into the SQL generator's view.
    """
    valid: dict[str, set[str]] = {
        t.name.lower(): {c.name.lower() for c in t.columns} for t in schema.tables
    }

    selected: set[tuple[str, str]] = set()

    # Attempt 1: JSON parse
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                for tname, cols in obj.items():
                    tkey = str(tname).lower()
                    if tkey in valid and isinstance(cols, list):
                        for c in cols:
                            ckey = str(c).lower()
                            if ckey in valid[tkey]:
                                selected.add((tkey, ckey))
        except json.JSONDecodeError:
            pass

    # Attempt 2: dotted-name fallback (always run to catch anything missed by JSON)
    for tname, cname in _DOTTED_RE.findall(text):
        tkey, ckey = tname.lower(), cname.lower()
        if tkey in valid and ckey in valid[tkey]:
            selected.add((tkey, ckey))

    return Selection(columns=frozenset(selected))


# ---------- Lexical linker ----------

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(s)}


def _split_identifier(name: str) -> set[str]:
    """`first_name` -> {'first', 'name'}; `firstName` -> {'first', 'name'}."""
    snake_parts = name.split("_")
    out: set[str] = set()
    for part in snake_parts:
        if not part:
            continue
        # camelCase / PascalCase
        camel = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", part)
        out.update(c.lower() for c in camel if c)
        out.add(part.lower())
    return out


def lexical_link(
    example: BirdExample,
    schema: DatabaseSchema,
    *,
    min_score: float = 0.5,
) -> Selection:
    """Score each (table, column) by lexical overlap with question + evidence.

    Conservative — designed for high recall. We *don't* filter aggressively here;
    the LLM linker can be more selective. This pass exists to catch the obvious
    cases for free and to give us a ground floor when the LLM whiffs.
    """
    q_tokens = _tokens(example.question) | _tokens(example.evidence)
    if not q_tokens:
        return Selection.empty()

    selected: set[tuple[str, str]] = set()
    for t in schema.tables:
        table_token_match = bool(_tokens(t.name) & q_tokens)
        # Score each column
        for c in t.columns:
            col_tokens = _split_identifier(c.name)
            col_match = col_tokens & q_tokens
            score = 0.0
            if c.name.lower() in example.question.lower():
                score += 2.0
            elif col_match:
                score += 1.0 * len(col_match) / max(len(col_tokens), 1)
            if c.name.lower() in example.evidence.lower():
                score += 1.5
            if table_token_match:
                score += 0.3
            if score >= min_score:
                selected.add((t.name.lower(), c.name.lower()))

        # Cell-value matching: if any sampled value appears in the question,
        # link the column it came from. This catches "Find rows where status = 'X'".
        if t.sample_rows:
            qtext = (example.question + " " + example.evidence).lower()
            for row in t.sample_rows:
                for col, val in zip(t.columns, row):
                    if val is None:
                        continue
                    s = str(val).lower()
                    if len(s) >= 3 and s in qtext:
                        selected.add((t.name.lower(), col.name.lower()))

    return Selection(columns=frozenset(selected))


# ---------- Combinators ----------

def merge(*selections: Selection) -> Selection:
    out: frozenset[tuple[str, str]] = frozenset()
    for s in selections:
        out = out | s.columns
    return Selection(columns=out)


def ensure_keys(selection: Selection, schema: DatabaseSchema) -> Selection:
    """Force-include PK + FK columns of every selected table.

    Without this we routinely strip out join columns that the SQL generator needs
    but that the linker didn't think to mention. This is the single biggest
    recall fix vs. a naive linker.
    """
    if not selection.columns:
        return selection

    by_name = {t.name.lower(): t for t in schema.tables}
    extra: set[tuple[str, str]] = set()
    for tname in selection.tables:
        t = by_name.get(tname)
        if not t:
            continue
        for c in t.columns:
            if c.pk:
                extra.add((tname, c.name.lower()))
        for fk in t.foreign_keys:
            if fk.from_column:
                extra.add((tname, fk.from_column.lower()))
            # Also include the target side so the join is fully usable. SQLite's
            # PRAGMA foreign_key_list returns NULL for the target column when the
            # FK references the target's implicit PK; in that case fall back to the
            # target table's primary-key column.
            target = fk.to_table.lower() if fk.to_table else None
            if target and target in by_name:
                target_col = (fk.to_column or "").lower()
                if not target_col:
                    target_pk = by_name[target].primary_key
                    target_col = target_pk[0].lower() if target_pk else ""
                if target_col:
                    extra.add((target, target_col))
    return Selection(columns=selection.columns | frozenset(extra))


# ---------- Schema restriction (for downstream rendering) ----------

def restrict_schema(schema: DatabaseSchema, selection: Selection) -> DatabaseSchema:
    """Project the schema down to the selected tables/columns.

    The returned DatabaseSchema is a real one — `render_ddl_with_samples` works
    on it unchanged, and FK relationships are pruned to only point at retained
    columns. Sample rows are projected to the surviving columns.

    If `selection` is empty, returns `schema` unchanged (safer than empty
    schema, which would produce nonsense prompts).
    """
    if not selection.columns:
        return schema

    keep_by_table: dict[str, set[str]] = {}
    for tname, cname in selection.columns:
        keep_by_table.setdefault(tname, set()).add(cname)

    new_tables: list[TableSchema] = []
    for t in schema.tables:
        keep = keep_by_table.get(t.name.lower())
        if not keep:
            continue

        kept_cols = [c for c in t.columns if c.name.lower() in keep]
        if not kept_cols:
            continue

        kept_names = {c.name.lower() for c in kept_cols}
        kept_pk = [n for n in t.primary_key if n.lower() in kept_names]
        kept_fks = [
            fk for fk in t.foreign_keys
            if fk.from_column and fk.from_column.lower() in kept_names
            and fk.to_table and fk.to_table.lower() in keep_by_table
            and fk.to_column
            and fk.to_column.lower() in keep_by_table[fk.to_table.lower()]
        ]

        # Project sample rows to surviving columns.
        col_idx = [i for i, c in enumerate(t.columns) if c.name.lower() in keep]
        kept_samples = [tuple(row[i] for i in col_idx) for row in t.sample_rows]

        # Re-render a minimal CREATE TABLE so the prompt's DDL only shows kept cols.
        # CRITICAL: identifiers must be backtick-quoted because BIRD column names
        # routinely contain spaces and parens (e.g., `Free Meal Count (K-12)`).
        # An earlier version emitted them raw, producing invalid DDL that the model
        # "fixed" by inventing munged names (FreeMealCountK12), causing exec_errors.
        col_defs = [_column_def(c) for c in kept_cols]
        if kept_pk:
            col_defs.append(f"PRIMARY KEY ({', '.join(_quote(p) for p in kept_pk)})")
        for fk in kept_fks:
            col_defs.append(
                f"FOREIGN KEY ({_quote(fk.from_column)}) "
                f"REFERENCES {_quote(fk.to_table)}({_quote(fk.to_column)})"
            )
        ddl = f"CREATE TABLE {_quote(t.name)} (\n  " + ",\n  ".join(col_defs) + "\n);"

        new_tables.append(TableSchema(
            name=t.name,
            columns=kept_cols,
            primary_key=kept_pk,
            foreign_keys=kept_fks,
            create_sql=ddl,
            sample_rows=kept_samples,
        ))

    if not new_tables:
        return schema  # nothing matched — fall back to full schema rather than empty
    return DatabaseSchema(db_id=schema.db_id, tables=new_tables)


def _quote(name: str) -> str:
    """Backtick-quote a SQLite identifier. Always — backticks are safe on
    plain names too, and BIRD column names routinely contain spaces/parens
    that *must* be quoted (e.g., `Free Meal Count (K-12)`)."""
    return f"`{name}`"


def _column_def(c: ColumnInfo) -> str:
    parts = [_quote(c.name), c.type or "TEXT"]
    if c.notnull:
        parts.append("NOT NULL")
    if c.default is not None:
        parts.append(f"DEFAULT {c.default}")
    return " ".join(parts)
