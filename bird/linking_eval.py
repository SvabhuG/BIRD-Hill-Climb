"""Linking-recall meter: extract the ground-truth (table, column) refs from a SQL
query so we can measure whether our schema-linking step kept everything the gold
SQL needed.

Why this matters: schema linking is unrecoverable. If we drop a column the gold
SQL uses, the downstream SQL generator literally cannot produce the right answer
no matter how good its reasoning. Linking recall is therefore a hard ceiling on
EX from the linking stage, and is way more diagnostic than EX alone — it lets
us answer "are we leaving points on the table at the linking step or the
generation step?"

Strategy:
  1. Parse the gold SQL with sqlglot.
  2. Run the `qualify` optimizer pass with the real DB schema, which:
       - Resolves aliases (`SELECT a.x FROM t AS a` → t.x)
       - Expands `SELECT *` into the underlying columns
       - Adds table qualifiers to bare column references
  3. Walk all `exp.Column` nodes and emit (table, column) tuples (lower-cased).

Robust to: parse failures (returns empty set + status), missing schema info,
quoted/bracketed identifiers, sqlite-specific syntax.
"""
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify

from .schema import DatabaseSchema


def _schema_for_qualify(db_schema: DatabaseSchema) -> dict[str, dict[str, str]]:
    """Convert our DatabaseSchema into the dict shape sqlglot's optimizer wants."""
    return {
        t.name: {c.name: (c.type or "TEXT") for c in t.columns}
        for t in db_schema.tables
    }


def gold_columns(sql: str, db_schema: DatabaseSchema) -> tuple[set[tuple[str, str]], str]:
    """Return ((table, column) refs, status).

    `status` is one of: "ok", "parse_error", "qualify_error". On error the column
    set is the best-effort fallback (all column refs we found, mapped to any table
    that has them — over-recalls, but never silently empty).
    """
    sql = sql.strip().rstrip(";")
    if not sql:
        return set(), "parse_error"

    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return _fallback_columns(sql, db_schema), "parse_error"

    schema_dict = _schema_for_qualify(db_schema)
    try:
        ast = qualify(
            ast,
            schema=schema_dict,
            dialect="sqlite",
            validate_qualify_columns=False,
            quote_identifiers=False,
            expand_stars=True,
        )
    except Exception:
        # qualify can be brittle on weird subqueries; fall back to a looser pass
        return _loose_columns(ast, db_schema), "qualify_error"

    cols = _loose_columns(ast, db_schema)
    return cols, "ok"


def _loose_columns(ast: exp.Expression, db_schema: DatabaseSchema) -> set[tuple[str, str]]:
    """Walk the AST, emit (table, column) refs with aliases resolved to real names.

    sqlglot's `qualify` pass resolves *which* alias each column belongs to, but
    leaves the alias name on the column rather than the underlying table — so a
    `SELECT a.x FROM artist AS a` shows up as Column(table='a', name='x'). We
    walk all Table nodes first to build {alias: real_table}, then translate.
    """
    alias_map: dict[str, str] = {}
    for tbl in ast.find_all(exp.Table):
        real = (tbl.name or "").lower()
        if not real:
            continue
        alias_map[real] = real
        alias = (tbl.alias or "").lower()
        if alias:
            alias_map[alias] = real

    name_to_tables: dict[str, list[str]] = {}
    for t in db_schema.tables:
        for c in t.columns:
            name_to_tables.setdefault(c.name.lower(), []).append(t.name.lower())

    out: set[tuple[str, str]] = set()
    for col in ast.find_all(exp.Column):
        cname = (col.name or "").lower()
        tname = (col.table or "").lower()
        if not cname:
            continue
        if tname:
            real = alias_map.get(tname, tname)
            out.add((real, cname))
        else:
            for t in name_to_tables.get(cname, []):
                out.add((t, cname))
    return out


def _fallback_columns(sql: str, db_schema: DatabaseSchema) -> set[tuple[str, str]]:
    """Last-resort regex-y fallback when sqlglot can't even parse. Just looks for
    bare column names mentioned in the text. Over-recalls; that's the point."""
    text = sql.lower()
    out: set[tuple[str, str]] = set()
    for t in db_schema.tables:
        for c in t.columns:
            if c.name.lower() in text:
                out.add((t.name.lower(), c.name.lower()))
    return out


def linking_metrics(
    selected: set[tuple[str, str]],
    gold: set[tuple[str, str]],
) -> dict:
    """Recall is the headline. Precision is informational (high precision can mean
    we're not over-selecting — but over-selection is fine for our purposes since
    it just adds prompt tokens, while under-selection caps EX)."""
    sel = {(t.lower(), c.lower()) for t, c in selected}
    gld = {(t.lower(), c.lower()) for t, c in gold}
    if not gld:
        return {"recall": 1.0, "precision": 0.0, "n_gold": 0, "n_selected": len(sel),
                "missed": [], "selected_only": sorted(sel)}
    hit = sel & gld
    missed = gld - sel
    extra = sel - gld
    return {
        "recall": len(hit) / len(gld),
        "precision": (len(hit) / len(sel)) if sel else 0.0,
        "n_gold": len(gld),
        "n_selected": len(sel),
        "missed": sorted(missed),
        "selected_only": sorted(extra),
    }
