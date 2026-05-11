"""Richer SQLite schema profiling — adds distinct-value lists, row counts,
and an explicit foreign-key section to the rendered prompt block.

Ported from the proven exp18 schema renderer. The v3 SFT retry showed that
prompt-format wrapping (chat-tags vs flat) is NOT what produces our high
exec_error tail; the remaining gap to exp18 is the *content* of the schema
block. Specifically: when the model sees

    -- `Charter School (Y/N)` distinct values: ['Y', 'N']

it learns to filter with `= 'Y'`, not guess `= 'Yes'`. Our previous schema
renderer (render_ddl_with_samples) showed only DDL + 3 sample rows, leaving
the model to *infer* valid filter vocabularies — which it got wrong often
enough to produce wrong-but-valid SQL or stray-token exec_errors.

Behavior of profile_database:
  * Per table: CREATE SQL, columns (name, type), FKs, sample rows (as dicts
    keyed by column name), row count, distinct-value lists for columns with
    ≤ max_distinct unique values (only on tables ≤ 100,000 rows — the
    cardinality scan is O(table_size) and unaffordable on huge tables).
  * Globally: list of all (from_table, from_column, to_table, to_column) FKs.

Behavior of format_profile:
  * Header: "Database: <db_name>\\n"
  * Per table: CREATE TABLE; -- N rows; -- Sample rows from `t`: ...;
    -- `col` distinct values: [...] (when applicable).
  * Trailing: "-- Foreign Key Relationships:" section.
"""
from __future__ import annotations

import os
import sqlite3


def profile_database(db_path: str, sample_rows: int = 3, max_distinct: int = 20) -> dict:
    """Return an enriched schema profile for a SQLite database.

    sample_rows: rows per table returned for the prompt's sample-row block.
    max_distinct: columns with N <= max_distinct unique values get a distinct-
                  values list rendered (skipped on tables with > 100k rows).
    """
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    table_names = [row[0] for row in cursor.fetchall()]
    tables = []
    all_foreign_keys: list[dict] = []
    for table_name in table_names:
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        row = cursor.fetchone()
        create_sql = row[0] if row else ""

        cursor.execute(f"PRAGMA table_info(`{table_name}`)")
        columns = [{"name": col[1], "type": col[2]} for col in cursor.fetchall()]

        cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
        foreign_keys = []
        for fk in cursor.fetchall():
            fk_dict = {
                "from_table": table_name,
                "from_column": fk[3],
                "to_table": fk[2],
                "to_column": fk[4],
            }
            foreign_keys.append(fk_dict)
            all_foreign_keys.append(fk_dict)

        try:
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT ?", (sample_rows,))
            col_names = [desc[0] for desc in cursor.description]
            samples = [dict(zip(col_names, row)) for row in cursor.fetchall()]
        except Exception:
            samples = []
            col_names = [c["name"] for c in columns]

        try:
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            row_count = cursor.fetchone()[0]
        except Exception:
            row_count = 0

        column_values: dict[str, list] = {}
        # Cardinality scan is O(row_count) per column. Skip it on huge tables.
        if row_count <= 100000:
            for col in columns:
                col_name = col["name"]
                try:
                    cursor.execute(
                        f"SELECT COUNT(DISTINCT `{col_name}`) FROM `{table_name}`"
                    )
                    n_distinct = cursor.fetchone()[0]
                    if 0 < n_distinct <= max_distinct:
                        cursor.execute(
                            f"SELECT DISTINCT `{col_name}` FROM `{table_name}` "
                            f"ORDER BY `{col_name}` LIMIT ?",
                            (max_distinct,),
                        )
                        column_values[col_name] = [r[0] for r in cursor.fetchall()]
                except Exception:
                    continue

        tables.append(
            {
                "name": table_name,
                "create_sql": create_sql,
                "columns": columns,
                "foreign_keys": foreign_keys,
                "sample_rows": samples,
                "column_names": col_names,
                "row_count": row_count,
                "column_values": column_values,
            }
        )
    conn.close()
    return {
        "db_path": db_path,
        "db_name": os.path.basename(os.path.dirname(db_path)),
        "tables": tables,
        "foreign_keys": all_foreign_keys,
    }


def format_profile(profile: dict) -> str:
    """Render an enriched profile as a model-friendly schema block.

    Output format (verbatim from the proven exp18 recipe):

        Database: <db_name>

        CREATE TABLE `<t>` (...);
        -- N rows
        -- Sample rows from `<t>`:
        --   v1, v2, v3
        --   v1, v2, v3
        -- `<col>` distinct values: [a, b, c, ...]    (when applicable)

        ...

        -- Foreign Key Relationships:
        --   `t1`.`c1` -> `t2`.`c2`
        ...
    """
    parts = [f"Database: {profile['db_name']}", ""]
    for table in profile["tables"]:
        parts.append(table["create_sql"] + ";")
        parts.append(f"-- {table['row_count']} rows")
        if table["sample_rows"]:
            parts.append(f"-- Sample rows from `{table['name']}`:")
            col_names = table["column_names"]
            for row in table["sample_rows"]:
                vals = [str(row.get(c, "NULL"))[:50] for c in col_names]
                parts.append(f"--   {', '.join(vals)}")
        if table["column_values"]:
            for col_name, values in table["column_values"].items():
                str_values = [str(v) if v is not None else "NULL" for v in values]
                vals_str = ", ".join(str_values[:15])
                if len(str_values) > 15:
                    vals_str += ", ..."
                parts.append(f"-- `{col_name}` distinct values: [{vals_str}]")
        parts.append("")  # blank line between tables
    if profile["foreign_keys"]:
        parts.append("-- Foreign Key Relationships:")
        for fk in profile["foreign_keys"]:
            parts.append(
                f"--   `{fk['from_table']}`.`{fk['from_column']}` -> "
                f"`{fk['to_table']}`.`{fk['to_column']}`"
            )
        parts.append("")
    return "\n".join(parts)
