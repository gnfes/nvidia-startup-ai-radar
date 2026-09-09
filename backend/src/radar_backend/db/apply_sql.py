"""Applies a .sql file to DATABASE_URL — for migrations/seeds too large to
paste comfortably into the Supabase SQL Editor (nvidia_kb_seed.sql is ~10MB
and can hang the web editor at that size). Works for any file in
backend/db/migrations or backend/db/seed, run in the order the README lists.

Usage: uv run python -m radar_backend.db.apply_sql db/seed/nvidia_kb_seed.sql
"""

from __future__ import annotations

import sys
from pathlib import Path

from radar_backend.db.session import get_connection


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run python -m radar_backend.db.apply_sql <path/to/file.sql>", file=sys.stderr)
        raise SystemExit(1)

    sql_path = Path(sys.argv[1])
    sql = sql_path.read_text(encoding="utf-8")

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
    print(f"Applied {sql_path}")


if __name__ == "__main__":
    main()
