"""SQLite access — the app's only database.

Two tables share one file on the Domino dataset:

* ``app_config`` — a single row naming the active model (Phase 1).
* ``inference_tasks`` — the async submit/poll queue (Phase 6), modelled on the
  lease/heartbeat schema in ``async_arch_for_cpu_bound_background_tasks.md``.

We use the stdlib ``sqlite3`` module directly rather than an ORM: there is one
writer (the app process), the schema is tiny, and it keeps the dependency
surface minimal — in the spirit of the plan's "don't over-engineer" constraint.
WAL mode lets the async poll reads run concurrently with the worker's writes.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

from core import settings

_local = threading.local()


def _connect() -> sqlite3.Connection:
    settings.ensure_dirs()
    conn = sqlite3.connect(
        settings.DB_PATH,
        timeout=30.0,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_conn() -> sqlite3.Connection:
    """A per-thread connection (SQLite connections aren't thread-safe to share)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Commit on success, roll back on error."""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


SCHEMA = """
CREATE TABLE IF NOT EXISTS app_config (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    source_type  TEXT NOT NULL,            -- 'registry' | 'custom_function'
    params_json  TEXT NOT NULL,            -- source-specific config
    display_name TEXT NOT NULL DEFAULT '',
    slug         TEXT NOT NULL DEFAULT '',
    updated_at   TEXT NOT NULL DEFAULT '',
    updated_by   TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS inference_tasks (
    id                  TEXT PRIMARY KEY,
    slug                TEXT NOT NULL DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'queued',   -- queued|running|succeeded|failed|cancelled|expired
    error_message       TEXT NOT NULL DEFAULT '',

    -- caller attribution (from Domino app headers / token)
    user_id             TEXT NOT NULL DEFAULT '',
    user_name           TEXT NOT NULL DEFAULT '',

    -- request shape: by-value (inline JSON) or by-reference (input_file)
    mode                TEXT NOT NULL DEFAULT 'value',    -- 'value' | 'reference'
    input_file_path     TEXT NOT NULL DEFAULT '',
    output_file_path    TEXT NOT NULL DEFAULT '',
    result_json         TEXT NOT NULL DEFAULT '',         -- inline result for by-value tasks

    -- progress + chunked-resume cursor (by-reference only)
    total_items         INTEGER NOT NULL DEFAULT 0,
    completed_items     INTEGER NOT NULL DEFAULT 0,
    cursor_bytes        INTEGER NOT NULL DEFAULT 0,
    chunk_size          INTEGER NOT NULL DEFAULT 32,

    -- lease / heartbeat (see async doc §5, [FIX 1]/[FIX 2])
    owner_token         TEXT NOT NULL DEFAULT '',
    claimed_at          TEXT,
    heartbeat_at        TEXT,
    attempts            INTEGER NOT NULL DEFAULT 0,
    cancel_initiated_at TEXT,

    created_at          TEXT NOT NULL DEFAULT '',
    started_at          TEXT,
    finished_at         TEXT,
    expires_at          TEXT
);

CREATE INDEX IF NOT EXISTS ix_task_status ON inference_tasks (status);
CREATE INDEX IF NOT EXISTS ix_task_user   ON inference_tasks (user_id);
CREATE INDEX IF NOT EXISTS ix_task_claim  ON inference_tasks (status, heartbeat_at);
"""


def init_db() -> None:
    """Create tables/indexes if absent. Safe to call on every boot."""
    with transaction() as conn:
        conn.executescript(SCHEMA)


def reset_for_tests() -> None:
    """Drop the per-thread connection (used by the test harness between cases)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
