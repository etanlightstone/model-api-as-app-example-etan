"""Async task persistence + public shaping.

CRUD over the ``inference_tasks`` table plus the ``to_public`` projection that
renders the exact Domino async contract (``asyncPredictionId`` + status enum).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import uuid

from core import db
from services.tasks import storage

TERMINAL = {"succeeded", "failed", "cancelled", "expired"}

RETENTION_DAYS = int(os.environ.get("MODEL_APP_TASKS_RETENTION_DAYS", "7"))
USER_MAX_CONCURRENT = int(os.environ.get("MODEL_APP_TASKS_USER_MAX_CONCURRENT", "5"))


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def new_task_id() -> str:
    return "task_" + uuid.uuid4().hex


def _user_active_count(user_id: str) -> int:
    conn = db.get_conn()
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM inference_tasks WHERE user_id = ? AND status NOT IN "
        "('succeeded','failed','cancelled','expired')",
        (user_id,),
    ).fetchone()
    return int(row["c"])


class UserQuotaExceeded(Exception):
    pass


def create_task(
    *,
    slug: str,
    user_id: str,
    user_name: str,
    mode: str,
    records: list[dict] | None = None,
    input_file_ref: str | None = None,
    chunk_size: int = 32,
) -> str:
    """Create a queued task. ``mode`` is 'value' (inline records) or 'reference'."""
    if user_id and _user_active_count(user_id) >= USER_MAX_CONCURRENT:
        raise UserQuotaExceeded(
            f"You already have {USER_MAX_CONCURRENT} tasks in flight; wait for some to finish."
        )

    task_id = new_task_id()
    now = _now()
    expires = (_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(days=RETENTION_DAYS)).isoformat()
    out_path = storage.output_path(task_id)
    in_path = ""
    total = 0

    if mode == "value":
        records = records or []
        in_path = storage.input_path(task_id)
        storage.write_records(in_path, records)
        total = len(records)
    elif mode == "reference":
        # Resolve now to fail fast on a bad path; store the resolved path.
        in_path = storage.resolve_reference_path(input_file_ref or "")
        try:
            total = storage.count_lines(in_path) if in_path.endswith(".jsonl") else 0
        except Exception:
            total = 0
    else:
        raise ValueError(f"Unknown task mode: {mode}")

    storage.write_meta(task_id, {"slug": slug, "mode": mode, "created_at": now,
                                 "user_name": user_name})

    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO inference_tasks
                (id, slug, status, user_id, user_name, mode, input_file_path,
                 output_file_path, total_items, chunk_size, created_at, expires_at)
            VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, slug, user_id, user_name, mode, in_path, out_path,
             total, chunk_size, now, expires),
        )
    return task_id


def get_task(task_id: str):
    conn = db.get_conn()
    return conn.execute("SELECT * FROM inference_tasks WHERE id = ?", (task_id,)).fetchone()


def list_tasks(user_id: str | None = None, limit: int = 50) -> list:
    conn = db.get_conn()
    if user_id:
        rows = conn.execute(
            "SELECT * FROM inference_tasks WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM inference_tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return rows


def request_cancel(task_id: str):
    with db.transaction() as conn:
        row = conn.execute("SELECT status FROM inference_tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        if row["status"] not in TERMINAL:
            conn.execute(
                "UPDATE inference_tasks SET cancel_initiated_at = ? WHERE id = ?",
                (_now(), task_id),
            )
    return get_task(task_id)


def to_public(row) -> dict:
    """Render the Domino async poll contract."""
    status = row["status"]
    body: dict = {"asyncPredictionId": row["id"], "status": status}
    if status == "succeeded":
        if row["mode"] == "value" and row["result_json"]:
            body["result"] = json.loads(row["result_json"])
        else:
            body["result"] = {
                "output_file": row["output_file_path"],
                "completed_items": row["completed_items"],
            }
    elif status == "failed":
        body["errors"] = [row["error_message"] or "task failed"]
    elif status in ("queued", "running"):
        if row["total_items"]:
            body["progress"] = {
                "completed_items": row["completed_items"],
                "total_items": row["total_items"],
            }
    return body
