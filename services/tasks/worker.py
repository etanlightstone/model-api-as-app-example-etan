"""Single asyncio worker for the async submit/poll engine (Plan A).

Implements the async doc's hardened design: an **atomic lease claim** ([FIX 1])
so no two ticks double-run a task, a **timer-based heartbeat** ([FIX 2]) so a
long chunk isn't mistaken for a crash, and chunked, fsync-ordered checkpointing
so a restart resumes from the last durable chunk.

Two execution backends share the entire queue/resume layer:

* ``process`` (default) — a ``spawn`` ``ProcessPoolExecutor`` whose workers each
  load the model once (true parallelism, GIL sidestepped).
* ``thread`` — runs chunks in a thread against the in-process adapter. Cheaper
  to spin up; used when ``MODEL_APP_TASKS_BACKEND=thread`` (and by the tests).
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import datetime as _dt
import json
import logging
import multiprocessing as mp
import os
import shutil
import time
import uuid

from pathlib import Path

from core import config as config_mod
from core import db, settings, state
from services.tasks import service, storage
from services.tasks.runner import classify_chunk, init_worker

logger = logging.getLogger("model_app.tasks")
TERMINAL = service.TERMINAL

_WORKER_TOKEN = uuid.uuid4().hex
_pool: cf.ProcessPoolExecutor | None = None
_pool_signature: str = ""
_task: asyncio.Task | None = None
_stop: asyncio.Event | None = None
_wake: asyncio.Event | None = None
_in_flight: set[str] = set()
_last_tick: float = 0.0
_last_reap: float = 0.0
_last_error: str = ""


# --- knobs -------------------------------------------------------------------

def _enabled_flag() -> bool:
    return os.environ.get("MODEL_APP_TASKS_ENABLED", "1").lower() in ("1", "true", "yes")


def backend_name() -> str:
    return os.environ.get("MODEL_APP_TASKS_BACKEND", "process").lower()


def is_enabled() -> bool:
    """Tasks run only when enabled *and* a model is loaded and ready."""
    return _enabled_flag() and state.get_state().ready


def cpu_workers() -> int:
    val = int(os.environ.get("MODEL_APP_TASKS_CPU_WORKERS", "0") or 0)
    if val:
        return max(1, val)
    return max(1, (os.cpu_count() or 2) - 1)


def _max_concurrent() -> int:
    val = int(os.environ.get("MODEL_APP_TASKS_MAX_CONCURRENT", "0") or 0)
    return val if val > 0 else 2 * cpu_workers()  # [FIX 3]: above pool size


def _poll_seconds() -> int:
    return max(1, int(os.environ.get("MODEL_APP_TASKS_POLL_SECONDS", "2")))


def _heartbeat_seconds() -> int:
    return max(5, int(os.environ.get("MODEL_APP_TASKS_HEARTBEAT_SECONDS", "20")))


def _lease_miss_factor() -> int:
    return max(2, int(os.environ.get("MODEL_APP_TASKS_LEASE_MISS_FACTOR", "4")))


def _lease_stale_seconds() -> int:
    return _heartbeat_seconds() * _lease_miss_factor()


def _max_attempts() -> int:
    return int(os.environ.get("MODEL_APP_TASKS_MAX_ATTEMPTS", "3"))


def _reap_seconds() -> int:
    """How often the retention reaper sweeps (0 disables it)."""
    return max(0, int(os.environ.get("MODEL_APP_TASKS_REAP_SECONDS", "300") or 0))


def _chunk_timeout_seconds() -> float:
    """Per-chunk inference timeout (0 disables; generous default off)."""
    return max(0.0, float(os.environ.get("MODEL_APP_TASKS_CHUNK_TIMEOUT_SECONDS", "0") or 0))


# --- time helpers (fixed-width ISO so lexicographic compares are valid) ------

def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc)


def _iso(dt: _dt.datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _now_iso() -> str:
    return _iso(_utcnow())


# --- pool lifecycle ----------------------------------------------------------

def _config_signature(cfg: config_mod.ModelConfig) -> str:
    return json.dumps([cfg.source_type, cfg.params, cfg.slug], sort_keys=True, default=str)


def configure() -> None:
    """(Re)build the process pool to match the active config. Call on boot and
    whenever the owner changes the model."""
    global _pool, _pool_signature
    if backend_name() != "process" or not is_enabled():
        shutdown_pool()
        return
    cfg = config_mod.get_config()
    if cfg is None:
        shutdown_pool()
        return
    sig = _config_signature(cfg)
    if _pool is not None and sig == _pool_signature:
        return  # already matches
    shutdown_pool()
    ctx = mp.get_context("spawn")
    _pool = cf.ProcessPoolExecutor(
        max_workers=cpu_workers(),
        mp_context=ctx,
        initializer=init_worker,
        initargs=(cfg.source_type, cfg.params, cfg.display_name, cfg.slug,
                  int(os.environ.get("MODEL_APP_TASKS_TORCH_THREADS", "1"))),
    )
    _pool_signature = sig
    logger.info("task pool built (%d workers) for %s", cpu_workers(), cfg.slug)


def shutdown_pool() -> None:
    global _pool, _pool_signature
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
        _pool_signature = ""


# --- worker lifecycle --------------------------------------------------------

async def start_worker() -> None:
    global _task, _stop, _wake
    if _task is not None and not _task.done():
        return
    _stop = asyncio.Event()
    _wake = asyncio.Event()
    _task = asyncio.create_task(_run())
    logger.info("task worker started (backend=%s)", backend_name())


async def stop_worker() -> None:
    global _task
    if _stop:
        _stop.set()
    if _task:
        try:
            await asyncio.wait_for(_task, timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _task.cancel()
    _task = None


async def ensure_running() -> None:
    if _task is None or _task.done():
        await start_worker()


def nudge(_task_id: str | None = None) -> None:
    if _wake is not None:
        _wake.set()


# --- the loop ----------------------------------------------------------------

async def _run() -> None:
    global _last_tick, _last_error
    while _stop and not _stop.is_set():
        _last_tick = time.monotonic()
        try:
            if is_enabled():
                budget = _max_concurrent() - len(_in_flight)
                for tid in _load_due_ids():
                    if budget <= 0:
                        break
                    if tid in _in_flight:
                        continue
                    asyncio.create_task(_advance(tid))
                    budget -= 1
            _maybe_reap()
        except Exception as exc:  # noqa: BLE001
            logger.exception("task worker tick failed")
            _last_error = str(exc)
        # Wait for the poll interval or a nudge, whichever comes first.
        if _wake is not None:
            _wake.clear()
        try:
            await asyncio.wait_for(_wait_stop_or_wake(), timeout=_poll_seconds())
        except asyncio.TimeoutError:
            pass


async def _wait_stop_or_wake() -> None:
    waiters = [asyncio.create_task(_stop.wait())]  # type: ignore[union-attr]
    if _wake is not None:
        waiters.append(asyncio.create_task(_wake.wait()))
    done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
    for p in pending:
        p.cancel()


def _load_due_ids() -> list[str]:
    """Queued tasks + 'running' tasks whose lease went stale (orphan reclaim)."""
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT id, status, heartbeat_at FROM inference_tasks WHERE status NOT IN "
        "('succeeded','failed','cancelled','expired')"
    ).fetchall()
    stale_cut = _utcnow() - _dt.timedelta(seconds=_lease_stale_seconds())
    due = []
    for r in rows:
        if r["status"] == "queued":
            due.append(r["id"])
        elif r["status"] == "running":
            hb = r["heartbeat_at"]
            if not hb:
                due.append(r["id"])
            else:
                try:
                    if _dt.datetime.fromisoformat(hb) < stale_cut:
                        due.append(r["id"])
                except ValueError:
                    due.append(r["id"])
    return due


# --- retention reaper --------------------------------------------------------

def _under_tasks_dir(path: str) -> bool:
    """True only for blobs the app owns (the per-task dir under TASKS_DIR).

    By-reference ``.jsonl`` inputs keep ``input_file_path`` pointing at the
    caller's *own* dataset file — never ours to delete — so the reaper only
    touches files that live inside ``TASKS_DIR``.
    """
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(settings.TASKS_DIR.resolve())
        return True
    except (ValueError, OSError):
        return False


def _maybe_reap() -> None:
    """Run the retention sweep at most once per ``MODEL_APP_TASKS_REAP_SECONDS``."""
    global _last_reap
    interval = _reap_seconds()
    if not interval:
        return
    now_m = time.monotonic()
    if _last_reap and (now_m - _last_reap) < interval:
        return
    _last_reap = now_m
    try:
        _reap()
    except Exception:  # noqa: BLE001
        logger.exception("retention reaper sweep failed")


def _reap() -> None:
    """Make ``RETENTION_DAYS`` honest.

    1. Mark non-terminal tasks past ``expires_at`` as ``expired`` (covers tasks
       nothing ever claimed).
    2. Delete terminal tasks older than ``RETENTION_DAYS`` — both the row and
       the on-disk blobs we own — so rows and JSONL don't accumulate forever.
    """
    now = _utcnow()
    now_iso = _iso(now)
    conn = db.get_conn()

    # 1) Expire overdue, never-finished tasks.
    overdue = conn.execute(
        "SELECT id, expires_at FROM inference_tasks "
        "WHERE status NOT IN ('succeeded','failed','cancelled','expired') "
        "AND expires_at IS NOT NULL AND expires_at != ''"
    ).fetchall()
    expired = 0
    for r in overdue:
        try:
            if now <= _dt.datetime.fromisoformat(r["expires_at"]):
                continue
        except ValueError:
            continue
        with db.transaction() as c:
            cur = c.execute(
                "UPDATE inference_tasks SET status='expired', error_message=?, finished_at=? "
                "WHERE id=? AND status NOT IN ('succeeded','failed','cancelled','expired')",
                ("task expired before completion", now_iso, r["id"]),
            )
        expired += cur.rowcount

    # 2) Prune terminal tasks past the retention window (row + owned blobs).
    cutoff = _iso(now - _dt.timedelta(days=service.RETENTION_DAYS))
    old = conn.execute(
        "SELECT id, input_file_path, output_file_path FROM inference_tasks "
        "WHERE status IN ('succeeded','failed','cancelled','expired') "
        "AND COALESCE(NULLIF(finished_at,''), created_at) < ?",
        (cutoff,),
    ).fetchall()
    pruned = 0
    for r in old:
        for p in (r["input_file_path"], r["output_file_path"]):
            if not _under_tasks_dir(p):
                continue
            try:
                if os.path.isfile(p):
                    os.unlink(p)
            except OSError:
                pass  # best-effort
        # Drop the now-empty per-task dir (also clears meta.json) if it's ours.
        task_blob_dir = settings.TASKS_DIR / r["id"]
        try:
            if task_blob_dir.is_dir():
                shutil.rmtree(task_blob_dir, ignore_errors=True)
        except OSError:
            pass
        with db.transaction() as c:
            c.execute("DELETE FROM inference_tasks WHERE id=?", (r["id"],))
        pruned += 1

    if expired or pruned:
        logger.info("reaper: expired %d overdue, pruned %d retired task(s)", expired, pruned)


def _try_claim(task_id: str):
    """[FIX 1] Atomically transition a task to 'running' under our token."""
    stale_before = _iso(_utcnow() - _dt.timedelta(seconds=_lease_stale_seconds()))
    now = _now_iso()
    with db.transaction() as conn:
        cur = conn.execute(
            """
            UPDATE inference_tasks
               SET status='running', owner_token=?, claimed_at=?, heartbeat_at=?,
                   attempts = attempts + 1
             WHERE id = ?
               AND (status='queued'
                    OR (status='running' AND (heartbeat_at IS NULL OR heartbeat_at < ?)))
            """,
            (_WORKER_TOKEN, now, now, task_id, stale_before),
        )
        won = cur.rowcount == 1
    if not won:
        return None
    row = service.get_task(task_id)
    if row and not row["started_at"]:
        with db.transaction() as conn:
            conn.execute("UPDATE inference_tasks SET started_at=? WHERE id=?", (now, task_id))
        row = service.get_task(task_id)
    return row


def _finish(task_id: str, status: str, error: str = "", result_json: str = "") -> None:
    with db.transaction() as conn:
        conn.execute(
            "UPDATE inference_tasks SET status=?, error_message=?, result_json=?, finished_at=? WHERE id=?",
            (status, error, result_json, _now_iso(), task_id),
        )


def _cancel_requested(task_id: str) -> bool:
    row = service.get_task(task_id)
    return bool(row and row["cancel_initiated_at"])


def _expired(row) -> bool:
    exp = row["expires_at"]
    if not exp:
        return False
    try:
        return _utcnow() > _dt.datetime.fromisoformat(exp)
    except ValueError:
        return False


async def _advance(task_id: str) -> None:
    if task_id in _in_flight:
        return
    _in_flight.add(task_id)
    hb_task = None
    try:
        row = _try_claim(task_id)
        if row is None:
            return
        if _expired(row):
            _finish(task_id, "expired", "task expired before completion")
            return
        if row["attempts"] > _max_attempts():
            _finish(task_id, "failed", "exceeded max attempts (poison input?)")
            return

        hb_task = asyncio.create_task(_heartbeat_loop(task_id))
        loop = asyncio.get_running_loop()

        in_path = row["input_file_path"]
        out_path = row["output_file_path"]
        chunk_size = int(row["chunk_size"] or 32)
        cursor = int(row["cursor_bytes"] or 0)

        # By-reference inputs that aren't JSONL (e.g. CSV) are materialized to a
        # JSONL input file once, so the byte-cursor resume contract holds.
        if row["mode"] == "reference" and not in_path.endswith(".jsonl"):
            records = storage.load_reference_records(in_path)
            in_path = storage.input_path(task_id)
            storage.write_records(in_path, records)
            with db.transaction() as conn:
                conn.execute("UPDATE inference_tasks SET input_file_path=?, total_items=? WHERE id=?",
                             (in_path, len(records), task_id))

        window, window_end = [], cursor
        for offset, item in storage.iter_jsonl(in_path, start_byte=cursor):
            if not window and _cancel_requested(task_id):
                _finish(task_id, "cancelled", "cancelled by caller")
                return
            window.append(item)
            window_end = offset
            if len(window) >= chunk_size:
                await _flush(task_id, loop, window, window_end, out_path)
                window = []
        if window:
            await _flush(task_id, loop, window, window_end, out_path)

        # Assemble the inline result for by-value tasks (small, returned in poll).
        result_json = ""
        if row["mode"] == "value":
            outputs = storage.read_all(out_path)
            result = outputs[0] if (row["total_items"] == 1 and outputs) else outputs
            result_json = json.dumps(result, default=str)
        _finish(task_id, "succeeded", result_json=result_json)
    except Exception as exc:  # noqa: BLE001
        logger.exception("task %s failed", task_id)
        cur = service.get_task(task_id)
        if cur and cur["status"] not in TERMINAL:
            _finish(task_id, "failed", f"{type(exc).__name__}: {exc}")
    finally:
        if hb_task:
            hb_task.cancel()
        _in_flight.discard(task_id)


async def _heartbeat_loop(task_id: str) -> None:
    """[FIX 2] Refresh heartbeat_at on a fixed cadence while we own the lease."""
    interval = _heartbeat_seconds()
    try:
        while True:
            await asyncio.sleep(interval)
            with db.transaction() as conn:
                cur = conn.execute(
                    "UPDATE inference_tasks SET heartbeat_at=? WHERE id=? AND owner_token=? AND status='running'",
                    (_now_iso(), task_id, _WORKER_TOKEN),
                )
                if cur.rowcount != 1:
                    return
    except asyncio.CancelledError:
        pass


def _thread_classify(records: list[dict]) -> list[dict]:
    """Thread-backend compute path: validate + predict via the in-process adapter."""
    from core.predict_service import prepare_records

    adapter = state.get_adapter()
    if adapter is None:
        raise RuntimeError("No model loaded")
    return adapter.predict(prepare_records(adapter, records))


async def _flush(task_id, loop, items, end_offset, out_path) -> None:
    """Run one chunk off the loop, write results, advance the durable cursor."""
    if backend_name() == "process" and _pool is not None:
        awaitable = loop.run_in_executor(_pool, classify_chunk, items)
    else:
        awaitable = asyncio.to_thread(_thread_classify, items)

    timeout = _chunk_timeout_seconds()
    try:
        results = await (asyncio.wait_for(awaitable, timeout) if timeout else awaitable)
    except asyncio.TimeoutError:
        logger.warning("task %s chunk exceeded %gs timeout; killing compute", task_id, timeout)
        if backend_name() == "process":
            # A wedged pool worker can't be force-cancelled once running, so the
            # only reliable kill is to tear the pool down and rebuild it. This
            # also interrupts any *other* in-flight chunk on the pool; those
            # tasks fail this attempt and resume from their cursor on retry.
            shutdown_pool()
            configure()
        # Thread backend can't be force-killed: the chunk fails and the abandoned
        # thread runs to completion in the background. Re-raise either way so the
        # existing attempts/retry path in _advance takes over.
        raise TimeoutError(f"chunk exceeded {timeout:g}s timeout") from None

    for r in results:
        storage.append_jsonl_line(out_path, r)  # fsync before cursor bump
    with db.transaction() as conn:
        conn.execute(
            "UPDATE inference_tasks SET cursor_bytes=?, completed_items=completed_items+?, heartbeat_at=? WHERE id=?",
            (end_offset, len(results), _now_iso(), task_id),
        )


# --- observability -----------------------------------------------------------

def _queue_depth() -> dict:
    conn = db.get_conn()
    rows = conn.execute(
        "SELECT status, COUNT(*) AS c FROM inference_tasks GROUP BY status"
    ).fetchall()
    return {r["status"]: r["c"] for r in rows}


def worker_state() -> dict:
    return {
        "enabled": is_enabled(),
        "backend": backend_name(),
        "running": _task is not None and not _task.done(),
        "worker_token": _WORKER_TOKEN,
        "last_tick_age_s": round(time.monotonic() - _last_tick, 1) if _last_tick else None,
        "pool_workers": cpu_workers() if backend_name() == "process" else None,
        "pool_alive": _pool is not None,
        "max_concurrent": _max_concurrent(),
        "heartbeat_seconds": _heartbeat_seconds(),
        "lease_stale_s": _lease_stale_seconds(),
        "in_flight": sorted(_in_flight),
        "queue_depth": _queue_depth(),
        "last_error": _last_error,
    }
