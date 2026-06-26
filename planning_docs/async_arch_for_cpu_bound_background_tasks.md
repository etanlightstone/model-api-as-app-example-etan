# Async Architecture for CPU-Bound Background Tasks (a FastAPI Recipe)

A full, opinionated recipe for adding **CPU-heavy background jobs** (e.g. load a PyTorch or scikit-learn model and run classification over a list of inputs) to a FastAPI app that **must keep serving real-time requests concurrently** — under the same constraints the LLM Gateway already lives with: a single Domino app container, a Domino-dataset filesystem, SQLite as the only DB, no external broker, and the app can be restarted at any time.

It reuses the patterns we already proved out for batch inference (`services/batches/*`): **SQLite-as-queue**, **dataset-filesystem blobs**, a **single resumable asyncio worker**, **byte-cursor checkpointing**, **live-tunable concurrency caps**, and a **worker-state health snapshot**. The one genuinely new piece is the execution primitive: because the work is CPU-bound it must run **off the event loop** — either in **OS processes inside the app container** (Plan A) or in a **separate Domino Job container** (Plan B).

There are two backends, and they share the entire API/queue/storage/resume layer:

- **Plan A — in-process `ProcessPoolExecutor`** (§§1–13): best for sub-second, high-volume tasks that fit alongside live traffic. Most of this doc.
- **Plan B — Domino Jobs offload** (§A1): best for heavy/long/GPU/bursty tasks; the app container does *zero* inference so it can't contend with real-time endpoints at all. Structurally identical to the `passthrough` batch mode.

They can coexist behind one endpoint (an `execution_backend` column picks per task, just like `decide_execution_mode()` picks synthetic vs passthrough per batch).

> Read `batch_inference_findings.md` first — this doc assumes that mental model and only describes the deltas for the CPU-bound case.

> **Changelog vs. the first draft of this recipe.** Three correctness bugs in the original lease/concurrency design are fixed here and called out inline with **`[FIX n]`** markers, plus several honesty caveats are added (**`[CAVEAT]`**). Summary, so a reviewer who saw the first draft can diff quickly:
> 1. **`[FIX 1]` Atomic lease claim.** The original claimed a task by stamping the lease *after* an in-memory `_in_flight` check. `_in_flight` is per-process, so it can't prevent a duplicate claim across instances (the very future the lease exists for) and has a single-process hole when a long chunk lets the lease go stale. The claim is now a single conditional `UPDATE ... WHERE` that only one caller can win.
> 2. **`[FIX 2]` Heartbeat on its own timer.** The original only refreshed `heartbeat_at` once per chunk. A CPU chunk can run longer than `LEASE_STALE`, so a *healthy* task looked orphaned. A heartbeat is now bumped on an independent cadence, decoupled from chunk completion, and `LEASE_STALE` is derived from the chunk-duration ceiling rather than a flat constant.
> 3. **`[FIX 3]` Pool saturation vs. `max_concurrent`.** The original defaulted `tasks.max_concurrent` to pool size, but each task advances chunks *serially*, so the pool starved whenever any task was between chunks. `max_concurrent` now defaults *above* pool size, and the rationale is spelled out.
> Plus `[CAVEAT]`s on duplicate output rows, cancel latency, and the Plan B Jobs-API shape.

## 0. The GIL reality (why this design uses processes)

PyTorch and scikit-learn **only partially release the GIL** (PyTorch during ATen kernels; sklearn only for BLAS-backed estimators; tree ensembles and all the Python-level feature-prep glue hold it). "Basic classification" always has meaningful per-item Python that holds the GIL. Therefore:

- **Default execution = `ProcessPoolExecutor`** (true parallelism, GIL sidestepped, correct for any model).
- **Threads are an opt-in optimization** only for provably BLAS/ATen-bound estimators where you want to share one in-memory model copy. Not the default.

Everything below is built for the process-pool case.

## 1. Design goals

1. **Never block the event loop.** Real-time endpoints (`/v1/...`) must stay responsive while jobs run. CPU work happens in worker processes; the main process only orchestrates and does I/O.
2. **Survive restarts.** A job mid-flight when the Domino app is redeployed must resume (or cleanly re-run) on boot. State is durable in SQLite + the dataset.
3. **Bounded resource use.** Pool workers must not saturate every core and starve the event-loop process; concurrency is capped and live-tunable.
4. **Submit-and-poll API.** Same UX as the batch API: `POST` returns a `task_id` in milliseconds; the caller polls.
5. **Zero new infra.** SQLite + dataset filesystem only.

## 2. Architecture at a glance

```
  Real-time clients ─▶ FastAPI (async routes, event loop)  ◀── stays free, I/O only
                              │
        POST /v1/tasks        │  enqueue row in SQLite, write input blob to dataset
        GET  /v1/tasks/{id}   │  read row (poll)
                              ▼
                    services/tasks/worker.py   (single asyncio task in lifespan)
                              │  atomically claims due rows, submits to the pool, awaits futures,
                              │  writes results + checkpoints — all DB/file I/O on the loop side
                              ▼
            ProcessPoolExecutor (N = cpu_count - 1 worker processes)
              ├─ proc 1: model loaded ONCE via initializer ─▶ classify(chunk)
              ├─ proc 2: model loaded ONCE                  ─▶ classify(chunk)
              └─ proc N: ...                                 (pure compute, NO DB access)

  SQLite (tasks table)  ── queue + lifecycle + lease/heartbeat + cursor
  Dataset filesystem    ── <dataset>/llm_gateway/tasks/<id>/{input,output}.jsonl
```

The division of labor is the load-bearing rule:

- **Event-loop side (main process):** HTTP, SQLite reads/writes, dataset file reads/writes, submitting work, awaiting futures, checkpointing. All I/O, all fast.
- **Pool side (worker processes):** pure CPU. Loads the model, runs inference. **No SQLite handle, no dataset writes** (keeps the single-writer SQLite assumption intact and avoids fork-time handle corruption). Workers receive plain data, return plain data.

## 3. The execution primitive: ProcessPoolExecutor done right

Four expert details that make or break this:

### 3a. Use the `spawn` start method, not `fork`

A FastAPI process has a running event loop, threads, and open file/SQLite handles. `fork()` copies all of that into the child and routinely deadlocks with native thread pools (OpenMP/BLAS/torch). **Always create the pool with a `spawn` context.**

### 3b. Load the model **once per worker** via an initializer

Don't ship the model across the pickle boundary on every call (slow, and some models don't pickle). Load it once when each worker process starts, into a module global:

```python
# services/tasks/runner.py  — imported in the WORKER processes
import os
_MODEL = None

def init_worker(model_uri: str, torch_threads: int = 1):
    """Runs once per pool process at startup. Loads the model into a global."""
    # Pin intra-op threads BEFORE importing torch-heavy code to avoid
    # oversubscription (see 4b). Must be set as env for some BLAS backends.
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))
    global _MODEL
    _MODEL = load_model(model_uri)   # torch.load(...) / joblib.load(...)
    try:
        import torch
        torch.set_num_threads(torch_threads)
    except ImportError:
        pass

def classify_chunk(rows: list[dict]) -> list[dict]:
    """Pure compute. Runs in a worker process. NO DB, NO dataset writes."""
    out = []
    for r in rows:
        label, score = _predict(_MODEL, r["text"])   # the CPU-heavy part
        out.append({"custom_id": r["custom_id"], "label": label, "score": score})
    return out
```

### 3c. Create the pool once, in the lifespan

```python
# app.py lifespan
import concurrent.futures as cf
import multiprocessing as mp
from services.tasks import worker as task_worker
from services.tasks.runner import init_worker

@asynccontextmanager
async def lifespan(_: FastAPI):
    if task_worker.is_enabled():
        ctx = mp.get_context("spawn")
        pool = cf.ProcessPoolExecutor(
            max_workers=task_worker.cpu_workers(),       # see 4a
            mp_context=ctx,
            initializer=init_worker,
            initargs=(task_worker.model_uri(), task_worker.torch_threads()),
        )
        task_worker.set_pool(pool)
        await task_worker.start_worker()
    yield
    await task_worker.stop_worker()
    task_worker.shutdown_pool()                          # pool.shutdown(wait=True/cancel)
```

### 3d. Bridge the pool future into asyncio so the worker stays async

`loop.run_in_executor` turns a pool submission into an awaitable, so the worker's per-job coroutine looks identical to the I/O-bound batch worker — it `await`s and yields the loop while the CPU work runs in another process:

```python
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(pool, classify_chunk, chunk_rows)
```

That single `await` is the whole reason live requests stay responsive: the heavy work is in another process on another core, and the loop is parked.

## 4. CPU & concurrency sizing (the part that actually prevents starvation)

### 4a. Reserve a core for the event loop

The README mandates 2+ cores. Size the pool to `**cpu_count - 1**` so at least one core is always available to the main process for real-time traffic:

```python
def cpu_workers() -> int:
    val = gateway_settings.get_setting("tasks.cpu.max_workers", 0)  # 0 = auto
    n = int(val) if val else max(1, (os.cpu_count() or 2) - 1)
    return max(1, n)
```

Live-tunable like every other batch knob, so an admin can dial it down if job load is hurting p99 latency on the real-time endpoints.

### 4b. Avoid thread oversubscription

This is the silent killer. If you have `K` worker processes and each torch/BLAS call internally spawns `T` threads, you get `K × T` threads fighting over `C` cores → cache thrash and *worse* throughput than fewer threads. **Pin intra-op threads to 1 in each worker** (`torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`) and get your parallelism from the *number of processes*. Rule of thumb: `K = cores - 1`, `T = 1`. Only raise `T` if you run a single big job at a time and want intra-op parallelism instead.

### 4c. Three concurrency ceilings (mirroring the batch engine) — **`[FIX 3]`**

- `**tasks.cpu.max_workers**` — pool size; the hard CPU ceiling (`= cores − 1`).
- `**tasks.max_concurrent**` — how many tasks the worker advances at once. **Default = `2 × cpu_workers` (NOT `cpu_workers`).** See the saturation note below for why.
- `**tasks.user.max_concurrent**` — per-user cap enforced at enqueue (default 5), so one caller can't monopolize the queue. Same check as `service.create_batch`'s per-user cap.

**`[FIX 3]` Why `max_concurrent` must exceed pool size.** Each task advances its chunks **serially** inside `_advance` — it submits one chunk to the pool, `await`s it, writes results, checkpoints, *then* submits the next. So a single task keeps **at most one** pool worker busy at a time. If `max_concurrent == cpu_workers`, then in steady state the pool only stays full while *every* in-flight task happens to have a chunk executing — but the moment any task is between chunks (doing its on-loop DB commit + fsync), the pool worker it would have fed goes idle, and the core it was reserved for sits unused. The original draft defaulted `max_concurrent = cpu_workers` and therefore silently under-utilized the pool whenever any task was checkpointing.

The fix is to admit **more concurrent tasks than there are pool workers**, so there's always a chunk queued behind each worker to pick up the instant the previous one returns. `2 × cpu_workers` is a safe default (one task feeding each worker, one more ready to hand off). The pool's *internal* bounded queue is still the real backpressure — submitting more chunks than `max_workers` simply queues them inside the executor, which is fine and keeps memory O(workers + small backlog), not O(queued tasks). If you'd rather not over-admit tasks, the alternative is to keep `max_concurrent = cpu_workers` but have each task keep **2 chunks in flight** (submit N+1 before awaiting N); that's more code for the same effect, so the default here is the simpler "more tasks than workers" lever.

> Backpressure remains correct either way: if the pool is full, additional chunks wait in the executor's queue and additional tasks wait in `queued` — both visible to the caller via status.

### 4d. Memory

Each process holds its **own copy of the model**. `K` workers = `K ×` model RAM. For a 2 GB model and 4 workers that's 8 GB — right at the container limit. Mitigations: fewer workers, a smaller/quantized model, or (if the estimator is provably GIL-releasing) switch to a **thread pool** so all threads share one in-memory model. Document the model's RAM footprint next to `tasks.cpu.max_workers`.

## 5. SQLite as the queue (the "file DB")

One table, following the `BatchJob` pattern (indexed metadata only; blobs go on disk). The DB lives on the Domino dataset next to `llm_gateway.db`, so it's durable across restarts.

```python
class InferenceTask(Base):
    __tablename__ = "inference_tasks"
    id = Column(String, primary_key=True, default=new_id)     # "task_<uuid>"
    request_id = Column(String, nullable=False)
    kind = Column(String, nullable=False)                     # "classification"
    model_uri = Column(String, default="")

    # caller / governance attribution (same fields as UsageLog / BatchJob)
    user_id = Column(String, default=""); user_name = Column(String, default="")
    project_id = Column(String, default=""); org_id = Column(String, default="")

    # lifecycle — mirrors the batch vocabulary so a poller reads it verbatim
    #   queued → running → succeeded | failed | cancelled | expired
    status = Column(String, default="queued")
    error_message = Column(Text, default="")

    # blob pointers (dataset paths, not BLOBs in SQLite)
    input_file_path = Column(Text, default="")
    output_file_path = Column(Text, default="")

    # progress + chunked-resume cursor (reused straight from the batch engine)
    total_items = Column(Integer, default=0)
    completed_items = Column(Integer, default=0)
    failed_items = Column(Integer, default=0)
    cursor_bytes = Column(Integer, default=0)                 # resume offset into input.jsonl
    chunk_size = Column(Integer, default=32)                  # items per pool submission

    # LEASE / heartbeat — the key to safe restart-resume (see §8 and §9)
    #   owner_token identifies WHICH worker/process holds the claim, so a stale
    #   owner can be distinguished from the current one (supports [FIX 1]).
    owner_token = Column(String, default="")                  # uuid of the claiming worker run
    claimed_at = Column(DateTime, nullable=True)
    heartbeat_at = Column(DateTime, nullable=True)
    attempts = Column(Integer, default=0)

    # cancellation flag the worker checks at chunk boundaries
    cancel_initiated_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_task_status", "status"),                   # worker scan
        Index("ix_task_user", "user_id"),
        # Composite index supporting the atomic-claim predicate in [FIX 1].
        Index("ix_task_claim", "status", "heartbeat_at"),
    )
```

Why a lease/heartbeat (`owner_token`, `claimed_at`, `heartbeat_at`, `attempts`) when the batch engine didn't strictly need one? Because a CPU task has no external upstream id to resume against. The lease lets boot-time recovery distinguish "genuinely running" from "orphaned by a crash" (§8). It also future-proofs for the eventual Postgres/multi-instance world where the atomic claim becomes `SELECT ... FOR UPDATE SKIP LOCKED`.

## 6. Dataset filesystem layout

Identical to the batch engine — reuse `services/batches/storage.py` wholesale (the `append_jsonl_line` + `fsync`, `iter_jsonl` byte-offset reader, and `stream_request_body_to_file` are exactly what we need):

```
<dataset>/llm_gateway/
  tasks/
    <task_id>/
      input.jsonl     # caller-supplied items, immutable after enqueue
      output.jsonl    # appended chunk-by-chunk by the worker, fsynced
      meta.json       # debug snapshot at enqueue time
```

Large inputs stream to disk during the POST (never materialized in memory); results stream back out during polling. Same memory ceiling guarantees as batches.

> **`[CAVEAT]` `iter_jsonl` resume assumes an immutable, append-stable input file.** That holds here because `input.jsonl` is immutable after enqueue. Do **not** reuse this cursor reader against any file that's rewritten in place — the byte offset would point into garbage. (The output file *is* appended to, including duplicate replays — see §8's duplicate-row caveat — but nothing resumes off the output file, so that's fine.)

## 7. The FastAPI endpoints

A submit-and-poll surface. Routes are `async def` and do **only** fast I/O — they never touch the model.

```python
# routes/tasks.py
import asyncio
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from database import get_db
from routes.gateway import _resolve_caller
from services.tasks import service, worker as task_worker
from services.batches import storage   # reuse the proven I/O helpers

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])
TERMINAL = {"succeeded", "failed", "cancelled", "expired"}


def _require_enabled():
    if not task_worker.is_enabled():
        raise HTTPException(404, "Background tasks are not enabled")


@router.post("")
async def create_task(request: Request, file: UploadFile = File(...),
                      kind: str = "classification", model_uri: str = "",
                      db: Session = Depends(get_db)):
    """Enqueue a CPU job. Returns immediately with a task_id; work runs later."""
    _require_enabled()
    await task_worker.ensure_running()                 # self-heal the worker
    caller = await _resolve_caller(request, db)

    task_id = service.new_task_id()
    in_path = storage.input_path_for_task(task_id)     # <dataset>/.../tasks/<id>/input.jsonl
    # Stream the upload straight to disk, chunk by chunk — never in memory whole.
    total = 0
    with open(in_path, "wb") as f:
        while chunk := await file.read(65536):
            f.write(chunk)
            total += len(chunk)

    task = service.create_task(                         # validates + per-user cap + persist
        caller=caller, task_id=task_id, kind=kind,
        model_uri=model_uri or task_worker.model_uri(),
        input_file_path=in_path, total_bytes=total,
    )
    task_worker.nudge(task_id)                          # start without waiting for the poll tick
    return service.to_public(task)                      # {"id", "status": "queued", ...}


@router.get("/{task_id}")
async def get_task(task_id: str, request: Request, wait: bool = False,
                   timeout: int = 30, db: Session = Depends(get_db)):
    """Poll status. ?wait=true long-polls (capped at 60s) like the batch GET."""
    _require_enabled()
    caller = await _resolve_caller(request, db)
    task = service.get_task(task_id)
    if not task or not service.can_access(task, caller):
        raise HTTPException(404, "Task not found")
    if wait and task.status not in TERMINAL:
        deadline = min(max(timeout, 1), 60)
        waited = 0.0
        while waited < deadline and task.status not in TERMINAL:
            await asyncio.sleep(1.0)
            waited += 1.0
            task = service.get_task(task_id)
    return service.to_public(task)


@router.get("/{task_id}/result")
async def get_result(task_id: str, request: Request, db: Session = Depends(get_db)):
    """Stream the output JSONL once the task is terminal.

    [CAVEAT] At-least-once delivery: because a crash re-runs the last
    un-checkpointed chunk (§8), output.jsonl MAY contain more than one row for
    the same custom_id. The contract is that the CONSUMER dedups on custom_id,
    keeping the last occurrence. We do not dedup server-side because doing so
    would mean buffering or rewriting the whole output file (defeating the
    streaming + O(1)-memory guarantee). Callers reading this stream must treat
    custom_id as the idempotency key.
    """
    _require_enabled()
    caller = await _resolve_caller(request, db)
    task = service.get_task(task_id)
    if not task or not service.can_access(task, caller):
        raise HTTPException(404, "Task not found")
    if task.status not in TERMINAL:
        raise HTTPException(409, f"Task not finished (status={task.status})")
    return StreamingResponse(storage.stream_file_bytes(task.output_file_path),
                             media_type="application/x-ndjson")


@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str, request: Request, db: Session = Depends(get_db)):
    _require_enabled()
    caller = await _resolve_caller(request, db)
    task = service.request_cancel(task_id, caller=caller)   # stamps cancel flag, nudges worker
    if not task:
        raise HTTPException(404, "Task not found")
    return service.to_public(task)


@router.get("")
async def list_tasks(request: Request, db: Session = Depends(get_db)):
    _require_enabled()
    caller = await _resolve_caller(request, db)
    return {"tasks": [service.to_public(t)
                      for t in service.list_my_tasks(user_id=caller["user_id"])]}
```

The `POST` is the critical one: it does only disk-write + a DB insert and returns. No CPU. So even while 100 classification jobs churn in the pool, submitting the 101st is instant and the real-time endpoints are untouched.

## 8. The worker: single asyncio task, atomic lease claim, chunked checkpointing

This is the batch `worker.py` + `synthetic.py` design adapted for the pool. Same loop shape, same `_in_flight` guard, same live-tunable poll cadence, same `worker_state()` snapshot — but with an **atomic DB claim** (`[FIX 1]`) and a **timer-based heartbeat** (`[FIX 2]`) that the I/O-bound batch engine didn't need.

```python
# services/tasks/worker.py
import asyncio, logging, os, time, uuid
from datetime import datetime, timezone, timedelta
from sqlalchemy import update
from database import SessionLocal, InferenceTask, utcnow
from services import settings as gateway_settings
from services.batches import storage
from services.tasks.runner import classify_chunk

logger = logging.getLogger(__name__)
TERMINAL = {"succeeded", "failed", "cancelled", "expired"}

# This worker run's identity. Stamped into owner_token on claim so we can tell
# OUR live claim apart from a stale one left by a previous (crashed) run.
_WORKER_TOKEN = uuid.uuid4().hex

_pool = None
_task: asyncio.Task | None = None
_stop: asyncio.Event | None = None
_in_flight: set[str] = set()
_last_tick: float = 0.0
_last_error: str = ""


def set_pool(p):
    global _pool; _pool = p


def _poll_seconds() -> int:
    return max(1, int(gateway_settings.get_setting("tasks.worker.poll_seconds", 3) or 3))


def _max_concurrent() -> int:
    # [FIX 3] default is 2x pool size, not pool size — see §4c.
    default = 2 * cpu_workers()
    return max(1, int(gateway_settings.get_setting("tasks.max_concurrent", default) or default))


def _heartbeat_seconds() -> int:
    # [FIX 2] heartbeat cadence is independent of chunk completion.
    return max(5, int(gateway_settings.get_setting("tasks.worker.heartbeat_seconds", 20) or 20))


def _lease_stale() -> timedelta:
    # [FIX 2] LEASE_STALE is DERIVED from the heartbeat cadence, not a flat 120s.
    # A live task heartbeats every _heartbeat_seconds(); we declare it orphaned
    # only after it misses several beats. This decouples "is it alive" from
    # "how long is one chunk" — a chunk can now run for minutes without the
    # task being mistaken for a corpse, as long as the heartbeat timer ticks.
    misses = max(2, int(gateway_settings.get_setting("tasks.worker.lease_miss_factor", 4) or 4))
    return timedelta(seconds=_heartbeat_seconds() * misses)
```

### The scan: find due work (queued + genuinely-orphaned running)

```python
def _load_due_ids() -> list[str]:
    """Queued tasks, plus 'running' tasks whose lease went stale (orphan reclaim).

    NOTE: this is only a CANDIDATE scan. It does not claim anything — the claim
    is the atomic UPDATE in _try_claim (). Two ticks (or two instances) can both
    surface the same id here; only one will win the claim. [FIX 1]
    """
    db = SessionLocal()
    try:
        rows = db.query(InferenceTask.id, InferenceTask.status,
                        InferenceTask.heartbeat_at) \
                 .filter(InferenceTask.status.notin_(list(TERMINAL))).all()
        now = datetime.now(timezone.utc)
        stale = _lease_stale()
        due = []
        for tid, status, hb in rows:
            if status == "queued":
                due.append(tid)
            elif status == "running":
                hb = hb.replace(tzinfo=timezone.utc) if hb and hb.tzinfo is None else hb
                if not hb or (now - hb) > stale:      # missed enough heartbeats ⇒ orphan
                    due.append(tid)
        return due
    finally:
        db.close()
```

### `[FIX 1]` The atomic claim — the heart of the correctness fix

```python
def _try_claim(db, task_id: str) -> InferenceTask | None:
    """Atomically transition a task to 'running' under THIS worker's token.

    Returns the row if we won the claim, else None. This is a single conditional
    UPDATE so that — even with two ticks in one process, or two app instances on
    a future Postgres — exactly one caller can win. The WHERE clause matches
    only:
      * a still-queued task, OR
      * a 'running' task whose lease is provably stale (orphan reclaim).
    rowcount == 1 means we own it; rowcount == 0 means someone else did, or it
    moved to terminal, so we skip it. The in-memory _in_flight set is now just a
    fast-path dedupe WITHIN this process; the UPDATE is the real guard.
    """
    now = utcnow()
    stale_before = now - _lease_stale()
    stmt = (
        update(InferenceTask)
        .where(InferenceTask.id == task_id)
        .where(
            (InferenceTask.status == "queued")
            | (
                (InferenceTask.status == "running")
                & (InferenceTask.heartbeat_at < stale_before)
            )
        )
        .values(
            status="running",
            owner_token=_WORKER_TOKEN,
            claimed_at=now,
            heartbeat_at=now,
            started_at=InferenceTask.started_at,   # leave first-start untouched if set
            attempts=InferenceTask.attempts + 1,
        )
    )
    res = db.execute(stmt)
    db.commit()
    if res.rowcount != 1:
        return None                                # lost the race / already terminal
    row = db.query(InferenceTask).filter(InferenceTask.id == task_id).first()
    # set started_at only on the very first claim
    if row and row.started_at is None:
        row.started_at = now
        db.commit()
    return row
```

### The loop

```python
async def _run():
    global _last_tick, _last_error
    while _stop and not _stop.is_set():
        _last_tick = time.monotonic()
        try:
            if not _is_paused():
                budget = _max_concurrent() - len(_in_flight)
                for tid in _load_due_ids():
                    if budget <= 0:
                        break
                    if tid in _in_flight:            # fast path; real guard is _try_claim
                        continue
                    asyncio.create_task(_advance(tid))
                    budget -= 1
        except Exception as exc:
            logger.exception("task worker tick failed")
            _last_error = str(exc)
        try:
            await asyncio.wait_for(_stop.wait(), timeout=_poll_seconds())
        except asyncio.TimeoutError:
            pass
```

### Advancing one task: claim, then chunk with a live heartbeat

```python
async def _advance(task_id: str):
    if task_id in _in_flight:
        return
    _in_flight.add(task_id)
    db = SessionLocal()
    hb_task = None
    try:
        # ── [FIX 1] atomic claim. If we don't win it, bail immediately. ──
        row = _try_claim(db, task_id)
        if row is None:
            return
        if _expired(row):
            _finish(db, row, "expired"); return
        if row.attempts > int(gateway_settings.get_setting("tasks.max_attempts", 3)):
            _finish(db, row, "failed", "exceeded max attempts (poison input?)")
            return

        # ── [FIX 2] start an independent heartbeat that bumps heartbeat_at on a
        # timer, REGARDLESS of how long the current chunk runs. This is what lets
        # a multi-minute chunk avoid being declared orphaned. It only refreshes
        # the lease while WE still own the token. ──
        hb_task = asyncio.create_task(_heartbeat_loop(task_id))

        loop = asyncio.get_running_loop()
        cursor = int(row.cursor_bytes or 0)
        chunk_size = int(row.chunk_size or 32)
        out_path = row.output_file_path
        in_path = row.input_file_path

        window, window_end = [], cursor
        for offset, item in storage.iter_jsonl(in_path, start_byte=cursor):
            # cancellation checked at chunk boundaries (cheap PK lookup once/chunk)
            if not window and _cancel_requested(db, task_id):
                _finish(db, row, "cancelled"); return
            window.append(item); window_end = offset
            if len(window) >= chunk_size:
                await _flush(db, task_id, loop, window, window_end, out_path)
                window = []

        if window:
            await _flush(db, task_id, loop, window, window_end, out_path)
        _finish(db, _reload(db, task_id), "succeeded")
    except Exception as exc:
        logger.exception("task %s failed", task_id)
        row = _reload(db, task_id)
        if row and row.status not in TERMINAL:
            _finish(db, row, "failed", str(exc))
    finally:
        if hb_task:
            hb_task.cancel()
        _in_flight.discard(task_id)
        db.close()
```

### `[FIX 2]` The heartbeat loop

```python
async def _heartbeat_loop(task_id: str):
    """Refresh heartbeat_at on a fixed cadence while we still own the lease.

    Crucially this is NOT tied to chunk completion. Even if classify_chunk runs
    for several minutes, this keeps stamping heartbeat_at, so _load_due_ids on
    another tick/instance will NOT mistake the live task for an orphan. The
    owner_token guard means that if we somehow lost ownership (we shouldn't, but
    defensively), we stop refreshing rather than stomping a new owner's lease.
    """
    interval = _heartbeat_seconds()
    db = SessionLocal()
    try:
        while True:
            await asyncio.sleep(interval)
            stmt = (
                update(InferenceTask)
                .where(InferenceTask.id == task_id)
                .where(InferenceTask.owner_token == _WORKER_TOKEN)
                .where(InferenceTask.status == "running")
                .values(heartbeat_at=utcnow())
            )
            res = db.execute(stmt); db.commit()
            if res.rowcount != 1:
                return            # we no longer own it / it's terminal — stop
    except asyncio.CancelledError:
        pass
    finally:
        db.close()
```

### The chunk flush: fsync, THEN checkpoint

```python
async def _flush(db, task_id, loop, items, end_offset, out_path):
    """Run one chunk in a worker PROCESS, write results, advance the durable cursor."""
    # The ONLY line that touches the pool. Awaits → event loop is free meanwhile.
    results = await loop.run_in_executor(_pool, classify_chunk, items)
    for r in results:
        storage.append_jsonl_line(out_path, r)          # fsync before cursor bump
    row = db.query(InferenceTask).filter(InferenceTask.id == task_id).first()
    row.cursor_bytes = end_offset                        # checkpoint AFTER fsync
    row.completed_items = (row.completed_items or 0) + len(results)
    row.heartbeat_at = utcnow()                          # also refresh on progress
    db.commit()
```

Note the **fsync-then-checkpoint ordering** (same as `synthetic.py`): results hit disk and are fsynced (`storage.append_jsonl_line`) *before* `cursor_bytes` advances, so a crash mid-chunk re-runs that chunk on resume — at-least-once per item, with `custom_id` as the dedup key. (The heartbeat bump here is belt-and-suspenders on top of the `[FIX 2]` timer; either alone keeps the lease alive.)

## 9. Shutdown & resume on boot

Three layers cooperate:

1. **Graceful shutdown.** The lifespan calls `stop_worker()` (sets the stop event, lets in-flight chunks finish) then `pool.shutdown(wait=True)`. Clean redeploys drain naturally.
2. **Hard kill (the common Domino case).** The process just dies. In-flight tasks are left in `running` with a `heartbeat_at` that stops advancing and an `owner_token` belonging to the now-dead run.
3. **Boot recovery.** On the next start, `_WORKER_TOKEN` is a fresh uuid, so the dead run's token will never match the new heartbeat guard. `_load_due_ids()` surfaces any `running` task whose `heartbeat_at` is older than `_lease_stale()` as a candidate, and `_try_claim` atomically re-claims it under the *new* token. Because the cursor is durable and fsync-ordered, the task resumes from the **last completed chunk**, not from scratch. The pool is brand-new (fresh processes, model reloaded via the initializer), so there's nothing stale to clean up.

This is the "due purely from persisted columns" property the batch worker has, plus a lease so we can tell a live run from a corpse — and now (`[FIX 1]`/`[FIX 2]`) a claim that's atomic and a liveness signal that doesn't lie when a chunk runs long.

An `attempts` counter (checked in `_advance` right after the claim) fails a task that keeps crashing instead of looping forever:

```python
if row.attempts > int(gateway_settings.get_setting("tasks.max_attempts", 3)):
    _finish(db, row, "failed", "exceeded max attempts (poison input?)")
    return
```

> **`[CAVEAT]` Poison-pill with ProcessPoolExecutor.** If a worker process is killed mid-call (OOM, segfault in native code), the pool raises `BrokenProcessPool` and *all* in-flight futures fail — not just the one that hit the bad input. Handle it by marking the affected tasks for retry and **recreating the pool** (`shutdown_pool()` + rebuild); the `attempts` cap then quarantines the actual poison input after a few tries. Note the blast radius: one bad item can fail several *other* tasks' current chunks (they'll resume from their last checkpoint on retry, so no data is lost, but their latency takes a hit). This failure mode has no analog in the I/O-bound batch engine and must be handled explicitly. This is also a strong argument for Plan B when inputs are untrusted — a segfault there dies in its own container.

> **`[CAVEAT]` Duplicate output rows on resume.** Because resume re-runs the last un-checkpointed chunk, `output.jsonl` can contain repeated `custom_id`s. The result endpoint streams the file as-is and the contract pushes dedup to the consumer (§7). If your consumers can't dedup, you need a compaction pass before serving results — which costs the streaming/O(1)-memory property — so prefer fixing the consumer.

## 10. Why the real-time endpoints stay responsive

Put together, the guarantees are:

- **No CPU on the loop.** Every heavy operation is `await loop.run_in_executor(pool, ...)` — it runs in another process on another core, and the loop is parked, exactly like an I/O await.
- **A core is always reserved** (`cpu_workers = cores - 1`), so the OS scheduler always has somewhere to run the event-loop process.
- **Thread pinning** (`torch_threads=1`) stops `K×T` thread oversubscription from thrashing the reserved core.
- **Bounded fan-out** (`tasks.max_concurrent`, per-user cap) keeps the pool queue and memory O(workers + small backlog). Note `[FIX 3]`: `max_concurrent` is deliberately *above* pool size to keep workers fed, but the pool's own queue is what actually bounds memory — admitting more *tasks* doesn't admit more *concurrent chunks* than the executor allows.
- **The only loop-blocking work left** is the same small, fast I/O the batch engine already does on the loop: SQLite commits and fsync'd JSONL appends. These are micro-stalls, bounded in frequency by the concurrency caps, and identical to what we already run in production for batches. The extra heartbeat `UPDATE` (`[FIX 2]`) adds one tiny indexed write per task per ~20s — negligible.

> **`[CAVEAT]` The residual blocking I/O is real.** SQLite commits and `os.fsync` on the NFS-mounted dataset are synchronous on the loop. They're small relative to multi-second/-minute CPU chunks, and the concurrency caps bound their frequency, but they are micro-stalls on the shared loop. If profiling shows the fsync'd appends are non-trivial under heavy task load, move *just the file append* to `asyncio.to_thread` (file I/O releases the GIL), and keep DB writes on the loop to preserve single-writer semantics. Don't move the DB writes off the loop casually — that's how you reintroduce the multi-writer problem §13 warns about.

> **`[CAVEAT]` Cancel latency scales with chunk duration.** Cancellation is checked at chunk *boundaries* (§8), so "in-flight chunk finishes" can mean a multi-minute wait between a cancel request and the `cancelled` status when `chunk_size` is large and per-item cost is high — much longer than the batch engine's I/O-bound windows ever took. If responsive cancellation matters, lower `chunk_size` (faster boundaries, at the cost of more checkpoint overhead) or accept the tail and document it for callers. There is no safe way to interrupt a running `classify_chunk` mid-flight short of killing the worker process.

## 11. Observability & health

Reuse the `worker_state()` pattern verbatim — surface it at `GET /api/tasks/worker` and fold it into `GET /health?verbose=1`:

```python
def worker_state() -> dict:
    return {
        "running": _task is not None and not _task.done(),
        "paused": _is_paused(),
        "worker_token": _WORKER_TOKEN,
        "last_tick_age_s": round(time.monotonic() - _last_tick, 1),
        "pool_workers": cpu_workers(),
        "max_concurrent": _max_concurrent(),       # [FIX 3] surfaced so the 2x default is visible
        "heartbeat_seconds": _heartbeat_seconds(), # [FIX 2]
        "lease_stale_s": _lease_stale().total_seconds(),
        "in_flight_task_ids": sorted(_in_flight),
        "in_flight_count": len(_in_flight),
        "queue_depth": _queue_depth(),             # {queued, running, cancelling}
        "pool_broken": _pool_is_broken(),          # surfaced so admins see a dead pool
        "last_error": _last_error,
    }
```

Stalled-worker badge if `last_tick_age_s > 3 × poll_seconds`; a `severity=warning` audit row if the pool has been broken/rebuilding, or queue depth is climbing while throughput is zero (a stuck/poison job). Same alerting surface (the Audit view) as the backup-age and mirror-degraded watchdogs.

> **`[CAVEAT]` Watch for "running but no progress."** With the timer heartbeat (`[FIX 2]`), a wedged task can keep its lease fresh (heartbeat still ticking) while `completed_items` doesn't move — it looks alive but isn't progressing. Add an alert on `completed_items` stagnation over N heartbeats, not just on lease staleness, or `[FIX 2]` will mask the exact stall the original flat-`LEASE_STALE` would have caught by accident.

## 12. Settings knobs (live-tunable, no restart)

| Setting key                     | Default                  | What it does                                                                          |
| ------------------------------- | ------------------------ | ------------------------------------------------------------------------------------- |
| `tasks.enabled`                 | `false`                  | Master toggle (worker + routes). Off ⇒ routes 404.                                    |
| `tasks.worker.poll_seconds`     | `3`                      | Worker scan cadence.                                                                  |
| `tasks.worker.heartbeat_seconds`| `20`                     | **`[FIX 2]`** Lease-refresh cadence, independent of chunk completion.                 |
| `tasks.worker.lease_miss_factor`| `4`                      | **`[FIX 2]`** Missed heartbeats before a running task is declared orphaned. `LEASE_STALE = heartbeat_seconds × this`. |
| `tasks.worker.paused`           | `false`                  | Admin pause (finishes in-flight chunks, starts no new work).                          |
| `tasks.cpu.max_workers`         | `0` (auto = cores−1)     | Process-pool size. **Requires pool rebuild to apply** (see note).                     |
| `tasks.max_concurrent`          | **`2 × pool size`**      | **`[FIX 3]`** How many tasks advance at once — above pool size to keep workers fed.   |
| `tasks.user.max_concurrent`     | `5`                      | Per-user enqueue cap.                                                                 |
| `tasks.torch_threads`           | `1`                      | Intra-op threads per worker (anti-oversubscription). Pool rebuild to apply.           |
| `tasks.max_attempts`            | `3`                      | Re-run ceiling before a task is failed (poison-input guard).                          |
| `tasks.output.retention_days`   | `30`                     | Dataset cleanup horizon (reuse `scripts/purge_batch_files.py` pattern).               |

> Pool-shape knobs (`cpu.max_workers`, `torch_threads`) can't change a live `ProcessPoolExecutor` — applying them means draining and rebuilding the pool. Expose an admin "Rebuild pool" action (drain in-flight → `shutdown_pool()` → recreate) rather than pretending they're hot. The rest are read live every tick, exactly like the batch knobs.

## 13. Failure modes

| Scenario                                   | Behavior                                                                                                                                                           |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| App killed mid-task                        | Lease goes stale (fresh `_WORKER_TOKEN` on reboot can't match the dead one); boot recovery atomically re-claims; resumes from last fsync'd chunk. At-least-once per item (`custom_id` dedup). |
| Two ticks race for the same orphan         | **`[FIX 1]`** Both surface it in `_load_due_ids`, but the conditional `UPDATE` in `_try_claim` lets exactly one win (`rowcount==1`); the other gets `None` and skips. |
| Long chunk (minutes) on a healthy task     | **`[FIX 2]`** The timer heartbeat keeps `heartbeat_at` fresh independent of chunk completion, so the task is NOT mistaken for an orphan. |
| Worker process OOM / segfault              | `BrokenProcessPool`; in-flight futures fail (blast radius > 1 task); pool rebuilt; affected tasks retried up to `tasks.max_attempts` from their checkpoints.        |
| Poison input (repeatedly crashes a worker) | `attempts` exceeds cap ⇒ task `failed` with a clear message; pool stays healthy for everyone else.                                                                 |
| Caller cancels                             | Flag checked at chunk boundary; in-flight chunk finishes (latency tail scales with `chunk_size` — see §10 caveat); task `cancelled` with partial output kept.       |
| Task wedged but lease fresh                | **`[FIX 2]` side effect** — heartbeat hides it; caught instead by the `completed_items` stagnation alert (§11 caveat).                                              |
| Model RAM × workers exceeds container      | Worker init OOMs; surfaced via `pool_broken`; admin lowers `tasks.cpu.max_workers` and rebuilds.                                                                   |
| Two app instances on one dataset           | Still single-writer SQLite (unsupported for writes), BUT the atomic claim (`[FIX 1]`) + `owner_token` are forward-compatible with `SELECT ... FOR UPDATE SKIP LOCKED` once on Postgres — the claim logic won't need to change, only the lock primitive. |
| Dataset full                               | `append_jsonl_line` raises `OSError`; task `failed` with `storage_exhausted`.                                                                                       |

## 14. What we reuse vs. what's new

**Reused unchanged from `services/batches/`:** `storage.py` (fsync append, byte-cursor `iter_jsonl`, streaming reader, request-body streaming), the single-asyncio-worker loop shape, `_in_flight` dedup (now a fast-path only — see `[FIX 1]`), `nudge()`, live-tunable settings via `get_setting`, the `worker_state()` health snapshot, per-user concurrency cap, submit-and-poll + `?wait=` long-poll, dataset-blob + SQLite-metadata split, the cursor checkpoint + at-least-once resume contract.

**New for the CPU case (Plan A):** a `ProcessPoolExecutor` (spawn context, per-worker model initializer, thread pinning) created once in the lifespan; `await loop.run_in_executor(...)` as the dispatch primitive; the `cores − 1` sizing rule; an **atomic lease claim** (`[FIX 1]`, `owner_token` + conditional `UPDATE`); a **timer-based heartbeat** decoupled from chunk duration (`[FIX 2]`); `max_concurrent` defaulting **above** pool size for saturation (`[FIX 3]`); and explicit `BrokenProcessPool` / poison-pill handling with an `attempts` cap.

**Alternative backend (Plan B, §A1):** a Domino-Jobs wrapper modeled on `services/domino_scheduled_jobs.py` (same auth/host/tier-default logic); `execution_backend` + `domino_job_id` columns; and a worker branch that launches a job once then polls it — reusing the dataset as the cross-container IPC channel and `domino_job_id` as the resume anchor, exactly like passthrough batches poll `upstream_batch_id`.

## A1. Plan B — offload to Domino Jobs (separate compute, zero in-app contention)

Everything above keeps the CPU work **inside the app container** and fights for its cores. There's a structurally cleaner alternative for genuinely heavy work: don't run the model in the app at all — launch a **Domino Job** (a one-off run on its own hardware tier, in its own container) and have the gateway act purely as an **orchestrator + poller**. The app container then does *zero* CPU inference, so by construction it can't starve the real-time endpoints — there's no `ProcessPoolExecutor`, no GIL question, no `cores − 1` juggling, no `BrokenProcessPool`, no model-RAM-times-workers budget.

This is not a new pattern for us: it's **exactly the `passthrough` batch shape** (`create → poll an external id → materialize results`), except the "vendor" is a Domino Job instead of OpenAI. And we already call Domino's REST API this way today — `services/domino_scheduled_jobs.py` creates/updates/polls scheduled jobs using `get_admin_headers()` from `services/domino_auth.py`. Plan B just points the same machinery at the **one-off Jobs API** instead of the *scheduled*-jobs API.

### When to pick Plan B over the in-process pool (Plan A)

| Signal                           | Plan A (in-process `ProcessPoolExecutor`)         | Plan B (Domino Job offload)                         |
| -------------------------------- | ------------------------------------------------- | --------------------------------------------------- |
| Per-task duration                | ms → low seconds                                  | seconds → minutes/hours                             |
| Resource profile                 | fits in the app container alongside live traffic  | wants its own/bigger tier, or a **GPU**             |
| Startup overhead tolerable?      | must be near-zero (no job spin-up budget)         | job spin-up (~tens of seconds) is fine              |
| Isolation from real-time traffic | shares cores; needs `cores−1` + caps to stay safe | **physically isolated** — different container       |
| Throughput shape                 | many small jobs, steady                           | fewer big jobs, bursty                              |
| Crash blast radius               | a poison input can break a pool worker (and other in-flight chunks) | a bad job dies in its own container; app unaffected |
| Cost model                       | "free" (uses idle app cores)                      | spins up billable compute per job                   |
| Untrusted input                  | risky (segfault takes down a shared pool)         | **preferred** (blast radius is one container)       |

Rule of thumb: **sub-second, high-volume → Plan A. Heavy/long/GPU/bursty/untrusted → Plan B.** They can coexist — the same `inference_tasks` table and the same API can route a task to either backend based on an `execution_backend` column decided at enqueue (mirroring how `decide_execution_mode()` picks synthetic vs passthrough once per batch).

### The only thing that changes: the dispatch + a poll branch

The API surface (§7), the SQLite `inference_tasks` table (§5), the dataset blob layout (§6), the single asyncio worker loop (§8), the lease/heartbeat resume (§9), the per-user caps, and the health snapshot (§11) are **all identical**. The worker just gains a second code path: instead of submitting a chunk to a process pool, it launches a Domino Job once and then *polls the Domino job id* on subsequent ticks — the same way `passthrough.advance_passthrough` polls `upstream_batch_id`.

> **`[FIX 2]` note for Plan B:** the lease/heartbeat is simpler here — the *Domino Job's own status* is the liveness signal, so you poll `domino_job_id` instead of running the heartbeat timer. The atomic claim (`[FIX 1]`) still matters: it prevents two ticks from launching two jobs for the same task. Claim first, then launch.

Add two columns to the table for this backend:

```python
    execution_backend = Column(String, default="pool")   # "pool" | "domino_job"
    domino_job_id      = Column(String, default="")       # the launched run's id (the resume anchor)
```

A thin Jobs-API wrapper, modeled on `domino_scheduled_jobs.py` (same auth, same host, same hardware-tier-default lookup):

```python
# services/tasks/domino_jobs.py — sibling of services/domino_scheduled_jobs.py
import requests
from services.domino_auth import DOMINO_API_HOST, DOMINO_PROJECT_ID, get_admin_headers

def start_job(*, command: str, hardware_tier_id: str | None = None,
              title: str = "llm-gateway-task") -> dict:
    """Launch a one-off Domino Job. Returns the run object (carries the job id).

    Mirrors services/domino_scheduled_jobs.py: admin headers, project-scoped
    URL, project-default tier when unspecified.

    [CAVEAT] The exact one-off Jobs endpoint and request/response shape VARIES
    BY DOMINO VERSION and is the single biggest unknown in Plan B. Do not ship
    the path/body below as gospel — verify it against YOUR install's API guide
    (Admin > API docs, or the Jobs section of the Platform API reference) and
    pin to it, exactly as the scheduled-jobs wrapper was pinned. The /v4/jobs/*
    paths here are illustrative.
    """
    headers = {**get_admin_headers(), "Content-Type": "application/json"}
    url = f"{DOMINO_API_HOST.rstrip('/')}/v4/jobs/start"
    body = {
        "projectId": DOMINO_PROJECT_ID,
        "runCommand": command,            # e.g. "python scripts/run_task.py <task_id>"
        "title": title,
        # omit hardwareTierId to take the project default, like the
        # scheduled-jobs wrapper's _project_default_hardware_tier() does
    }
    if hardware_tier_id:
        body["hardwareTierId"] = hardware_tier_id
    r = requests.post(url, headers=headers, json=body, timeout=15)
    r.raise_for_status()
    return r.json()

def get_job(job_id: str) -> dict | None:
    headers = get_admin_headers()
    url = f"{DOMINO_API_HOST.rstrip('/')}/v4/jobs/{job_id}"
    r = requests.get(url, headers=headers, timeout=10)
    return r.json() if r.status_code == 200 else None

def stop_job(job_id: str) -> None:
    headers = {**get_admin_headers(), "Content-Type": "application/json"}
    url = f"{DOMINO_API_HOST.rstrip('/')}/v4/jobs/{job_id}/stop"
    requests.post(url, headers=headers, timeout=15)
```

The worker branch (replaces the pool `_flush` for `domino_job` tasks) — note it's `requests`-based blocking I/O, so wrap it in `asyncio.to_thread` to keep the loop free, since `requests` (unlike our pooled compute) isn't awaitable. The claim still happens first via `_try_claim` (`[FIX 1]`):

```python
async def _advance_domino_job(db, row):
    if not row.domino_job_id:
        # First tick (already claimed via _try_claim): launch the job once.
        # The job script reads input.jsonl from the dataset, writes output.jsonl
        # back, and exits.
        cmd = f"python {script_path('run_task.py')} {row.id}"
        job = await asyncio.to_thread(domino_jobs.start_job, command=cmd,
                                      title=f"task-{row.id}")
        row.domino_job_id = job["id"]; row.status = "running"
        row.started_at = utcnow(); row.heartbeat_at = utcnow()
        db.commit()
        return                                   # poll on the next tick

    # Subsequent ticks: poll Domino (bounded by tasks.domino.poll_seconds,
    # default 30 — be polite, same as batch.upstream.poll_seconds).
    state = await asyncio.to_thread(domino_jobs.get_job, row.domino_job_id)
    row.heartbeat_at = utcnow(); db.commit()
    status = (state or {}).get("status", "Running")
    if status in ("Succeeded", "Completed"):
        _finish(db, row, "succeeded")            # output.jsonl already on the dataset
    elif status in ("Failed", "Error"):
        _finish(db, row, "failed", (state or {}).get("statusMessage", "job failed"))
    elif status in ("Stopped", "Cancelled"):
        _finish(db, row, "cancelled")
    # else still running → do nothing, next tick polls again
```

### Why this reuses our tech so cleanly

- **The dataset *is* the IPC channel.** The Job container mounts the *same* Domino dataset the app uses, so the job reads `<dataset>/llm_gateway/tasks/<id>/input.jsonl` and writes `output.jsonl` right back to where the gateway's `GET /v1/tasks/{id}/result` already streams from. No new transport, no result-shipping API — the file DB pattern we built for batches doubles as cross-container handoff.
- **Resume is free.** `domino_job_id` is the durable anchor, exactly like `upstream_batch_id` for passthrough batches. App restart? The next worker tick just re-polls the still-running Domino job. The app being down doesn't even pause the work — the job runs independently and the app reconnects on boot.
- **The job script is just a `scripts/run_task.py`** that loads the model, reads the input JSONL, writes the output JSONL (fsync via the same `storage.append_jsonl_line`), and exits — structurally the same as our existing maintenance scripts, and command resolution can reuse the `script_command()` / frozen-shim logic in `domino_scheduled_jobs.py`.
- **Cancellation** maps to `domino_jobs.stop_job(domino_job_id)` (best-effort) then mark `cancelled` — same shape as passthrough's `_cancel_upstream`.

### Plan B tradeoffs to be honest about

- **`[CAVEAT]` Spin-up latency**: every job pays container startup (~tens of seconds), so it's wrong for sub-second tasks. Batch many items into one job to amortize.
- **`[CAVEAT]` Cost**: each job is billable compute, unlike Plan A's "free" use of idle app cores. Stamp the projected cost (like `batch_estimated_cost`) so admins see it before launching.
- **Concurrency is now governed by Domino**, not our pool: cap concurrent launched jobs with `tasks.domino.max_concurrent_jobs` and the per-user cap so we don't fork-bomb the cluster's job scheduler.
- **`[CAVEAT]` Auth dependency**: relies on the same admin credentials (`get_admin_headers()`) and `DOMINO_API_HOST`/`DOMINO_PROJECT_ID` the scheduled-jobs feature already needs — degrade gracefully (surface a clear error, don't crash the worker) when they're absent, exactly like `_ensure_configured()`.
- **`[CAVEAT]` No fine-grained progress** unless the job writes incremental output the app can `count_lines()` between polls — doable (the job appends chunk-by-chunk; the poller reads `completed_items` off the file), but it's coarser than Plan A's per-chunk cursor.
- **`[CAVEAT]` Jobs-API shape is version-dependent** — see the wrapper's inline caveat. Confirm the one-off Jobs endpoint against your install before building on it; it's the load-bearing unknown for this whole plan.

### New settings knobs for Plan B

| Setting key                        | Default                | What it does                                                                                                                       |
| ---------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `tasks.domino.poll_seconds`        | `30`                   | How often the worker polls Domino for job status (politeness, like `batch.upstream.poll_seconds`).                                 |
| `tasks.domino.max_concurrent_jobs` | `10`                   | Global cap on simultaneously-launched Domino Jobs.                                                                                 |
| `tasks.domino.hardware_tier_id`    | `""` (project default) | Tier to run task jobs on — e.g. a GPU tier. Falls back to the project default via the same lookup the scheduled-jobs wrapper uses. |
| `tasks.default_backend`            | `pool`                 | `pool` (Plan A) or `domino_job` (Plan B) when the caller doesn't specify.                                                          |

## 15. One-paragraph summary

Keep the FastAPI process doing exactly what it's good at — async I/O and orchestration — and push every CPU cycle **off the event loop**. For light, high-volume work that means **Plan A**: a `spawn`-based `ProcessPoolExecutor` whose workers each load the model once and run pure compute with intra-op threads pinned to 1, bridged via `run_in_executor` and sized to `cores − 1` so real-time traffic always has a core. The queue/resume layer is the batch engine's, hardened in three ways for the CPU case: an **atomic lease claim** so no two workers (or future instances) ever double-run a task (`[FIX 1]`), a **timer-based heartbeat** so a long-running chunk is never mistaken for a crash (`[FIX 2]`), and `max_concurrent` defaulted **above** pool size so workers stay fed instead of idling between a task's serial chunks (`[FIX 3]`). For heavy/long/GPU/untrusted work, **Plan B** is cleaner: launch a **Domino Job** on its own hardware tier and orchestrate it exactly like `passthrough` batches (claim → launch once → poll `domino_job_id` → results land on the shared dataset), so the app container does zero inference and physically can't contend with live endpoints. Both backends reuse the same battle-tested machinery (SQLite-as-queue, dataset JSONL blobs as storage *and* cross-container IPC, fsync'd byte-cursor checkpoints, lease/heartbeat resume, live concurrency caps, worker-state health) — only the dispatch primitive differs, and an `execution_backend` column lets one API route a task to whichever fits. Known sharp edges, all documented above: at-least-once delivery means duplicate `custom_id`s on resume (consumer dedups), cancel latency scales with `chunk_size`, a `BrokenProcessPool` blast radius spans more than the poison task, and the Plan B Jobs-API shape must be pinned to your Domino version.
