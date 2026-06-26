# How Async Inference Works

A walkthrough, for the technically curious, of what actually happens when you
submit an async prediction to this app — from the HTTP request, through the
queue and worker, into the model, and back out as a poll result. It's a
companion to the [README](README.md): the README tells you *how to call* the
async endpoint; this doc explains *how it works underneath* and why it's built
the way it is.

The design rationale (and the bugs that were fixed along the way) lives in
[`planning_docs/async_arch_for_cpu_bound_background_tasks.md`](planning_docs/async_arch_for_cpu_bound_background_tasks.md).
This document describes the engine **as built** in `services/tasks/` and
`routes/async_api.py`.

---

## The one-sentence version

Submitting a job writes a row to SQLite and a blob to disk and returns
immediately; a single background worker on the event loop picks the row up,
hands the actual CPU-heavy model work to a pool of separate **processes**, and
streams results back to disk chunk-by-chunk so that polling is just a cheap
database read — and so a redeploy mid-job resumes from the last finished chunk
instead of starting over.

---

## Why async at all?

The sync endpoint (`POST …/models/{slug}/latest/model`) runs the model inline
and blocks until it has an answer. That's perfect for a single sub-second
prediction. It's a poor fit for:

- **Large inputs** — thousands of rows, or a multi-megabyte CSV, that take
  minutes to score.
- **Keeping the app responsive** — a long inference call on the request path
  would tie up the server while it runs.

The async endpoint solves both with a classic **submit-and-poll** contract,
byte-for-byte compatible with Domino's async Model API:

```
POST …/api/modelApis/async/v1/{slug}   →  {"asyncPredictionId": "task_abc…"}
GET  …/api/modelApis/async/v1/{slug}/{id}  →  {"status": "running", "progress": {…}}
                                            →  {"status": "succeeded", "result": …}
```

You get an id in milliseconds and poll it until the status is terminal.

---

## The big picture

```
  Client ──POST──▶ FastAPI route (routes/async_api.py)   ← async, fast I/O only
                       │  validate → write input blob → INSERT row 'queued' → nudge()
                       ▼
                   SQLite  (inference_tasks)        Dataset filesystem
                   queue + lifecycle + lease        tasks/<id>/{input,output}.jsonl
                       ▲                                   ▲
                       │  claim / checkpoint / poll        │  fsync'd appends
                       │                                   │
              services/tasks/worker.py   ← ONE asyncio task in the app's lifespan
                       │  claims due rows, awaits the pool, writes results + cursor
                       ▼
              ProcessPoolExecutor  (spawn, N = cores − 1)
                ├─ proc 1: model loaded ONCE ─▶ classify_chunk(records)
                ├─ proc 2: model loaded ONCE ─▶ classify_chunk(records)
                └─ proc N: …                    (pure compute — no DB, no disk)
```

The load-bearing rule is the **division of labor**:

- **Event-loop side** (the main app process): HTTP, all SQLite reads/writes, all
  dataset file reads/writes, submitting work to the pool, awaiting it,
  checkpointing. Everything here is fast I/O.
- **Pool side** (the worker processes): pure CPU. Each process loads the model
  once and runs inference. **No database handle, no file writes.** Workers
  receive plain Python data and return plain Python data.

That split is what keeps the sync endpoint responsive while async jobs grind
away — the heavy work is on other cores, in other processes.

---

## Two ways to send input

The submit route (`routes/async_api.py`, `submit()`) accepts the same two
shapes Domino's async API does.

**By value** — the records travel inline in the request body:

```json
{"parameters": {"month": "7", "state": "Alabama", "...": "..."}}
```

A single record or a small list. It's validated up front (exactly like the sync
route) so bad input fails fast with a `422`, then persisted. The result comes
back **inline** in the poll response's `result` field.

**By reference** — the body points at a file already on the dataset:

```json
{"parameters": {"input_file": "predictions/january_batch.jsonl"}}
```

Good for large inputs. The file (`.jsonl` or `.csv`) is resolved at submit time
(a bad path fails immediately with `422`), then streamed through the model. The
poll result carries an `output_file` pointer instead of inline data, and you
stream the rows back from `…/{id}/result`.

Both shapes flow through the **same** queue, worker, and resume machinery — the
only differences are where the input comes from and how the result is returned.

---

## Step by step: the life of a request

### 1. Submit (`POST`, returns in milliseconds)

In `routes/async_api.py`:

1. `_require_ready(slug)` checks a model is configured, loaded, and that async
   tasks are enabled. Otherwise `503`/`404`.
2. The caller's identity is resolved from Domino's app-auth headers
   (for attribution and the per-user quota).
3. For **by-value**, the records are normalized and validated against the
   model's schema right now (in passthrough mode — no known schema — they're
   forwarded as-is). For **by-reference**, the file path is resolved.
4. `service.create_task(...)` enforces the per-user concurrency cap, writes the
   input blob to `tasks/<id>/input.jsonl`, and `INSERT`s a row with
   `status='queued'`.
5. `task_worker.nudge()` wakes the worker so it doesn't have to wait for its
   next poll tick.
6. The route returns `{"asyncPredictionId": "task_…"}`.

No model code runs on the request path. Submitting the 101st job while 100 are
churning is just as fast as the first.

### 2. The worker picks it up

A single asyncio task — started in the app's `lifespan` (`app.py`) and living in
`services/tasks/worker.py` — loops forever:

```
every poll interval (or on a nudge):
    budget = max_concurrent − in_flight
    for each "due" task id, up to budget:
        spawn _advance(task_id)
```

`_load_due_ids()` returns two kinds of work: tasks that are still `queued`, and
tasks stuck in `running` whose **lease has gone stale** — i.e. orphans left
behind by a crashed previous run (more on that below).

### 3. Claiming a task (the atomic lease)

Before doing any work, `_advance` calls `_try_claim`, which is a single
conditional SQL `UPDATE`:

```sql
UPDATE inference_tasks
   SET status='running', owner_token=?, claimed_at=?, heartbeat_at=?,
       attempts = attempts + 1
 WHERE id = ?
   AND (status='queued'
        OR (status='running' AND (heartbeat_at IS NULL OR heartbeat_at < ?)))
```

Only one caller can win this race — `rowcount == 1` means "I own it," `0` means
someone else got there first (or it's already terminal), so we skip it. The
`owner_token` is a fresh UUID minted per app-process run; it's how we tell *our*
live claim apart from a stale one a dead process left behind. This atomic claim
is the correctness backbone: even two ticks in the same process, or (in a future
multi-instance world) two app instances, can never double-run a task.

### 4. Running the work in chunks

Once claimed, `_advance` streams the input file and groups records into windows
of `chunk_size` (default 32). Each window is handed to `_flush`, which is the
**only** line that touches the process pool:

```python
results = await loop.run_in_executor(_pool, classify_chunk, items)
```

That single `await` is the whole trick: the CPU work runs in another process on
another core, and the event loop is parked exactly as if it were waiting on I/O.
Live sync requests keep being served the entire time.

`classify_chunk` (in `services/tasks/runner.py`) runs **inside** a pool worker.
The worker loaded the model once at startup via `init_worker` into a module
global, so each chunk is pure compute — validate, predict, return rows. No
re-loading the model, no database, no disk.

### 5. Checkpointing (fsync, *then* advance the cursor)

After each chunk, `_flush`:

1. Appends each result row to `tasks/<id>/output.jsonl` and **fsyncs it** —
   results hit durable storage first.
2. *Then* updates the row's `cursor_bytes` to the byte offset just past the last
   processed input line, bumps `completed_items`, and refreshes the heartbeat.

The ordering matters: results are durable **before** the cursor advances past
them. If the process dies between chunks, the worst case on resume is that the
last chunk is re-run — never that results are silently lost.

### 6. Finishing

When the input is exhausted, `_finish` marks the task terminal:

- **By-value:** the output rows are assembled into `result_json` and stored on
  the row, so the poll response can return them inline (a single record if there
  was exactly one input, otherwise a list).
- **By-reference:** nothing inline — the poll result points at the
  `output_file`, which you stream from `…/{id}/result`.

### 7. Poll

`GET …/{slug}/{id}` is just `service.to_public(get_task(id))` — a single indexed
SQLite read shaped into the Domino contract:

- `queued` / `running` → status, plus a `progress` block
  (`completed_items` / `total_items`) when the total is known.
- `succeeded` → inline `result`, or an `output_file` pointer.
- `failed` → an `errors` list.

Polling never touches the model or the worker — it's cheap enough to hammer.

---

## Why processes, not threads?

This is the single most important design decision, and it comes down to
Python's **GIL**.

PyTorch and scikit-learn only *partially* release the GIL — PyTorch during its
native kernels, sklearn only for BLAS-backed estimators. Tree ensembles and all
the Python-level feature-prep glue hold the GIL the whole time. So a thread pool
would serialize the very work we're trying to parallelize.

The default backend is therefore a **`ProcessPoolExecutor`** built with the
`spawn` start method (`worker.py`, `configure()`):

- **`spawn`, not `fork`** — a FastAPI process has a running event loop, threads,
  and open SQLite handles. `fork()` copies all of that into children and
  routinely deadlocks with native thread pools. `spawn` starts clean processes.
- **Model loaded once per worker** via the `initializer` (`init_worker`), into a
  module global — not pickled across the process boundary on every call.
- **Sized to `cores − 1`** so at least one core is always free for the event
  loop and live traffic.
- **Intra-op threads pinned to 1** (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `torch.set_num_threads(1)`) so `K` processes × `T` threads don't oversubscribe
  the cores and thrash the cache. Parallelism comes from the *number of
  processes*, not threads-within-processes.

There's also an opt-in **thread backend** (`MODEL_APP_TASKS_BACKEND=thread`) that
runs chunks against the single in-process model. It's lighter to spin up (no pool
of model copies) and is what the test suite uses — but it's only a win for
provably BLAS/ATen-bound models, so it's not the default.

> **Memory note:** each pool process holds its own copy of the model. `K` workers
> means `K ×` the model's RAM. For a big model, lower `MODEL_APP_TASKS_CPU_WORKERS`
> or use the thread backend so all chunks share one copy.

---

## Surviving a restart (the part that makes it robust)

Domino apps get redeployed and hard-killed routinely. The engine is built so a
job in flight when the process dies **resumes** rather than vanishing. Three
mechanisms cooperate:

**The durable cursor.** Because results are fsynced before `cursor_bytes`
advances (step 5), the row in SQLite always points at the last input byte whose
output is safely on disk. The input file is immutable after enqueue, so the
byte-offset reader (`storage.iter_jsonl(start_byte=cursor)`) can seek straight to
where work left off.

**The lease + heartbeat.** A live task stamps `heartbeat_at` on an independent
timer (`_heartbeat_loop`, every ~20s) — *not* only once per chunk. This is
deliberate: a chunk can legitimately run for minutes, and if liveness were tied
to chunk completion, a healthy long chunk would look like a corpse. The
heartbeat decouples "is it alive?" from "how long is one chunk?".

**Boot recovery.** On restart, the app process gets a brand-new `owner_token`, so
a dead run's token can never match. `_load_due_ids()` surfaces any `running` task
whose `heartbeat_at` is older than the stale threshold
(`heartbeat_seconds × lease_miss_factor`, default 20 × 4 = 80s) as an orphan, and
`_try_claim` atomically re-claims it under the new token. The fresh pool reloads
the model from scratch, and the task picks up from its last checkpoint.

An `attempts` counter (incremented on every claim, checked right after) fails a
task that keeps crashing — the poison-input guard — instead of letting it loop
forever.

---

## Concurrency: three ceilings

The engine bounds resource use with three independent caps (`worker.py`):

| Cap | Default | What it bounds |
| --- | --- | --- |
| `MODEL_APP_TASKS_CPU_WORKERS` | `cores − 1` | Pool size — the hard CPU ceiling. |
| `max_concurrent` | `2 × cpu_workers` | How many tasks the worker advances at once. |
| `USER_MAX_CONCURRENT` | `5` | Per-user in-flight tasks (enforced at submit). |

The subtle one is **`max_concurrent` defaulting to *twice* the pool size**. Each
task advances its chunks **serially** — submit a chunk, await it, write results,
checkpoint, *then* submit the next. So one task keeps at most one pool worker
busy. If we admitted only `cpu_workers` tasks, the pool would starve every time a
task paused between chunks to do its checkpoint. Admitting more tasks than there
are workers keeps a chunk always queued behind each worker. The pool's own
internal queue is the real backpressure, so memory stays bounded regardless.

---

## Things to know (honest sharp edges)

- **At-least-once delivery.** A crash re-runs the last un-checkpointed chunk, so
  `output.jsonl` can contain duplicate rows for the same input. By-reference
  consumers should treat their record id as an idempotency key and keep the last
  occurrence. (By-value results are assembled once at finish, so this is really a
  by-reference concern.)
- **Cancel latency scales with `chunk_size`.** Cancellation is checked at chunk
  *boundaries*, so a cancel request waits for the in-flight chunk to finish. With
  a large `chunk_size` and slow per-item cost, that tail can be long. Lower
  `chunk_size` for snappier cancels (at the cost of more checkpoint overhead).
  There's no safe way to interrupt a running chunk short of killing the process.
- **Poison input + `BrokenProcessPool`.** If a worker process is killed
  mid-call (OOM, native segfault), the pool fails *all* its in-flight futures,
  not just the bad one. Affected tasks resume from their checkpoints on retry, so
  no data is lost, but their latency takes a hit. The `attempts` cap eventually
  quarantines a genuinely poisonous input.
- **Residual loop I/O.** SQLite commits and `os.fsync` on the dataset are
  synchronous on the event loop. They're tiny next to multi-second CPU chunks and
  bounded in frequency by the concurrency caps, but they are micro-stalls on the
  shared loop.
- **By-reference scope.** `input_file` resolves to dataset paths only (absolute,
  or relative to the project's dataset dir). Remote stores like S3 are out of
  scope for now. CSV inputs are materialized to a `.jsonl` file once so the
  byte-cursor resume contract holds.

---

## Configuration knobs

All read from the environment (`worker.py`, `service.py`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_APP_TASKS_ENABLED` | `1` | Master toggle for the async engine. |
| `MODEL_APP_TASKS_BACKEND` | `process` | `process` (pool) or `thread` (in-process). |
| `MODEL_APP_TASKS_CPU_WORKERS` | `0` (auto = cores−1) | Process-pool size. |
| `MODEL_APP_TASKS_MAX_CONCURRENT` | `0` (auto = 2×workers) | Tasks advanced at once. |
| `MODEL_APP_TASKS_POLL_SECONDS` | `2` | Worker scan cadence. |
| `MODEL_APP_TASKS_HEARTBEAT_SECONDS` | `20` | Lease-refresh cadence. |
| `MODEL_APP_TASKS_LEASE_MISS_FACTOR` | `4` | Missed beats before a task is declared orphaned. |
| `MODEL_APP_TASKS_MAX_ATTEMPTS` | `3` | Re-run ceiling before failing (poison guard). |
| `MODEL_APP_TASKS_TORCH_THREADS` | `1` | Intra-op threads per worker (anti-oversubscription). |
| `MODEL_APP_TASKS_USER_MAX_CONCURRENT` | `5` | Per-user in-flight cap. |
| `MODEL_APP_TASKS_RETENTION_DAYS` | `7` | Task expiry horizon. |

Live health is exposed via `worker_state()` — backend, pool size, in-flight ids,
queue depth by status, last tick age, and last error.

---

## Where to look in the code

| Concern | File |
| --- | --- |
| HTTP submit / poll / result / cancel | `routes/async_api.py` |
| The worker loop, lease, heartbeat, chunking | `services/tasks/worker.py` |
| Pool-worker entrypoints (`init_worker`, `classify_chunk`) | `services/tasks/runner.py` |
| Task CRUD + the public poll projection (`to_public`) | `services/tasks/service.py` |
| JSONL blobs, fsync append, byte-cursor reader | `services/tasks/storage.py` |
| `inference_tasks` table schema | `core/db.py` |
| Pool lifecycle wired into the app lifespan | `app.py` |
| Browser playground driving submit+poll | `static/app.js` (`runAsync`) |

---

## Try it from the browser

The app's **Endpoints** page has a playground: pick **async (submit + poll)**,
fill the form, and submit. Under the hood it does exactly what a programmatic
caller does — `POST` the `{"parameters": …}` body, read back the
`asyncPredictionId`, then poll once a second until the status is terminal
(`static/app.js`, `runAsync`). The app-auth cookie carries authentication, so
there's no token handling in the browser.
