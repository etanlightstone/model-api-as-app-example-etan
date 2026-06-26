# Async Hardening Suggestions

*Findings from a code read of the async submit/poll engine (`services/tasks/*`, `routes/async_api.py`, `core/db.py`). The goal here is **not** to redesign — the lease/heartbeat/checkpoint core is sound. This is a prioritized list of gaps with concrete, minimal fixes, plus an explicit "leave it alone" section so we don't gold-plate.*

## How it works today (one paragraph)

Submit writes a `queued` row to the `inference_tasks` SQLite table and returns an `asyncPredictionId`. One asyncio worker (`services/tasks/worker.py`) polls every ~2s, **atomically claims** due tasks (`[FIX 1]`), runs them chunk-by-chunk on a `spawn` `ProcessPoolExecutor`, **heartbeats on an independent 20s timer** (`[FIX 2]`) so a long chunk isn't mistaken for a crash, and **byte-cursor checkpoints** each flushed chunk so a restart resumes from the last durable point. Poll reads the row and renders the Domino contract via `to_public`. Terminal statuses: `succeeded | failed | cancelled | expired`.

---

## Priority 1 — gaps likely to bite in normal operation

> **Status (2026-06-26): both Priority 1 items are implemented** in `services/tasks/worker.py`.
> The new config knobs (`MODEL_APP_TASKS_RETENTION_DAYS`, `MODEL_APP_TASKS_REAP_SECONDS`,
> `MODEL_APP_TASKS_CHUNK_TIMEOUT_SECONDS`) are documented in the main README's
> Configuration table. Priority 2 and 3 are **not** done. See the per-item
> **✅ DONE** notes below for exactly what shipped.

### 1.1 No retention enforcement for completed results *(common case)*

**What we found.** `RETENTION_DAYS` (default 7) is used **only** to compute `expires_at` at create time (`service.py:63`), and `expires_at` is **only** checked at claim time for not-yet-started tasks (`worker.py:304`). For a task that actually ran, `to_public` never consults `expires_at` (`service.py:132`). There is **no reaper anywhere** — no sweep marks old terminal tasks `expired`, prunes rows, or deletes output files.

**Consequence.** Succeeded rows and their on-disk output JSONL (`output_file_path`) accumulate **forever**. On a Domino-dataset filesystem this is a slow disk leak; the SQLite table grows unbounded. The "7-day retention" implied by the knob simply doesn't happen for completed work.

**Suggested fix (small).** Add a periodic reaper coroutine to the worker loop (it already wakes every poll interval — piggyback on it, run at most once per N minutes):
- Mark non-terminal tasks past `expires_at` as `expired` (closes the case where nothing ever claims them).
- For terminal tasks older than `RETENTION_DAYS`, delete the row **and** unlink `input_file_path` / `output_file_path`.
- Guard file deletes with `os.path.isfile` and swallow errors (best-effort).

This is ~30 lines and makes the existing `RETENTION_DAYS` knob mean what it says.

> **✅ DONE.** Added `_reap()` + `_maybe_reap()` to the worker loop (`worker.py`).
> Each sweep (1) flips non-terminal tasks past `expires_at` to `expired` and (2)
> deletes terminal tasks older than `RETENTION_DAYS` — both the row and the
> on-disk blobs. The sweep piggybacks on the existing poll loop, throttled to run
> at most once per `MODEL_APP_TASKS_REAP_SECONDS` (default 300; set `0` to
> disable). File deletes are best-effort (`os.path.isfile` guard, errors
> swallowed). **Important safety fix vs. the original suggestion:** by-reference
> `.jsonl` tasks keep `input_file_path` pointing at the *caller's own dataset
> file*, so the reaper only unlinks blobs under `TASKS_DIR` (`_under_tasks_dir`)
> — it never touches the user's source data. The empty per-task dir (incl.
> `meta.json`) is removed too.

### 1.2 No timeout on inference — a hung `predict` pins resources forever *(edge case, high blast radius)*

**What we found.** `await loop.run_in_executor(_pool, classify_chunk, items)` (`worker.py:389`) has **no timeout**; the thread path (`asyncio.to_thread`, `worker.py:392`) likewise. A genuinely wedged prediction (infinite loop, deadlock, external call with no timeout) keeps the heartbeat ticking (separate coroutine), so:
- it never goes stale → never reclaimed,
- `expires_at` isn't checked mid-run,
- cancel is only checked **between** chunks (`worker.py:331`), so it can't interrupt the in-flight one.

It runs effectively forever, holding a pool worker + a concurrency slot until the app restarts.

**Note on what *is* already handled.** A worker process that *crashes* (OOM, segfault) surfaces as a `BrokenProcessPool` and is caught → `failed` (`worker.py:349`). And on restart, an interrupted `running` task goes stale after `lease_stale_seconds` (80s default), is reclaimed, and resumes from its cursor. So crash and restart are covered; only the **hang** is not.

**Suggested fix (small, process backend only).** Wrap the executor await in `asyncio.wait_for(..., timeout=MODEL_APP_TASKS_CHUNK_TIMEOUT_SECONDS)`. On `TimeoutError`: for the process backend, **shut down and rebuild the pool** (the only reliable way to kill a wedged worker — futures can't be force-cancelled once running), then let the chunk fail → the existing attempts/retry path takes over. Make the timeout default generous (e.g. 0 = disabled) so we don't break legitimately slow models; opt-in per deployment. Thread backend can't be force-killed, so document that it only marks the task failed and abandons the thread.

> **✅ DONE.** `_flush()` now wraps the executor await in
> `asyncio.wait_for(..., MODEL_APP_TASKS_CHUNK_TIMEOUT_SECONDS)` when the knob is
> set (default `0` = disabled, so slow-but-legitimate models are unaffected). On
> a breach: the `process` backend calls `shutdown_pool()` + `configure()` to kill
> the wedged worker and rebuild the pool, then re-raises so `_advance`'s existing
> handler marks the task `failed` and the attempts/retry path resumes from the
> last durable cursor. The `thread` backend re-raises without a kill (threads
> can't be force-stopped) — the chunk fails and the abandoned thread runs to
> completion in the background. **Caveat:** rebuilding the pool also interrupts
> any *other* in-flight chunk sharing it; those tasks fail this attempt and
> resume from their cursor on the next claim. Acceptable for an opt-in safety
> valve against a hang.

---

## Priority 2 — correctness sharp edges (lower frequency)

### 2.1 Duplicate output rows on resume after a partial chunk

**What we found.** `_flush` writes all result lines to the output file **and then** bumps `cursor_bytes` in a separate step (`worker.py:393-399`). The append is fsync'd before the cursor moves (good ordering), but if the process dies *after* writing some/all output lines but *before* the cursor `UPDATE` commits, resume re-runs that chunk and **appends the same outputs again**. The async-arch doc already flags this as a known `[CAVEAT]` (duplicate output rows).

**Consequence.** By-reference output files can contain duplicate records after an ill-timed crash. Consumers that assume 1:1 input:output line counts will be off.

**Suggested fix (small).** Cheapest honest option: document the at-least-once guarantee for by-reference output and have consumers dedupe on a record id. If we want at-most-once, write output to a temp region keyed by chunk and only "commit" the lines in the same transaction that advances the cursor — meaningfully more complex; probably not worth it unless a consumer actually breaks.

### 2.2 Cancel latency is a full chunk *(working as designed, but undocumented to callers)*

Cancel is honored only at chunk boundaries (`worker.py:331`). With a large `chunk_size` and a slow model, "cancelled" can take a long time to take effect, and an in-flight chunk's work is still computed (and billed). This is a deliberate tradeoff, not a bug — but the async doc lists it as a `[CAVEAT]` and the **API doesn't tell the caller**. Worth a one-line note in the cancel endpoint's response/docstring (`routes/async_api.py:117`) so clients don't assume immediate stop.

### 2.3 By-value `result_json` has no size ceiling

`to_public` inlines the full `result_json` for by-value tasks (`service.py:137`), and it's assembled by reading the entire output back into memory (`worker.py:344-347`). A caller submitting a large by-value batch gets the whole result materialized in one DB row and one poll response. Probably fine for the intended "interactive, small payload" use, but a `MODEL_APP_TASKS_MAX_INLINE_ITEMS` guard at submit time (steer large jobs to by-reference) would prevent a pathological row. Low priority unless we see big by-value submits.

---

## Priority 3 — observability (cheap wins)

- **Surface `error_message` for `expired` tasks.** `_finish` writes `"task expired before completion"` but `to_public` only reads `error_message` for `failed` (`service.py:144`), so the poller sees a bare `{"status": "expired"}` with no hint why. Either add the message to the expired branch or document that expired is silent. (Came up directly while investigating.)
- **Counter for orphan reclaims and timeout kills.** `worker_state()` (`worker.py:412`) already exposes a nice health snapshot. Adding cumulative counters for reclaims, retries, and (if 1.2 lands) timeout-kills would make "is async healthy?" answerable from `/health` without log-diving.

---

## Explicitly NOT recommending (avoid overengineering)

- **No external broker / Redis / Celery.** The SQLite-as-queue + single asyncio worker is the right scale for one Domino app container. Don't add infrastructure.
- **No multi-instance coordination beyond the existing lease.** The atomic claim (`[FIX 1]`) already handles the duplicate-claim case correctly; we don't need distributed locks.
- **No mid-chunk cancellation / preemption.** Chunk-boundary cancel is good enough; interrupting compute mid-flight isn't worth the complexity.
- **No exactly-once output.** At-least-once + consumer dedupe (2.1) is the pragmatic guarantee for a restartable single-container worker.

---

## Suggested order if we do anything

1. ~~**1.1 reaper**~~ — ✅ **done** (closes the disk/row leak, makes `RETENTION_DAYS` honest).
2. **3 observability** — near-free, makes the rest debuggable. *(not started)*
3. ~~**1.2 chunk timeout**~~ — ✅ **done** (opt-in safety valve; default-disabled so slow models are unaffected).
4. Everything in P2 is "document the behavior" unless a consumer actually trips on it. *(not started)*
