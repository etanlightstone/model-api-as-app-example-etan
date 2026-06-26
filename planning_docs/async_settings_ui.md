# Plan: Async settings as a UI section (owner-editable, DB-backed)

*Goal: let the project owner view and tune the async engine's operational knobs
from the Settings page — instead of only via environment variables that require
a redeploy — while keeping env vars working as deploy-time defaults.*

This doc plans the work. It is **not** implemented yet. It builds on the engine
described in [`HOW_ASYNC_WORKS.md`](../HOW_ASYNC_WORKS.md) and the knobs added in
[`async_hardening_suggestions.md`](async_hardening_suggestions.md).

---

## Does this make sense? (short answer: yes, for a subset)

Today every async knob is an environment variable (see the README's
Configuration table). Env vars are the right *default* mechanism — they're set
once at deploy time and live in the Compute Environment / app config. But they
have two real costs for an operator tuning a live deployment:

1. **Changing one requires a full app redeploy** (and a model re-warm). For a
   "retention is too aggressive" or "bump the chunk timeout" tweak, that's heavy.
2. **They're invisible.** Nothing in the UI tells the owner what the async engine
   is *currently* doing, so "is retention even on?" means reading source or
   hitting `/health?verbose=true`.

A Settings → Async section fixes both: it surfaces the **effective** values and
lets the owner override the *policy* knobs live. The catch is that not every knob
is safe to expose, so the plan splits them into two tiers.

### Tier A — policy knobs (expose as editable)

These change *behavior/policy*, are cheap to apply, and are already (or easily
made) read-live. **Recommended for the editable UI section:**

| Knob | Default | Apply mode |
| --- | --- | --- |
| `MODEL_APP_TASKS_ENABLED` | `1` | live + `configure()` (toggling off must also drop the pool) |
| `MODEL_APP_TASKS_RETENTION_DAYS` | `7` | live *(after refactor — see below)* |
| `MODEL_APP_TASKS_REAP_SECONDS` | `300` | live |
| `MODEL_APP_TASKS_CHUNK_TIMEOUT_SECONDS` | `0` | live |
| `MODEL_APP_TASKS_MAX_ATTEMPTS` | `3` | live |
| `MODEL_APP_TASKS_USER_MAX_CONCURRENT` | `5` | live *(after refactor)* |
| `MODEL_APP_TASKS_MAX_CONCURRENT` | `0` (auto) | live |
| `MODEL_APP_TASKS_POLL_SECONDS` | `2` | live |
| `MODEL_APP_TASKS_HEARTBEAT_SECONDS` | `20` | live |
| `MODEL_APP_TASKS_LEASE_MISS_FACTOR` | `4` | live |

All of Tier A except the two refactor cases are picked up automatically on the
next worker tick / next heartbeat iteration — the knob functions in `worker.py`
already call `os.environ.get(...)` at call time. The only plumbing they need is a
*new source* to read from (the DB override) ahead of the env var.

### Tier B — infra/sizing knobs (show read-only first; editable is Phase 2)

These are tied to the **Compute Environment and its memory**, and applying them
means tearing down and rebuilding the process pool, which **interrupts every
in-flight chunk** (they resume from their cursors, but it's disruptive):

| Knob | Default | Why it's delicate |
| --- | --- | --- |
| `MODEL_APP_TASKS_BACKEND` | `process` | switches the whole execution model |
| `MODEL_APP_TASKS_CPU_WORKERS` | auto (cores−1) | `K` workers = `K ×` model RAM; oversubscription risk |
| `MODEL_APP_TASKS_TORCH_THREADS` | `1` | intra-op threads; interacts with worker count |

**Recommendation for v1: render Tier B read-only, but still show their current
effective value and where it came from.** The owner can *see* exactly what the
engine is using without any footgun. Concretely, each Tier B row displays:

- the **effective value** the engine is actually running with right now, and
- a **source tag** — `default` if the engine fell back to the hardcoded default,
  or `env` (e.g. "from `MODEL_APP_TASKS_CPU_WORKERS`") if we detected that env var
  set on the deployment.

This reuses the exact same `source` resolution the editable Tier A knobs use (see
*Observability* and the `GET /settings/async` response below) — Tier B just
renders it without an input control. So even though these can't be changed in the
UI, the owner still learns, e.g., "we're running 7 pool workers because
`MODEL_APP_TASKS_CPU_WORKERS` is unset and it auto-sized to cores−1" vs. "5,
because the env var pins it." That visibility is most of the value; editing comes
later.

Make Tier B **editable** in a later phase with an explicit "this restarts the
worker pool and interrupts running tasks" confirmation (see Code change #4 and
Phase 2). `MODEL_APP_DATA_DIR` and other *location* settings are **never**
UI-editable — the DB we'd store overrides in lives there.

> **Note on `CPU_WORKERS` "detected" value:** its default is `0` (auto), which the
> engine resolves to `cores − 1` at runtime. The read-only row should show the
> *resolved* number actually in use (e.g. `7`), not the literal `0`, with the
> source tag clarifying it was auto-derived vs. pinned by the env var. `worker.py`
> already computes this in `cpu_workers()`; surface that.

---

## Design: a three-level precedence chain

The whole feature rests on one rule, applied per knob:

```
UI override (DB row present)  >  environment variable  >  hardcoded default
```

- A knob with a DB override row → use it.
- No DB row → fall back to exactly today's behavior (`os.environ.get` → default).

This is **fully backwards compatible**: a deployment that never opens the new UI
section behaves identically to today. "Reset to default" in the UI just deletes
the override row, restoring the env/default value.

---

## Data model

Mirror the `app_config` precedent, but use a **key/value override table** so we
store *only* the knobs the owner has actually changed (absence = fall back).

```sql
-- core/db.py SCHEMA (add)
CREATE TABLE IF NOT EXISTS async_settings (
    key        TEXT PRIMARY KEY,   -- e.g. 'MODEL_APP_TASKS_RETENTION_DAYS'
    value      TEXT NOT NULL,      -- stored as text, parsed by the typed getter
    updated_at TEXT NOT NULL DEFAULT '',
    updated_by TEXT NOT NULL DEFAULT ''
);
```

New persistence module `core/async_settings.py` (sibling of `config.py`):

```python
def get_overrides() -> dict[str, str]: ...      # {key: value} for present rows
def set_override(key, value, updated_by): ...    # upsert one row
def clear_override(key) -> None: ...             # delete (reset to env/default)
def clear_all(updated_by) -> None: ...           # reset the whole section
```

Storing as text (not typed columns) keeps the table generic; the typed
*getters* in `worker.py`/`service.py` already coerce (`int(...)`, `float(...)`)
and clamp, so the parse/validate logic stays in one place.

### In-process cache (important for performance)

`_heartbeat_seconds()`, `_chunk_timeout_seconds()`, etc. are called **many times
per tick and per task**. Hitting SQLite on every call is wasteful. Add a tiny
cached snapshot in `core/async_settings.py`:

```python
_cache: dict[str, str] | None = None
def get_overrides() -> dict[str, str]:
    global _cache
    if _cache is None:
        _cache = _load_from_db()
    return _cache
def _invalidate(): global _cache; _cache = None   # called by set/clear
```

The worker is single-process, so a module-level cache invalidated on write is
sufficient — no cross-process coordination needed (consistent with the "one
container, one writer" stance in `core/db.py`).

---

## Code changes

### 1. A single typed resolver helper

Introduce one helper both modules use so precedence + caching live in one place:

```python
# core/async_settings.py
def env_or_override(key: str, default: str) -> str:
    """UI override, else env var, else default — the precedence chain."""
    ov = get_overrides()
    if key in ov:
        return ov[key]
    return os.environ.get(key, default)
```

### 2. `worker.py` knob functions read through the resolver

Each existing getter changes from `os.environ.get(K, d)` to
`async_settings.env_or_override(K, d)`. The clamping (`max(...)`) stays. Example:

```python
def _poll_seconds() -> int:
    return max(1, int(async_settings.env_or_override("MODEL_APP_TASKS_POLL_SECONDS", "2")))
```

This is a mechanical change across ~10 functions; their *signatures and clamps
are unchanged*, so the existing tests keep passing.

### 3. `service.py`: convert the two import-time constants to functions

`RETENTION_DAYS` and `USER_MAX_CONCURRENT` are module-level constants read once
at import — they can't be tuned live as-is. Replace with getters:

```python
def retention_days() -> int:
    return int(async_settings.env_or_override("MODEL_APP_TASKS_RETENTION_DAYS", "7"))
def user_max_concurrent() -> int:
    return int(async_settings.env_or_override("MODEL_APP_TASKS_USER_MAX_CONCURRENT", "5"))
```

Update the three call sites (`create_task` quota check, `create_task` expiry
computation, and `worker._reap`'s `service.RETENTION_DAYS`). **Watch:** changing
`RETENTION_DAYS` only affects `expires_at` for *future* submits (it's stamped at
create time); already-queued rows keep their old expiry. The reaper's prune
cutoff, by contrast, is computed live, so lowering retention *does* retroactively
prune old terminal rows on the next sweep. Note this asymmetry in the UI help
text.

### 4. Pool-rebuild knobs (Tier B, when made editable)

`configure()` only rebuilds when `_config_signature(cfg)` changes — and that
signature is **model-derived** (`source_type, params, slug`), so changing
`CPU_WORKERS`/`TORCH_THREADS` would *not* trigger a rebuild today. To support
Tier B edits, either:

- fold the pool-sizing knobs into `_config_signature` (so a change naturally
  invalidates), **or**
- add `configure(force=True)` and call it from the settings route after a Tier B
  change.

The route must also warn that a rebuild cancels in-flight pool futures (same
blast radius as the `BrokenProcessPool` path documented in `HOW_ASYNC_WORKS.md`).
Defer this; v1 keeps Tier B read-only.

### 5. Routes (`routes/settings.py`)

Add owner-gated endpoints, re-checking owner server-side like the existing ones:

```python
@router.get("/async")          # current effective values + source per knob + bounds
@router.post("/async")         # validate + upsert overrides for Tier A keys
@router.post("/async/reset")   # clear one key or all → back to env/default
```

`GET /settings/async` returns, per knob: `key`, `value` (effective),
`source` (`"ui" | "env" | "default"`), `editable` (Tier A/B), and `min`/`max`.
`POST` validates against the bounds table below, writes via
`async_settings.set_override`, then **applies**: for Tier A, nothing more is
needed (next tick); if `ENABLED` changed, also call `task_worker.configure()` so
the pool is dropped/rebuilt to match. Reuse the `_require_owner` guard.

### 6. UI (`templates/settings.html` + `static/settings.js`)

Add a third `<section class="card">` ("Async engine") below Diagnostics, seeded
from a new `async_init_json` blob (built in `routes/ui.py:settings_page`, exactly
like `settings_init_json`). Render a compact form:

- Tier A knobs → labeled number inputs / a toggle for `ENABLED`, each with a
  "default: N" hint and a per-row "reset" link (calls `/settings/async/reset`).
- Tier B knobs → **read-only rows** (no input control) showing the effective
  value plus a muted source tag: `default` or `from <ENV_VAR>` when we detected
  the env var set. For `CPU_WORKERS`, show the resolved worker count, not the
  literal `0`. This is the explicit v1 approach: visible but not editable.
- A "Save async settings" button → `POST /settings/async`, then toast + re-render
  (reuse `postJSON`, `toast`, `showInlineError` already in `settings.js`).
- Inline validation mirroring the server bounds (cheap UX; server is the source
  of truth).

No new JS framework — this matches the existing vanilla `settings.js` style.

---

## Validation / bounds table (server-enforced)

The route rejects out-of-range values with `422` (mirrors the existing
`select_model` validation style). Bounds match the current `max(...)` clamps:

| Knob | Type | Min | Max (suggested) | Notes |
| --- | --- | --- | --- | --- |
| `MODEL_APP_TASKS_ENABLED` | bool | — | — | toggle |
| `MODEL_APP_TASKS_RETENTION_DAYS` | int | 1 | 365 | future submits only for expiry; prune is retroactive |
| `MODEL_APP_TASKS_REAP_SECONDS` | int | 0 | 86400 | `0` disables the reaper |
| `MODEL_APP_TASKS_CHUNK_TIMEOUT_SECONDS` | float | 0 | 86400 | `0` disables; set generously |
| `MODEL_APP_TASKS_MAX_ATTEMPTS` | int | 1 | 10 | poison-input guard |
| `MODEL_APP_TASKS_USER_MAX_CONCURRENT` | int | 1 | 1000 | per-user submit cap |
| `MODEL_APP_TASKS_MAX_CONCURRENT` | int | 0 | 1000 | `0` = auto (2×workers) |
| `MODEL_APP_TASKS_POLL_SECONDS` | int | 1 | 300 | worker scan cadence |
| `MODEL_APP_TASKS_HEARTBEAT_SECONDS` | int | 5 | 600 | also drives lease-stale window |
| `MODEL_APP_TASKS_LEASE_MISS_FACTOR` | int | 2 | 20 | stale = heartbeat × factor |

Cross-knob sanity to surface (warn, don't block): `HEARTBEAT_SECONDS ×
LEASE_MISS_FACTOR` should comfortably exceed the longest expected chunk, or a
healthy long chunk gets reclaimed as an orphan.

---

## Observability

Extend `worker_state()` (already surfaced via `/health?verbose=true`) to report,
per knob, the **effective value and its source** (`ui`/`env`/`default`). This
makes the UI section and the health endpoint agree, and answers "why is the
engine behaving this way?" without log-diving. Low effort — `worker_state`
already returns most of these values; add the source tag.

---

## Testing

- **Unit:** `env_or_override` precedence (override > env > default); cache
  invalidation on set/clear; bounds validation rejects/accepts at the edges.
- **Behavioral:** set `REAP_SECONDS`/`RETENTION_DAYS` via the store and assert the
  reaper picks them up on the next sweep (extends the reaper test already written
  for the hardening work); set `CHUNK_TIMEOUT_SECONDS` and assert `_flush` times
  out (extends the existing timeout test).
- **Route:** owner-gating (403 for non-owner), `POST` then `GET` round-trips the
  effective value with `source="ui"`, reset returns it to `source="env"/"default"`.
- The existing `tests/test_app.py` thread-backend suite must stay green — the
  knob-function refactor is signature-preserving, so it should.

---

## Phasing

1. **Phase 1 (this plan's core):** store + resolver + `worker.py`/`service.py`
   read-through + Tier A editable UI + **Tier B read-only, displaying the
   effective value and its source (`default` vs detected env var)** +
   observability source tags. Self-contained, backwards compatible, no
   pool-lifecycle risk.
2. **Phase 2:** make Tier B editable with a "rebuild pool / interrupt in-flight"
   confirmation and the `configure(force=True)` (or signature) change.
3. **Phase 3 (optional):** audit trail (who changed what, when) — the table
   already has `updated_by`/`updated_at`; just surface a small history.

---

## Explicitly NOT doing (avoid scope creep)

- **No per-knob hot-reload signaling.** Tier A is read live already; a DB read on
  the next tick is the entire "apply" mechanism. No pub/sub, no SIGHUP.
- **No location/infra settings in the UI** (`MODEL_APP_DATA_DIR`, proxy/registry
  hosts, identity header). Those are deploy-time and partly chicken-and-egg with
  where the override DB itself lives.
- **No multi-instance coordination.** Same stance as the rest of the engine: one
  container, one writer, a module-level cache is enough.
- **No env-var deprecation.** Env vars remain the supported default layer; the UI
  is an override, not a replacement.

---

## Open questions for the owner

1. Is read-only Tier B enough for v1, or is live `CPU_WORKERS` tuning a
   day-one need (it's the one most likely to want changing on a memory-tight box)?
2. Should toggling `MODEL_APP_TASKS_ENABLED` off via the UI **drain** in-flight
   tasks (let them finish) or just stop dispatching new ones? (Plan assumes the
   latter — matches how `is_enabled()` gates dispatch today.)
3. Do we want the audit trail (Phase 3) for compliance, or is "current value +
   who set it" enough?
