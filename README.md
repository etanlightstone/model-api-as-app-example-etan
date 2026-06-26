# Model API as App

A reusable **Domino App** that hosts any project model as a REST API whose URL
and payload shape mirror **Domino Model APIs** — but runs as an *app*, so it
gets Domino's native app authentication plus three things a Model API doesn't
have: a self-documenting browser UI, a built-in playground, and an always-on
async submit/poll surface.

The app is generic: point it at a model and it figures out the request/response
schema automatically, then serves a typed, validated endpoint.

```
┌──────────────────────────────────────────────────────────────┐
│  Domino App (one container)                                    │
│                                                                │
│   Browser  ─▶  self-documenting UI + playground   (templates/) │
│   Machine  ─▶  POST /models/{slug}/latest/model    (sync)      │
│   Machine  ─▶  POST /api/modelApis/async/v1/{slug}  (async)    │
│                                                                │
│   ModelAdapter ── inputs / outputs / predict()                 │
│   schema       ── signature → pydantic + JSON-Schema + curl    │
│   async worker ── SQLite queue + process pool (Plan A)         │
└──────────────────────────────────────────────────────────────┘
        SQLite (config + task queue) + JSONL blobs on a Domino dataset
```

---

## What you get

- **One endpoint per model, API-compatible with Domino Model APIs.** Same
  `{"data": …}` request envelope, same async `asyncPredictionId` + status
  contract — only the URL *prefix* differs (app routing instead of
  `/models/{id}`). Existing client code keeps the same body and method; only the
  host and the source of the bearer token change.
- **Auto-schema.** The app infers a model's input/output shape from a registry
  model's **MLflow signature** or a custom function's **typed signature**, and
  exposes a typed, validated REST endpoint from it.
- **Self-documenting + a playground.** Visiting the app in a browser explains
  every endpoint (field tables, request/response shapes, copy-paste curl) and
  lets you exercise it live — no separate docs to maintain.
- **Always-on async.** Every hosted model automatically gets the real-time
  endpoint *and* async submit/poll, with no extra deploy-time decision.
- **Owner-gated settings.** The project owner picks which model to host from the
  UI; everyone else sees a friendly "not set up yet" state until then.

---

## Quick start

### 1. Deploy as a Domino App

Publish this project as a Domino App with **`app.sh`** as the entry point. The
app listens on port 8888 behind Domino's reverse proxy and inherits Domino's
app authentication — everyone who can reach it has already authenticated.

```bash
# app.sh runs:
uvicorn app:app --host 0.0.0.0 --port 8888 --proxy-headers --forwarded-allow-ips '*'
```

On first load the app shows **"This model endpoint is not set up yet."**

### 2. Configure the model (owner only)

Open **Settings** and choose one of:

- **Registry model** — pick a registered MLflow model + version. Its signature
  defines the schema.
- **Custom function** — point at a `file.py` + function, e.g.
  `example/weather_regressor/model_api.py` → `predict`. The schema is inferred
  from the typed signature; an optional `model_app.yaml` next to the file
  overrides types/examples/output names and flags image fields.

Saving warms the model and the endpoints light up immediately.

### 3. Call it

Synchronous (real-time) — identical body to a Domino Model API:

```bash
# Inside any Domino workload, fetch a short-lived access token from the proxy:
TOKEN=$(curl -s http://localhost:8899/access-token)

curl -X POST "https://apps.<deployment>/<...>/models/weather-regressor/latest/model" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"data": {"month": "7", "week_of": "28", "state": "Alabama",
               "precipitation": "0.1", "wind_speed": "5.0", "wind_direction": "20"}}'
# → {"result": {"avg_temp": 83.2, "max_temp": 93.3, "min_temp": 72.6}, "request_id": "...", "timing_ms": 33.8}
```

Asynchronous (submit → poll), same contract as Domino's async Model API:

```bash
TOKEN=$(curl -s http://localhost:8899/access-token)
BASE="https://apps.<deployment>/<...>/api/modelApis/async/v1/weather-regressor"

ID=$(curl -s -X POST "$BASE" \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $TOKEN" \
  -d '{"parameters": {"month": "1", "state": "Texas", "...": "..."}}' | jq -r .asyncPredictionId)

curl -s "$BASE/$ID" -H "Authorization: Bearer $TOKEN"   # repeat until status is terminal
```

The app's **Endpoints** page generates this exact copy-paste snippet for the
hosted model, with the real URL, slug, and an example payload filled in. The
**Playground** runs the same calls from your browser (the Domino app-auth cookie
carries auth — no token handling in the browser).

---

## Authentication

Endpoints rely on **Domino app auth**: everyone who can reach the app has
authenticated. Programmatic callers just need a valid **Domino access token** in
the `Authorization: Bearer` header.

- **Inside a Domino workload** (workspace, job, app), fetch one ephemerally from
  the local token proxy: `curl -s http://localhost:8899/access-token`. The proxy
  base is templated from `DOMINO_API_PROXY` — no static key, no model-token
  issuance, no stale-token footgun.
- **From a laptop / CI**, paste a Personal Access Token in place of `$TOKEN`. The
  UI's "Calling from" toggle shows both forms.

---

## Sync vs. async, and batch scope

| | Sync | Async |
| --- | --- | --- |
| URL | `…/models/{slug}/latest/model` | `…/api/modelApis/async/v1/{slug}` |
| Use | real-time, sub-second | longer / batched work; submit-and-poll |
| Body | `{"data": {…}}` | `{"parameters": {…}}` |

Async supports the same surface Domino's async Model API does:

- **By value** — a single record (or small multi-record payload) returned inline
  in the poll response's `result`.
- **By reference** — `{"parameters": {"input_file": "<dataset path>.jsonl|.csv"}}`
  for larger inputs; the worker streams the file through the model and the poll
  result carries an `output_file` pointer (stream it from `…/{id}/result`).

Orchestrating a very large CSV as a standalone Domino Job remains the user's
path (unchanged from today).

### How async runs (Plan A)

The async engine follows
[`planning_docs/async_arch_for_cpu_bound_background_tasks.md`](planning_docs/async_arch_for_cpu_bound_background_tasks.md):
a SQLite-as-queue, JSONL blobs on the dataset, and a single asyncio worker that
dispatches CPU work to a `spawn` `ProcessPoolExecutor` (model loaded once per
worker). It uses an **atomic lease claim**, a **timer-based heartbeat**, and
fsync-ordered chunk checkpoints, so a redeploy resumes in-flight work from the
last durable chunk. Set `MODEL_APP_TASKS_BACKEND=thread` to run chunks in a
thread against the in-process model instead (lighter; used by the test suite).

---

## Supported model types

| Type | How | Example |
| --- | --- | --- |
| scikit-learn | custom function or registry pyfunc | `example/weather_regressor` |
| PyTorch | custom function or registry pyfunc | `example/diabetes_classer` |
| Image classifier | custom function, base64 `image` field | `example/image_classifier` |
| Any registered MLflow model | registry signature → schema | — |

Images follow the Domino convention: a base64 string in a JSON field (no
multipart). Mark the field with `image_fields:` in `model_app.yaml` and the
playground renders a file picker that base64-encodes client-side.

---

## Compatibility with Domino Model APIs

| Aspect | Real Model API | This app |
| --- | --- | --- |
| Sync URL | `/models/{id}/latest/model` | `{APP_BASE}/models/{slug}/latest/model` |
| Async submit/poll | `/api/modelApis/async/v1/{id}[/{predId}]` | same, `{slug}` prefix |
| Request/response envelope | `{"data": …}` / `{"result": …}` | **identical** |
| Async contract | `asyncPredictionId` + status enum | **identical** |
| Auth | model access token | **Domino app auth** (Bearer = Domino access token) |

**Known deltas:** the response echoes `request_id` + `timing_ms` but does not
reproduce Domino's full `release` metadata block.

---

## Configuration (environment)

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_APP_DATA_DIR` | `<dataset>/<project>/.model_app` | SQLite DB + async blobs |
| `MODEL_APP_TASKS_BACKEND` | `process` | `process` (Plan A pool) or `thread` |
| `MODEL_APP_TASKS_CPU_WORKERS` | `0` (auto = cores−1) | process-pool size |
| `MODEL_APP_TASKS_ENABLED` | `1` | master toggle for the async engine |
| `MODEL_APP_USER_HEADER` | (auto-probed) | identity header for owner-gating |
| `MODEL_APP_DEV_OWNER` | off in Domino | treat caller as owner (local dev only) |
| `DOMINO_API_PROXY` | `http://localhost:8899` | token-proxy base (set by Domino) |

If owner-gating misbehaves on a deployment, open **Settings → Diagnostics →
Show identity headers** (or `GET /settings/whoami`) to see which header carried
your identity, then set `MODEL_APP_USER_HEADER` accordingly.

---

## Development

```bash
pip install -r requirements.txt

# Run locally (treats you as owner; thread backend for a fast loop):
MODEL_APP_DEV_OWNER=1 MODEL_APP_TASKS_BACKEND=thread \
  uvicorn app:app --reload --port 8888

# Tests (no pytest needed):
python -m unittest tests.test_app
```

Layout: `app.py` (FastAPI + lifespan), `routes/` (ui, predict, async_api,
settings, health), `core/` (adapter, schema, registry, identity, config, state),
`services/tasks/` (async worker, Plan A), `templates/` + `static/` (UI).

---

## Appendix — the example models

Two small, self-contained examples that train a model with Domino's MLflow
tracking and serve it as a Model API two ways — as **custom scoring code**
(`model_api.py`) and as a **registry model that Domino auto-wraps**
(`pyfunc_model.py`, logged with a signature). They double as the validation
fixtures for this app.

| | [`diabetes_classer`](example/diabetes_classer/) | [`weather_regressor`](example/weather_regressor/) | [`image_classifier`](example/image_classifier/) |
| --- | --- | --- | --- |
| Framework | PyTorch | scikit-learn | PIL/numpy (stand-in) |
| Task | Binary classification | Multi-output regression | Image classification |
| Output | `is_diabetic`, `probability` | `avg_temp`, `max_temp`, `min_temp` | `label`, `probabilities` |

Each sample's `README.md` has the full data schema and commands. Typical flow:

```bash
cd example/<sample>
python train.py     # train + track + log the registry model
python predict.py   # sanity-check predictions locally
python model_api.py # smoke-test the custom-code scoring function
```

Then either publish via **Publish → Model APIs** / the **Model Registry** (the
classic Domino paths), or host it through this app (point Settings at the
`model_api.py` → `predict` function, or pick the registered model).
