# Model-API-as-App — Phased Implementation Plan

A reusable **Domino App** that hosts any project model as a REST API whose URL
and payload shape mirror Domino **Model APIs**, but runs as an *app* so it can
use Domino's native app auth (and a few advantages Model APIs don't have: a
self-documenting browser UI, a built-in playground, and an always-on async
submit/poll surface). The app is generic — the operator points it at a model
and it figures out the request/response schema automatically.

> **Status:** plan only. Nothing here is built yet. Phases are ordered so each
> one is independently demoable. "Don't over-engineer" is a first-class
> constraint — every phase calls out the minimum that ships and what is deferred.

---

## 1. Goals & non-goals

**Goals**
- One FastAPI app that serves a selected model as a REST endpoint.
- **API-shape compatible** with Domino Model APIs (same envelope, same async
  submit/poll contract) — only the URL *prefix* differs (app routing, not the
  `/models/{id}` prefix Model APIs own).
- **Headless by default** (machines POST JSON), **self-documenting in a
  browser** (endpoint reference + live playground), using a light Domino theme.
- **Auto-schema**: infer the model's input/output shape from (a) a registry
  model's MLflow signature, or (b) a custom inference function's typed
  signature — and expose a typed, validated REST endpoint from it.
- **Always-on async**: every hosted model gets the real-time endpoint *and* the
  async submit/poll endpoints, with no extra deploy-time decision by the user.
- **Settings UI** for the app owner to pick which registry model to deploy;
  a graceful "not set up yet" state for everyone else before that happens.

**Non-goals (explicit, to avoid scope creep)**
- No bespoke React/Vite/SPA build step. UI is server-rendered HTML + a small
  amount of vanilla JS (and a CDN component lib for Domino styling). 
- Not trying to *replace* Domino Model APIs or reproduce their `/models/{id}`
  URL prefix, monitoring, or token issuance.
- Not a multi-model router in v1 — **one app instance hosts one model** (the
  same 1:1 relationship a Model API has). Multi-model is a possible later lever,
  not a v1 goal.
- Large-scale batch (10k-row CSV) is **only** supported to the extent Domino's
  own async Model API supports it — see §5.4. The job-based bulk path stays the
  user's responsibility, exactly as today.

---

## 2. Decisions locked in (from clarifying Q&A)

| Topic | Decision |
| --- | --- |
| **Endpoint auth** | **Domino app auth only.** The app sits behind Domino's native app authentication; everyone who can reach the app can call its endpoints. No separate model tokens to issue or manage. (This is the core motivation for the whole project.) Programmatic callers authenticate with a **Domino access token** — fetched ephemerally from the in-workload token proxy `http://localhost:8899/access-token` and sent as `Authorization: Bearer <token>`. The self-doc UI generates this exact all-in-one snippet per endpoint (§5.7). |
| **Schema inference** | **Infer from the function signature, allow override.** Introspect typed parameters (e.g. `predict(calories_wk: float, ...)`) to auto-build the schema; let the operator annotate/override when introspection is ambiguous. Registry models use their MLflow signature directly. |
| **Async design** | Use the patterns in [`async_arch_for_cpu_bound_background_tasks.md`](async_arch_for_cpu_bound_background_tasks.md): **Plan A** (in-process `spawn` `ProcessPoolExecutor`, SQLite-as-queue, dataset-blob storage, single asyncio worker, lease/heartbeat resume) as the default; **Plan B** (Domino Jobs offload) as an optional later backend for heavy/GPU work. |
| **Batch scope** | **Match what Domino's async Model API supports** (§5.4): single-record by value, or **by-reference** (an `input_file` pointer) for larger payloads, with a 10 KB packet guideline. No new batch surface beyond that. |

---

## 3. How Domino Model APIs look today (the shape we mirror)

Captured from the two working examples in this project and the Domino docs, so
the app's contract is concrete.

### 3.1 Synchronous (real-time)

- **Real Model API URL:** `POST {DOMINO_URL}/models/{MODEL_ID}/latest/model`
- **Auth (real):** bearer/basic model access token.
- **Request envelope:**
  ```json
  { "data": { "calories_wk": "3000.0", "hrs_exercise_wk": "5.0", "...": "..." } }
  ```
- **Response:** the function's/pyfunc's returned object, wrapped by Domino with
  `release` / `timing` / `request_id` metadata:
  ```json
  {
    "result": { "is_diabetic": false, "probability": 0.0067, "threshold": 0.5 },
    "release": { "...": "..." }, "request_id": "...", "timing": 12.3
  }
  ```
- Values arrive as **strings** and are coerced internally (why the example
  pyfuncs declare string input signatures). Multi-row works because the pyfunc
  accepts a DataFrame.

### 3.2 Asynchronous (submit + poll)

- **Submit:** `POST {DOMINO_URL}/api/modelApis/async/v1/{MODEL_ID}`
  ```json
  { "parameters": { "calories_wk": "3000.0", "...": "..." } }
  ```
  → `{ "asyncPredictionId": "<id>" }`
- **Poll:** `GET {DOMINO_URL}/api/modelApis/async/v1/{MODEL_ID}/{prediction_id}`
  - `{ "status": "queued" }`
  - `{ "status": "succeeded", "result": { ... } }`
  - `{ "status": "failed", "errors": [ ... ] }`
- **Constraints:** ~10 KB request/response packet guideline; large payloads sent
  **by reference** (`{"parameters": {"input_file": "s3://.../in.csv"}}`); Domino
  keeps a picked-up request alive ~30 min; statuses are `queued` / `succeeded` /
  `failed`.

### 3.3 What our app changes vs. keeps

| Aspect | Real Model API | This app |
| --- | --- | --- |
| Sync URL | `/models/{id}/latest/model` | `{APP_BASE}/models/{slug}/latest/model` (app-routed prefix) |
| Async submit | `/api/modelApis/async/v1/{id}` | `{APP_BASE}/api/modelApis/async/v1/{slug}` |
| Async poll | `…/async/v1/{id}/{predId}` | `…/async/v1/{slug}/{predId}` |
| Request/response envelope | `{"data": …}` / `{"result": …}` | **identical** |
| Async submit/poll contract | `asyncPredictionId`, status enum | **identical** |
| Auth | model access token | **Domino app auth** (Bearer = ephemeral Domino access token) |

> `{APP_BASE}` is the app's reverse-proxy URL, e.g.
> `https://apps.<deployment>/apps/<app-name>`. Existing client code keeps the
> same body shape and method; only the host/prefix and the **source of the
> Bearer token** change — instead of a static model token, callers use a
> short-lived Domino access token (§5.7). We document the delta prominently.

---

## 4. Target architecture (lightweight)

```
                         ┌─────────────────────────────────────────────┐
  Browser (owner) ─────▶ │  FastAPI app  (single Domino app container)  │
  Browser (viewer) ────▶ │                                             │
  Machine client ──────▶ │  routes/                                    │
                         │    ui.py        self-doc UI + playground     │
                         │    predict.py   sync  /models/{slug}/latest  │
                         │    async_api.py submit+poll  /api/modelApis  │
                         │    settings.py  owner-only model selection    │
                         │  core/                                       │
                         │    adapter.py   ModelAdapter (load+predict)  │
                         │    schema.py    signature → pydantic + JSON  │
                         │    registry.py  Domino registry API client   │
                         │    identity.py  owner vs viewer (app headers)│
                         │    config.py    selected-model config store  │
                         │  services/tasks/  async worker (Plan A)       │
                         └───────────────┬─────────────────────────────┘
                                         │
              SQLite (config + async task queue)  ── on Domino dataset
              Dataset filesystem (async in/out JSONL blobs)
              ProcessPoolExecutor (spawn) — model loaded once per worker
```

**Load-bearing rules**
- The event loop only does I/O + orchestration. All model inference for async
  runs off-loop in worker processes (per the async doc). For **sync** requests,
  inference also runs via `await loop.run_in_executor(pool, …)` so a slow model
  never blocks other live requests.
- The model is described by a single `ModelAdapter` abstraction so the UI,
  sync route, and async worker all share one notion of "inputs, outputs,
  predict()". Adding a model *type* = adding an adapter, nothing else.
- One config record names the active model + its source. Empty = unconfigured
  ("not set up yet" state).

---

## 5. The core mechanics (the parts worth getting right)

### 5.1 `ModelAdapter` — the single abstraction

```python
class ModelAdapter(Protocol):
    name: str                 # display name / slug source
    input_schema: Schema      # ordered fields: name, type, required, example
    output_schema: Schema     # named outputs + types (best-effort)
    def predict(self, records: list[dict]) -> list[dict]: ...
    def warmup(self) -> None: ...      # load once (registry pyfunc / torch / joblib)
```

Three concrete adapters (added across phases):
1. **RegistryAdapter** — loads a registered MLflow model by name+version via the
   project model registry; schema comes straight from the model's **signature**.
   Covers the "registry natively hosts it / auto-deploys" case and both example
   pyfuncs.
2. **CustomFunctionAdapter** — operator points at `file.py` + function name (the
   `model_api.py` → `predict` pattern). Schema inferred from the **typed
   signature** (parameter names → input fields, type hints → field types,
   defaults → optional + example); return annotation / a sample call result →
   output fields. Operator can override via an optional sidecar (see 5.2).
3. **Pyfunc/Framework adapter** — thin wrappers for raw torch/sklearn artifacts
   when there's no signature (operator supplies feature columns); mostly folds
   into #1 once a signed pyfunc exists.

### 5.2 Schema inference + override

- **Registry path:** read `signature.inputs` / `signature.outputs` (names +
  types). Done — no guessing. (The example models declare string inputs; we
  surface those verbatim and note coercion happens inside the model.)
- **Custom-function path:** `inspect.signature` → field list; `typing` hints →
  JSON types (`float`→number, `int`→integer, `str`→string, `bool`→boolean);
  parameters without hints default to **string** (matching Domino's verbatim
  string forwarding) and are flagged in the UI as "type unverified".
- **Override:** optional `model_app.yaml` next to the function lets the operator
  pin types, examples, output field names, and an `image` flag (5.5). This is
  the "allow override" half of the decision — introspection first, annotation
  only when needed.
- From the resolved schema we generate (a) a **pydantic model** for request
  validation, (b) a JSON-Schema blob for the UI/playground, and (c) example
  payloads for docs and the "try it" form.

### 5.3 Sync endpoint

- Route `POST {APP_BASE}/models/{slug}/latest/model`, body `{"data": {...}}`.
- Validate against the generated pydantic model (coercing strings like Domino),
  run `adapter.predict([record])`, wrap as `{"result": ..., "request_id": ...,
  "timing_ms": ...}` to echo Domino's response metadata.
- Also accept a bare record (no `data` wrapper) for convenience, but the docs
  show the Domino-compatible `data` form first.

### 5.4 Async endpoints (mirror Domino exactly) + batch scope

- **Submit:** `POST {APP_BASE}/api/modelApis/async/v1/{slug}` with
  `{"parameters": {...}}` → `{"asyncPredictionId": "<task_id>"}`.
- **Poll:** `GET {APP_BASE}/api/modelApis/async/v1/{slug}/{predId}` →
  `{"status": "queued"}` | `{"status":"succeeded","result":{...}}` |
  `{"status":"failed","errors":[...]}`.
- Backed by the async architecture doc's **Plan A**: enqueue row in SQLite,
  stream payload to a dataset blob, single asyncio worker claims + runs the
  chunk in the process pool, writes results, the poll reads status/result.
- **Batch = exactly Domino's async surface, nothing more:**
  - **By value:** a single record (or a small columnar/multi-record payload)
    within the ~10 KB guideline → handled in-process, result returned inline in
    the poll response's `result`.
  - **By reference:** `{"parameters": {"input_file": "<dataset-or-s3 path>"}}`
    for larger inputs → the worker streams the file through the model and writes
    an output file; the poll result carries an `output_file` pointer (plus an
    optional `GET …/{predId}/result` stream for convenience).
  - **Out of scope (unchanged from today):** orchestrating a 10k-row CSV as a
    standalone Domino Job is still the user's path. (Plan B in the async doc is
    the natural home if we ever bring that in-app — deferred.)

### 5.5 Image classification (later phase)

Domino Model APIs take images as a JSON field — typically a **base64-encoded
string** (or a URL/by-reference path), not multipart. We follow that: an `image`
input field (base64 string) → adapter decodes → model → class probabilities. The
UI playground gets a file picker that base64-encodes client-side so the same
JSON endpoint is exercised. Validate against a HF image classifier (e.g.
`google/vit-base-patch16-224` or `microsoft/resnet-50`; torchvision **AlexNet**
also works) as a `CustomFunctionAdapter` whose `predict` runs the HF/torchvision
pipeline. No special transport — just an adapter + an `image` schema flag.

### 5.6 Identity / owner gating

- Domino injects identifying headers into app requests (the run-as user). Read
  them in `core/identity.py`; treat the configured app owner (from
  `DOMINO_PROJECT_OWNER` / project metadata, confirmed during Phase 1 probing)
  as the only identity allowed to change settings.
- Viewers before configuration: every endpoint + the UI return the friendly
  **"This model endpoint is not set up yet."** state. Owners get the Settings
  empty state to pick a model.

### 5.7 Auth in practice — the all-in-one curl

Because endpoints rely on **Domino app auth**, a programmatic caller just needs a
valid **Domino access token** in the `Authorization` header. Inside any Domino
workload (workspace, job, scheduled job, app, launcher) that token is available
ephemerally from the local token proxy — no static key, no model-token issuance:

```bash
# Fetch a short-lived Domino access token (works in any Domino workspace/job)
TOKEN=$(curl -s http://localhost:8899/access-token)
```

The self-doc UI emits a **ready-to-paste, all-in-one snippet per endpoint**, with
the real `{APP_BASE}`, the model `{slug}`, and an example payload built from the
inferred input schema already filled in. Sync example:

```bash
TOKEN=$(curl -s http://localhost:8899/access-token)

curl -X POST "https://apps.<deployment>/apps/<app-name>/models/<slug>/latest/model" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"data": {"calories_wk": "3000.0", "hrs_exercise_wk": "5.0",
               "exercise_intensity": "0.8", "annual_income": "120000.0",
               "num_children": "0.0", "weight": "150.0"}}'
```

Async example (submit → capture id → poll), also generated verbatim:

```bash
TOKEN=$(curl -s http://localhost:8899/access-token)
BASE="https://apps.<deployment>/apps/<app-name>/api/modelApis/async/v1/<slug>"

ID=$(curl -s -X POST "$BASE" \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $TOKEN" \
  -d '{"parameters": {"calories_wk": "3000.0", "...": "..."}}' | jq -r .asyncPredictionId)

curl -s "$BASE/$ID" -H "Authorization: Bearer $TOKEN"   # repeat until status is terminal
```

Notes / caveats:
- `localhost:8899/access-token` only exists **inside** a Domino workload. For
  calls from a laptop/CI, the snippet documents the fallback: paste a Personal
  Access Token (or use a Domino Service Account token) in place of `$TOKEN`. The
  UI shows both forms with a one-line "where am I calling from?" toggle.
- The token is short-lived — the snippet fetches it inline each run rather than
  caching it, so copy/paste "just works" without a stale-token footgun.
- Confirm the proxy port/path (`8899` / `/access-token`) against this deployment
  during Phase 1 probing and template it from an env var, not a hardcode.

### 5.8 How the self-documenting UI works (mechanics)

The UI is **server-rendered HTML + a small amount of vanilla JS** — no SPA, no
build step. It is a *projection of the same `ModelAdapter` + generated schema*
the endpoints use, so docs can never drift from the live API.

- **Data source.** On render, the UI reads the active `ModelAdapter`'s
  `input_schema` / `output_schema` and the route table. Everything on the page
  (field list, types, examples, curl snippets, the playground form) is derived
  from these — there is no hand-maintained doc content.
- **Endpoints page.** A Jinja template loops the routes (sync, async submit,
  async poll) and, per route, renders: method + full URL, the request envelope,
  the field table (name / type / required / example), the response shape, and
  the §5.7 all-in-one curl. Snippets are built server-side from the schema so the
  example payload is always valid for the current model.
- **Playground.** From `input_schema` we render an HTML form (one input per
  field, typed, prefilled with the example). A ~50-line vanilla-JS handler
  serializes the form to the `{"data": {...}}` envelope, `fetch`es the **live**
  endpoint on the same origin (so the browser's Domino app-auth cookie carries
  auth — no token handling in the browser), and pretty-prints the JSON response
  (with status + latency). An "async" toggle switches it to submit-then-poll
  against the async routes, showing the status transitions.
- **Image models.** When the schema marks a field `image`, that field renders as
  a file picker; the JS base64-encodes the file client-side into the same JSON
  field, so the identical endpoint is exercised (§5.5).
- **Styling.** Light Domino theme via the Domino design-system CDN component lib
  (per the `domino-ui-design` guidance). FastAPI's `/docs` (OpenAPI) stays
  available as a secondary, machine-oriented surface; this custom UI is primary.
- **Empty/owner states.** Before configuration the page renders the "not set up
  yet" state (owners additionally see the Settings entry point, §5.6).

---

## 6. Phases

Each phase ends in something runnable. ✅ = ships in that phase, ⏳ = deferred.

### Phase 0 — Scaffold & deploy skeleton  *(half day)*
- FastAPI project layout (`app.py`, `routes/`, `core/`, `services/`), `app.sh`
  for Domino app deployment, `requirements.txt`, base-path/proxy-safe config.
- `GET /health` + a placeholder home page that renders "not set up yet".
- Deploy as a Domino app, confirm the reverse-proxy base path and that app auth
  gates access.
- ✅ App reachable in a browser behind Domino auth. ⏳ everything else.

### Phase 1 — Harness, config store, identity  *(1–2 days)*
- SQLite on the dataset (applying the storage patterns documented in the async
  doc — patterns only, no external code to import), a
  single `app_config` record (active model source + params, or empty).
- `core/identity.py`: owner-vs-viewer from Domino app headers; a probe script to
  confirm which headers/IDs are actually present in this deployment.
- **Probe the token proxy** (`http://localhost:8899/access-token`, §5.7):
  confirm the port/path on this deployment, that the returned token authenticates
  against the app's own routes, and template it from an env var rather than
  hardcoding.
- Config read/write service (owner-only writes).
- ✅ App knows who the owner is and whether it's configured. ⏳ model loading.

### Phase 2 — Adapters + schema inference  *(2–3 days)*
- `RegistryAdapter` (signature → schema) and `CustomFunctionAdapter` (signature
  introspection → schema, with `model_app.yaml` override).
- `core/registry.py`: Domino model-registry API client — list project-scoped
  registered models + versions, fetch a version's signature/artifact. Validate
  against the two known models in this project (diabetes classifier id
  `6a3db2b9a3e64010ef4d3fa2`, weather regressor id `6a3dc165a3e64010ef4d3fa7`).
- `core/schema.py`: schema → pydantic + JSON-Schema + example payloads.
- ✅ Given a model selection, the app can load it and describe its I/O. Unit
  tests over both examples (classification single-prob; regression multi-output).

### Phase 3 — Sync real-time endpoint  *(1–2 days)*
- `POST {APP_BASE}/models/{slug}/latest/model` with the Domino `{"data": …}`
  envelope, pydantic validation + string coercion, `run_in_executor` dispatch,
  Domino-style response wrapper.
- ✅ Identical request body to the real Model APIs returns equivalent results.
  Cross-check responses against the live diabetes/weather Model APIs.

### Phase 4 — Self-documenting UI + playground  *(2–3 days)*
- Build the UI per the mechanics in **§5.8** — server-rendered Jinja + small
  vanilla JS, no build step, projected from the live `ModelAdapter`/schema so
  docs can't drift.
- **Endpoints page:** auto-listed sync + async routes, each with the request/
  response schema, field table, and the **§5.7 all-in-one curl** (token-fetch +
  call) generated server-side with the real URL/slug/example payload.
- **Playground:** schema-generated form → `fetch`es the live endpoint same-origin
  (browser app-auth cookie carries auth; no token handling in JS) → pretty-prints
  JSON with status + latency; "async" toggle does submit-then-poll.
- Keep FastAPI's `/docs` (OpenAPI) available too, but the custom UI is the
  primary, friendlier surface.
- ✅ Visiting the app in a browser explains and exercises the endpoints, and the
  copy/paste curl works from a Domino workspace as-is.

### Phase 5 — Settings tab (owner-only model selection)  *(2 days)*
- Settings page: **empty state** on first run; for the **owner**, a model picker
  populated from `core/registry.py` (registry models + versions) and a
  "custom function" option (file + function path). Saving writes `app_config`
  and warms the adapter.
- Non-owners always see "not set up yet" until configured; after config they see
  the normal Endpoints/Playground UI.
- ✅ Owner configures the model from the UI; the endpoints light up live.

### Phase 6 — Async submit/poll (Plan A)  *(3–4 days)*
- `services/tasks/` worker per the async doc's **Plan A**: SQLite queue, dataset
  JSONL blobs, `spawn` `ProcessPoolExecutor` with per-worker model init, single
  asyncio worker, atomic lease claim + timer heartbeat + resume.
- Routes `POST/GET /api/modelApis/async/v1/{slug}[/{predId}]` with the exact
  Domino `asyncPredictionId` + status contract; by-value and by-reference
  (`input_file`) inputs (§5.4).
- Playground gets an "async" toggle (submit → auto-poll → show result).
- ✅ Every hosted model has the async surface with zero extra owner setup.
- ⏳ **Plan B (Domino Jobs offload)** left as a documented, optional backend
  (`execution_backend` column) for heavy/GPU/untrusted work — build only if a
  real workload needs it.

### Phase 7 — Model-type coverage + image classification  *(2–3 days)*
- Confirm sklearn (weather), PyTorch (diabetes), and registry-native pyfunc all
  work end-to-end through one adapter set.
- Add the **image** input flag + base64 handling (§5.5) and validate a HF/
  torchvision image classifier (ViT/ResNet-50/AlexNet) as a custom adapter.
- ✅ Classification (single + multi-class probs), regression (single + multi),
  and image classification all demonstrably hosted.

### Phase 8 — Docs, README, polish  *(1–2 days)*
- Rewrite the top-level `README.md`: **primary content = this app harness**
  (what it is, deploy, configure, call, sync vs async, auth model, schema
  inference, supported model types). Move the existing example-model write-ups
  to an **Appendix**.
- Error states, input validation messages, empty/owner states, a short
  "compatibility with Domino Model APIs" section.
- ✅ A newcomer can deploy the app, point it at a model, and call it.

---

## 7. Risks & open questions (to resolve as we build)

1. **Domino app identity headers** — exact header/field names for run-as user
   and project owner vary by Domino version; Phase 1 includes a probe to pin
   them. (Auth *enforcement* is Domino's; we only need identity for owner-gating
   Settings.)
2. **Model registry API surface** — confirm the project-scoped "list registered
   models / get version signature / fetch artifact" endpoints against this
   install during Phase 2 (mirrors the caveat in the async doc about
   version-dependent Domino APIs).
3. **Response-metadata fidelity** — we echo `request_id`/`timing` but won't
   reproduce Domino's `release` block exactly; documented as a known delta.
4. **Sync inference that's genuinely slow** — handled by `run_in_executor`, but
   a single huge model loaded once per pool worker has the RAM-×-workers cost
   from the async doc §4d; size the pool accordingly.
5. **By-reference paths** — which storage (Domino dataset vs S3) we accept for
   async `input_file`; default to dataset paths (already mounted), treat S3 as
   later.
6. **One model per app** — if multi-model hosting is wanted, it's an additive
   change (config becomes a list, slug routing already exists), but it's out of
   v1 scope unless you say otherwise.

---

## 8. Appendix — file/dir layout (proposed)

```
app.py                      # FastAPI + lifespan (pool, worker, warmup)
app.sh                      # Domino app entrypoint
requirements.txt
routes/
  ui.py                     # endpoints page + playground (Jinja)
  predict.py                # sync  POST /models/{slug}/latest/model
  async_api.py              # submit+poll /api/modelApis/async/v1/{slug}
  settings.py               # owner-only model selection
  health.py
core/
  adapter.py                # ModelAdapter protocol + concrete adapters
  schema.py                 # signature → pydantic / JSON-Schema / examples
  registry.py               # Domino model-registry API client
  identity.py               # owner vs viewer from Domino app headers
  config.py                 # active-model config (SQLite)
services/
  tasks/                    # async worker (Plan A) — per async_arch doc
templates/                  # Jinja: base, endpoints, playground, settings, empty
static/                     # small vanilla JS + Domino theme assets (CDN-backed)
```

---

## 9. Reference material

- [`async_arch_for_cpu_bound_background_tasks.md`](async_arch_for_cpu_bound_background_tasks.md)
  — the async submit/poll engine (Plan A in-process pool, Plan B Jobs offload).
- `example/diabetes_classer/` — PyTorch binary classifier; custom-code
  (`model_api.py`) and signed-pyfunc (`pyfunc_model.py`) deploy paths.
- `example/weather_regressor/` — scikit-learn multi-output regressor; same two
  paths.
- Live Model APIs for cross-checking request/response shapes: diabetes
  (`/models/6a3db2b9a3e64010ef4d3fa2`), weather
  (`/models/6a3dc165a3e64010ef4d3fa7`).
```
