# Remote client examples

Small, self-contained Python scripts that call a model **hosted by this app**
(the Model-API-as-App harness) from **off-platform** — your laptop, a CI job, or
any machine outside Domino.

Inside a Domino workload you'd just grab a short-lived token from the local proxy
(`curl -s http://localhost:8899/access-token`). Off-platform there's no proxy, so
these examples show the two ways to authenticate from outside:

| Script | Auth | Sync/async | Best for |
| --- | --- | --- | --- |
| [`simple_with_pat.py`](simple_with_pat.py) | Personal Access Token | sync | quick scripts, CI, the simplest possible call |
| [`cli_with_oauth.py`](cli_with_oauth.py) | Browser sign-in (OAuth) | sync | interactive use on a laptop; no token to manage |
| [`async_with_oauth.py`](async_with_oauth.py) | Browser sign-in (OAuth) | async submit/poll | longer / bulk jobs, by-reference file inputs |

All three send the **same request envelope** a real Domino Model API expects, so
the body you build here is the body you'd send anywhere.

---

## Prerequisites

```bash
pip install requests
```

(or `pip install -r ../requirements.txt` for the whole project)

You also need the **endpoint URL** for your model, which the app prints on its
**Endpoints** page — including a ready-to-copy example payload. There's one URL
per shape:

- **Sync**: `https://apps.<deployment>/apps/<app-id>/models/<slug>/latest/model`
- **Async**: `https://apps.<deployment>/apps/<app-id>/api/modelApis/async/v1/<slug>`

---

## 1. `simple_with_pat.py` — PAT, synchronous

The minimal example. Authenticates by passing a Domino **Personal Access Token**
straight through as the bearer token.

```bash
export DOMINO_PAT="<your Domino personal access token>"
export MODEL_API_URL="https://apps.<deployment>/apps/<app-id>/models/<slug>/latest/model"
python simple_with_pat.py
```

- `DOMINO_PAT` (required) — create one in Domino under **Account Settings → API Keys / Tokens**.
- `MODEL_API_URL` (optional) — overrides the URL hardcoded near the top of the file.

This script keeps its endpoint and payload as plain constants at the top of the
file — edit them directly (see [Tweaking the payload](#tweaking-the-payload-for-your-own-model)).

---

## 2. `cli_with_oauth.py` — browser OAuth, synchronous

A friendlier alternative with no token to copy/paste. The **first run** prompts
for your Domino instance URL and your model endpoint URL, then opens a browser to
sign in (Keycloak, Authorization Code + PKCE on a localhost callback). It caches
both URLs and the tokens, so later runs are non-interactive.

```bash
python cli_with_oauth.py            # first run: prompts + browser; then cached
python cli_with_oauth.py --login    # force a fresh browser sign-in
python cli_with_oauth.py --logout   # clear ALL settings (tokens + saved URLs)
python cli_with_oauth.py --help     # usage + the state-file path
```

**Short-lived tokens are handled for you.** Domino access tokens last ~5 minutes;
the script refreshes silently with the stored offline refresh token, and if that
is gone too it re-opens the browser automatically — reusing the saved instance
URL so it never re-asks for the host. A token rejected mid-call (401/403)
triggers the same recovery plus one retry.

If the browser doesn't open, the script prints the sign-in URL to paste manually.

---

## 3. `async_with_oauth.py` — browser OAuth, async submit/poll

The async sibling. It **reuses `cli_with_oauth.py`'s** auth + settings helpers
(so the two scripts must live in the same directory) and speaks Domino's async
Model API contract: `POST` to submit → get an `asyncPredictionId` → `GET` poll
until the status is terminal (`succeeded | failed | cancelled | expired`).

```bash
# By value: submit the RECORD at the top of the file, then poll to completion
python async_with_oauth.py

# By reference: submit a dataset file path; stream the output when it's done
python async_with_oauth.py --input-file data.jsonl
python async_with_oauth.py --input-file in.csv --output out.jsonl

python async_with_oauth.py --login | --logout | --help
```

- `--input-file <path>` — a dataset path **on Domino** (`.jsonl`/`.csv`) for bulk
  inputs; the result comes back as an `output_file` pointer that the script then
  streams.
- `--output <path>` — write the streamed by-reference output here instead of stdout.
- `MODEL_API_TIMEOUT` (env, seconds, default `600`) — how long to keep polling
  before giving up.
- `Ctrl-C` while polling best-effort cancels the in-flight prediction.

Because it fetches a freshly-refreshed token on **every** request, a long poll
loop stays authenticated even past the access-token lifetime.

---

## Tweaking the payload for your own model

Every model has its own input schema. The example records here match the
`weather_regressor` sample (`month`, `state`, `precipitation`, …) — point a
script at a different model and you'll need to change the fields.

**Where to edit**, near the top of each file:

- `simple_with_pat.py` and `cli_with_oauth.py` → the `PAYLOAD` constant
  (sync wraps the record in `{"data": {...}}`):

  ```python
  PAYLOAD = {
      "data": {
          "your_field_a": "...",
          "your_field_b": "...",
      }
  }
  ```

- `async_with_oauth.py` → the `RECORD` constant
  (async wraps it in `{"parameters": {...}}` for you):

  ```python
  RECORD = {
      "your_field_a": "...",
      "your_field_b": "...",
  }
  ```

**The easiest way to get this right:** open your model in a browser and copy the
example payload straight off its **Endpoints** page (or use the **Playground** to
shape a working request), then paste the field values in.

**If the payload is wrong**, the scripts now print a readable explanation instead
of a raw stack trace — the fields you sent, the ones that are missing or invalid,
and what to do:

```
Request failed: HTTP 422 Unprocessable Entity

The payload doesn't match this model's expected input schema.
  You sent:         month, week_of, state, precipitation, wind_speed, wind_direction
  Missing required: calories_wk, hrs_exercise_wk, exercise_intensity, ...

What to do:
  1. Open the app in a browser and check its Endpoints page ...
  2. Update the record at the top of this script to match, then re-run.
```

---

## Where settings are stored

The two OAuth scripts share a single cache file:

```
~/.domino/model_api_auth.json   (mode 0600)
```

It holds your tokens **and** the saved instance/endpoint URLs. To wipe everything
and start fresh, run `--logout` (or just `rm` the file):

```bash
python cli_with_oauth.py --logout
```

> Note: environment variables (`MODEL_API_URL`, `MODEL_API_ASYNC_URL`,
> `DOMINO_URL`) always take precedence over the saved values and are never
> written to disk — unset them if you want to be re-prompted.

---

## Environment variables (reference)

| Variable | Used by | Purpose |
| --- | --- | --- |
| `DOMINO_PAT` | `simple_with_pat.py` | Personal Access Token (required) |
| `MODEL_API_URL` | `simple_with_pat.py`, `cli_with_oauth.py` | sync endpoint URL override |
| `MODEL_API_ASYNC_URL` | `async_with_oauth.py` | async base URL override |
| `DOMINO_URL` | OAuth scripts | Domino instance URL override (for sign-in) |
| `MODEL_API_TIMEOUT` | `async_with_oauth.py` | poll timeout in seconds (default 600) |
| `DOMINO_CA_BUNDLE` *(or `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` / `CURL_CA_BUNDLE`)* | OAuth scripts | custom CA bundle for private-CA deployments |
| `DOMINO_INSECURE=1` | OAuth scripts | disable TLS verification (testing only) |
