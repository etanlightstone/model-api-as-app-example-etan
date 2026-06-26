"""Async submit/poll against the model (hosted as a Domino App) via browser OAuth.

The async sibling of `cli_with_oauth.py`. It reuses that module's auth + settings
helpers (browser sign-in, token cache + silent refresh, saved-URL prompts) and
speaks Domino's async Model API contract:

    POST {ASYNC_BASE}            body {"parameters": {...}}  -> {"asyncPredictionId": id}
    GET  {ASYNC_BASE}/{id}       -> {"status": "queued|running|succeeded|failed|..."}
    GET  {ASYNC_BASE}/{id}/result   (by-reference output stream)

where ASYNC_BASE is `.../api/modelApis/async/v1/<slug>`. On first run it prompts
for (and saves) that base URL, separately from the sync endpoint the other
example uses, so the two don't clobber each other.

Because every HTTP call fetches a freshly-refreshed token, a long poll loop stays
authenticated even past the ~5-min access-token lifetime; a token rejected
mid-flight (401/403) forces a re-auth and one retry.

    python async_with_oauth.py                       # by-value submit + poll
    python async_with_oauth.py --input-file data.jsonl   # by-reference (bulk)
    python async_with_oauth.py --input-file in.csv --output out.jsonl
    python async_with_oauth.py --login | --logout

Env overrides: MODEL_API_ASYNC_URL (base), DOMINO_URL (host).
"""

from __future__ import annotations

import os
import sys
import time

import requests

from cli_with_oauth import TOKEN_FILE, _verify, get_saved_url, get_valid_token

# Status values that mean the prediction is done (mirrors services/tasks/service.py).
TERMINAL = {"succeeded", "failed", "cancelled", "expired"}

# How long to keep polling before giving up. Override via MODEL_API_TIMEOUT (s).
POLL_TIMEOUT = float(os.environ.get("MODEL_API_TIMEOUT", "600"))

# By-value input record. Same fields as the sync example, but async wraps it in
# {"parameters": ...} instead of {"data": ...}.
RECORD = {
    "month": "0",
    "week_of": "0",
    "state": "example",
    "precipitation": "0.0",
    "wind_speed": "0.0",
    "wind_direction": "0.0",
}


def get_async_base() -> str:
    return get_saved_url(
        "async_url",
        "MODEL_API_ASYNC_URL",
        "Which async endpoint should I call? (the async base URL from the app's Endpoints page)",
        "https://apps.your-company.domino.tech/apps/<app-id>/api/modelApis/async/v1/<slug>",
    )


def _authed(method: str, url: str, **kwargs) -> requests.Response:
    """Make a request with a fresh token, retrying once if the token is rejected.

    get_valid_token() auto-refreshes within the expiry margin, so calling it on
    every request keeps even a long-running poll loop authenticated for free.
    """
    headers = {**kwargs.pop("headers", {}), "Authorization": f"Bearer {get_valid_token()}"}
    resp = requests.request(method, url, headers=headers, verify=_verify(), **kwargs)
    if resp.status_code in (401, 403):
        print("Token rejected; re-authenticating and retrying once...")
        headers["Authorization"] = f"Bearer {get_valid_token(force_login=True)}"
        resp = requests.request(method, url, headers=headers, verify=_verify(), **kwargs)
    return resp


def submit(base: str, parameters: dict) -> str:
    resp = _authed(
        "POST", base,
        headers={"Content-Type": "application/json"},
        json={"parameters": parameters},
        timeout=30,
    )
    if not resp.ok:
        sys.exit(f"Submit failed: {resp.status_code} {resp.reason}\n{resp.text}")
    pred_id = resp.json().get("asyncPredictionId")
    if not pred_id:
        sys.exit(f"Submit returned no asyncPredictionId: {resp.text}")
    return pred_id


def poll(base: str, pred_id: str, timeout: float) -> dict:
    delay, deadline = 1.0, time.time() + timeout
    while True:
        resp = _authed("GET", f"{base}/{pred_id}", timeout=30)
        if not resp.ok:
            sys.exit(f"Poll failed: {resp.status_code} {resp.reason}\n{resp.text}")
        body = resp.json()
        status = body.get("status")
        if status in TERMINAL:
            return body

        progress = body.get("progress")
        line = f"  status={status}"
        if progress:
            line += f"  ({progress.get('completed_items')}/{progress.get('total_items')})"
        print(line)

        if time.time() >= deadline:
            sys.exit(f"Timed out after {timeout:.0f}s (last status={status}).")
        time.sleep(delay)
        delay = min(delay * 1.5, 10.0)


def stream_result(base: str, pred_id: str, out_path: str | None) -> None:
    """Stream a by-reference task's output JSONL to a file or stdout."""
    resp = _authed("GET", f"{base}/{pred_id}/result", stream=True, timeout=300)
    if not resp.ok:
        sys.exit(f"Result fetch failed: {resp.status_code} {resp.reason}\n{resp.text}")
    if out_path:
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(65536):
                fh.write(chunk)
        print(f"Wrote results to {out_path}")
    else:
        for chunk in resp.iter_content(65536):
            sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()


def best_effort_cancel(base: str, pred_id: str) -> None:
    try:
        _authed("POST", f"{base}/{pred_id}/cancel", timeout=15)
        print(f"\nRequested cancellation of {pred_id}.")
    except requests.RequestException:
        pass


def _arg_value(argv: list[str], flag: str) -> str | None:
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
        sys.exit(f"{flag} requires a value.")
    return None


def main(argv: list[str]) -> int:
    if "--logout" in argv:
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)
            print("Cached tokens and saved URLs removed.")
        else:
            print("Nothing cached to remove.")
        return 0

    input_file = _arg_value(argv, "--input-file")
    output = _arg_value(argv, "--output")

    # Prime auth (handles first-run prompts + browser) before we start.
    get_valid_token(force_login="--login" in argv)
    base = get_async_base()

    parameters = {"input_file": input_file} if input_file else RECORD
    pred_id = submit(base, parameters)
    print(f"Submitted. asyncPredictionId = {pred_id}\nPolling for completion...")

    try:
        final = poll(base, pred_id, POLL_TIMEOUT)
    except KeyboardInterrupt:
        best_effort_cancel(base, pred_id)
        return 130

    status = final.get("status")
    if status != "succeeded":
        errors = final.get("errors")
        print(f"Prediction {status}." + (f" {errors}" if errors else ""), file=sys.stderr)
        return 1

    result = final.get("result", {})
    # By-reference results come back as a pointer; stream the actual output.
    if isinstance(result, dict) and result.get("output_file"):
        print(f"Done. {result.get('completed_items', '?')} item(s); streaming output:")
        stream_result(base, pred_id, output)
    else:
        print("Done. Result:")
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
