"""Call the model (hosted as a Domino App) from off-platform via browser OAuth.

A friendlier alternative to pasting a PAT (`simple_with_pat.py`): the first run
prompts for your Domino instance URL *and* the model endpoint URL, then opens a
browser to sign in (Keycloak, Authorization Code + PKCE on a localhost callback).
Both URLs and the tokens are cached on disk, so later runs are non-interactive.

Domino access tokens are short-lived (~5 min). This script handles that for you:
it refreshes silently with the stored offline refresh token, and if that's gone
too it re-opens the browser automatically — reusing the saved instance URL, so it
never re-asks for the host. A token rejected mid-call (401/403) triggers the same
recovery and one retry.

    python cli_with_oauth.py            # first run: prompts + browser; then cached
    python cli_with_oauth.py --login    # force a fresh browser sign-in
    python cli_with_oauth.py --logout   # forget cached tokens + saved URLs

The companion `async_with_oauth.py` reuses this module's auth + settings helpers.
Env vars override the saved prompts: MODEL_API_URL (endpoint), DOMINO_URL (host).
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import time
import urllib.parse
import webbrowser

import requests

# --- Keycloak / client config (matches Domino's CLI + VS Code extension) ---
REALM = "DominoRealm"
CLIENT_ID = "domino-connect-client"
SCOPES = "openid profile email domino-jwt-claims offline_access"
AUTH_TIMEOUT = 300  # seconds to wait for the browser sign-in
REFRESH_MARGIN = 30  # refresh this many seconds before the access token expires

# --- Where we cache tokens + saved URLs. Per-user dotdir, file mode 0600. ---
TOKEN_DIR = os.path.expanduser("~/.domino")
TOKEN_FILE = os.path.join(TOKEN_DIR, "model_api_auth.json")

# The input record. Shape must match the model's schema (see the app's Endpoints
# page for the exact fields and an example payload).
PAYLOAD = {
    "data": {
        "month": "0",
        "week_of": "0",
        "state": "example",
        "precipitation": "0.0",
        "wind_speed": "0.0",
        "wind_direction": "0.0",
    }
}


# --- TLS handling: honor a custom CA bundle, or opt-in insecure for odd setups.
def _verify():
    for var in ("DOMINO_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
        path = os.environ.get(var)
        if path and os.path.exists(path):
            return path
    if os.environ.get("DOMINO_INSECURE", "").lower() in ("1", "true", "yes"):
        return False
    return True


def _keycloak_base(host: str) -> str:
    return f"{host}/auth/realms/{REALM}/protocol/openid-connect"


# --- On-disk state: tokens + saved URLs, all in one 0600 file -----------------
def load_state() -> dict | None:
    if not os.path.exists(TOKEN_FILE):
        return None
    try:
        with open(TOKEN_FILE) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def update_state(**fields) -> dict:
    """Merge fields into the on-disk state (creating it if needed) and return it.

    Merging (not overwriting) is what lets the sync + async examples share one
    cache file without clobbering each other's saved endpoint URLs.
    """
    state = load_state() or {}
    state.update(fields)
    os.makedirs(TOKEN_DIR, mode=0o700, exist_ok=True)
    with open(TOKEN_FILE, "w") as fh:
        json.dump(state, fh, indent=2)
    os.chmod(TOKEN_FILE, 0o600)
    return state


def store_tokens(host: str, tokens: dict) -> str:
    """Persist tokens and return the fresh access token."""
    update_state(
        domino_url=host,
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token"),
        expires_at=time.time() + tokens.get("expires_in", 300),
        token_type=tokens.get("token_type", "Bearer"),
        saved_at=time.time(),
    )
    return tokens["access_token"]


# --- PKCE + browser sign-in ---------------------------------------------------
def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return
        params = urllib.parse.parse_qs(parsed.query)
        self.server.oauth_code = params.get("code", [None])[0]
        self.server.oauth_state = params.get("state", [None])[0]
        self.server.oauth_error = params.get("error", [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<html><body style='font-family:system-ui;text-align:center;margin-top:4rem'>"
            b"<h2>Sign-in complete</h2><p>You can close this tab and return to your terminal.</p>"
            b"</body></html>"
        )

    def log_message(self, *args):  # silence the default request logging
        pass


def _open_browser(url: str) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            webbrowser.open(url)
    except Exception:
        pass  # the manual URL is always printed as a fallback


def browser_login(host: str) -> dict:
    """Run the Authorization Code + PKCE flow via a localhost callback."""
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(32)
    port = _free_port()
    redirect_uri = f"http://localhost:{port}/callback"
    auth_url = f"{_keycloak_base(host)}/auth?" + urllib.parse.urlencode(
        {
            "client_id": CLIENT_ID,
            "response_type": "code",
            "scope": SCOPES,
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
    )

    server = http.server.HTTPServer(("127.0.0.1", port), _CallbackHandler)
    server.oauth_code = server.oauth_state = server.oauth_error = None

    print("\nOpening your browser to sign in to Domino...")
    print("If it doesn't open, paste this URL manually:\n")
    print(f"  {auth_url}\n")
    _open_browser(auth_url)

    deadline = time.time() + AUTH_TIMEOUT
    while server.oauth_code is None and server.oauth_error is None:
        remaining = deadline - time.time()
        if remaining <= 0:
            server.server_close()
            sys.exit("Timed out waiting for browser sign-in.")
        server.timeout = remaining
        server.handle_request()
    server.server_close()

    if server.oauth_error:
        sys.exit(f"Sign-in failed: {server.oauth_error}")
    if server.oauth_state != state:
        sys.exit("Sign-in failed: state mismatch (possible CSRF).")

    resp = requests.post(
        f"{_keycloak_base(host)}/token",
        data={
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": server.oauth_code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        },
        timeout=30,
        verify=_verify(),
    )
    resp.raise_for_status()
    print("Signed in.\n")
    return resp.json()


def refresh_tokens(host: str, refresh_token: str) -> dict:
    resp = requests.post(
        f"{_keycloak_base(host)}/token",
        data={
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
        },
        timeout=30,
        verify=_verify(),
    )
    resp.raise_for_status()
    return resp.json()


def _prompt_url(question: str, example: str) -> str:
    print(f"\n{question}")
    print(f"  e.g. {example}")
    while True:
        raw = input("> ").strip().rstrip("/")
        if raw.startswith("http://") or raw.startswith("https://"):
            return raw
        print("  Please enter a full URL starting with https://")


def _prompt_host() -> str:
    return _prompt_url(
        "First-time setup: which Domino instance are you using?",
        "https://your-company.domino.tech",
    )


def get_saved_url(key: str, env_var: str, question: str, example: str) -> str:
    """Return a saved URL setting, prompting once and persisting it if unset.

    Resolution order: env var override → value saved in the state file → prompt
    the user and save. Shared by both examples (each passes its own key/prompt).
    """
    override = os.environ.get(env_var)
    if override:
        return override.rstrip("/")
    state = load_state() or {}
    if state.get(key):
        return state[key]
    url = _prompt_url(question, example)
    update_state(**{key: url})
    return url


def get_model_url() -> str:
    return get_saved_url(
        "model_url",
        "MODEL_API_URL",
        "Which model endpoint should I call? (the sync URL from the app's Endpoints page)",
        "https://apps.your-company.domino.tech/apps/<app-id>/models/<slug>/latest/model",
    )


def get_valid_token(force_login: bool = False) -> str:
    """Return a usable access token, signing in / refreshing as needed.

    Recovery ladder: cached-and-fresh → refresh with stored token → browser
    sign-in. The host is only ever asked for on the very first run.
    """
    state = load_state()
    if not state or not state.get("access_token") or not state.get("domino_url"):
        host = (state or {}).get("domino_url") or _prompt_host()
        return store_tokens(host, browser_login(host))

    host = state["domino_url"]
    if not force_login and time.time() < state.get("expires_at", 0) - REFRESH_MARGIN:
        return state["access_token"]

    refresh_token = state.get("refresh_token")
    if refresh_token:
        try:
            return store_tokens(host, refresh_tokens(host, refresh_token))
        except requests.RequestException:
            print("Session expired; re-opening the browser to sign in...")

    return store_tokens(host, browser_login(host))


def _parse_validation_errors(detail: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Pull (missing_fields, [(field, message)]) out of a pydantic error dump."""
    missing: list[str] = []
    other: list[tuple[str, str]] = []
    lines = detail.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        is_field = (
            bool(line)
            and not line[0].isspace()
            and not line.lower().startswith("record")
            and "validation error" not in line.lower()
        )
        if is_field and i + 1 < n and lines[i + 1][:1].isspace():
            field = line.strip()
            type_match = re.search(r"\[type=([^,\]]+)", lines[i + 1])
            etype = type_match.group(1) if type_match else ""
            message = re.sub(r"\s*\[type=.*$", "", lines[i + 1].strip())
            if etype == "missing":
                missing.append(field)
            else:
                other.append((field, message or etype))
            i += 2
            continue
        i += 1
    return missing, other


def format_error(resp: requests.Response, sent_fields: list[str] | None = None) -> str:
    """Render a server error for humans, with guidance for schema mismatches."""
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text

    out = [f"Request failed: HTTP {resp.status_code} {resp.reason}"]
    if resp.status_code == 422 and isinstance(detail, str) and "validation error" in detail.lower():
        missing, other = _parse_validation_errors(detail)
        out += ["", "The payload doesn't match this model's expected input schema."]
        if sent_fields is not None:
            out.append(f"  You sent:         {', '.join(sent_fields) or '(none)'}")
        if missing:
            out.append(f"  Missing required: {', '.join(missing)}")
        for field, message in other:
            out.append(f"  Invalid '{field}': {message}")
        out += [
            "",
            "What to do:",
            "  1. Open the app in a browser and check its Endpoints page — it lists this",
            "     model's exact fields and a ready-to-copy example payload.",
            "  2. Update the record at the top of this script to match, then re-run.",
        ]
    else:
        out.append(str(detail))
    return "\n".join(out)


def call_model(url: str, token: str) -> requests.Response:
    return requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json=PAYLOAD,
        timeout=30,
        verify=_verify(),
    )


def main(argv: list[str]) -> int:
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        print(f"State file (tokens + saved URLs): {TOKEN_FILE}")
        return 0

    if "--logout" in argv:
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)
            print(f"Cleared all settings (tokens + saved URLs): {TOKEN_FILE}")
        else:
            print("Nothing cached to remove.")
        return 0

    token = get_valid_token(force_login="--login" in argv)
    url = get_model_url()
    resp = call_model(url, token)

    # A token can be rejected even when we think it's fresh (clock skew, server
    # restart, revoked session). Recover once by forcing a new sign-in.
    if resp.status_code in (401, 403):
        print("Token rejected; re-authenticating and retrying once...")
        token = get_valid_token(force_login=True)
        resp = call_model(url, token)

    if not resp.ok:
        print(format_error(resp, sent_fields=list(PAYLOAD["data"].keys())), file=sys.stderr)
        return 1

    print(resp.json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
