"""Call the model (hosted as a Domino App) from off-platform via browser OAuth.

A friendlier alternative to pasting a PAT (`simple_with_pat.py`): the first run
prompts for your Domino instance URL and opens a browser to sign in (Keycloak,
Authorization Code + PKCE on a localhost callback). Tokens are cached on disk, so
later runs are non-interactive.

Domino access tokens are short-lived (~5 min). This script handles that for you:
it refreshes silently with the stored offline refresh token, and if that's gone
too it re-opens the browser automatically — reusing the saved instance URL, so it
never re-asks for the host. A token rejected mid-call (401/403) triggers the same
recovery and one retry.

    python cli_with_oauth.py            # first run: prompts + browser; then cached
    python cli_with_oauth.py --login    # force a fresh browser sign-in
    python cli_with_oauth.py --logout   # forget the cached tokens

Set MODEL_API_URL to point at a different model endpoint.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
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

# --- Where we cache tokens. Conventional per-user dotdir, file mode 0600. ---
TOKEN_DIR = os.path.expanduser("~/.domino")
TOKEN_FILE = os.path.join(TOKEN_DIR, "model_api_auth.json")

# The model's sync endpoint. Override per deployment via MODEL_API_URL.
MODEL_API_URL = os.environ.get(
    "MODEL_API_URL",
    "https://apps.cloud-dogfood.domino.tech/apps/91b27ca1-7996-4b7a-b966-e99b30b9cc0e/models/weathclasser/latest/model",
)

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


# --- Token cache --------------------------------------------------------------
def load_state() -> dict | None:
    if not os.path.exists(TOKEN_FILE):
        return None
    try:
        with open(TOKEN_FILE) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def store_tokens(host: str, tokens: dict) -> str:
    """Persist tokens (mode 0600) and return the fresh access token."""
    os.makedirs(TOKEN_DIR, mode=0o700, exist_ok=True)
    state = {
        "domino_url": host,
        "access_token": tokens["access_token"],
        "refresh_token": tokens.get("refresh_token"),
        "expires_at": time.time() + tokens.get("expires_in", 300),
        "token_type": tokens.get("token_type", "Bearer"),
        "saved_at": time.time(),
    }
    with open(TOKEN_FILE, "w") as fh:
        json.dump(state, fh, indent=2)
    os.chmod(TOKEN_FILE, 0o600)
    return state["access_token"]


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


def _prompt_host() -> str:
    print("First-time setup: which Domino instance are you using?")
    print("  e.g. https://your-company.domino.tech")
    while True:
        raw = input("Domino instance URL: ").strip().rstrip("/")
        if raw.startswith("http://") or raw.startswith("https://"):
            return raw
        print("  Please enter a full URL starting with https://")


def get_valid_token(force_login: bool = False) -> str:
    """Return a usable access token, signing in / refreshing as needed.

    Recovery ladder: cached-and-fresh → refresh with stored token → browser
    sign-in. The host is only ever asked for on the very first run.
    """
    state = load_state()
    if state is None:
        host = _prompt_host()
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


def call_model(token: str) -> requests.Response:
    return requests.post(
        MODEL_API_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json=PAYLOAD,
        timeout=30,
        verify=_verify(),
    )


def main(argv: list[str]) -> int:
    if "--logout" in argv:
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)
            print("Cached tokens removed.")
        else:
            print("No cached tokens to remove.")
        return 0

    token = get_valid_token(force_login="--login" in argv)
    resp = call_model(token)

    # A token can be rejected even when we think it's fresh (clock skew, server
    # restart, revoked session). Recover once by forcing a new sign-in.
    if resp.status_code in (401, 403):
        print("Token rejected; re-authenticating and retrying once...")
        token = get_valid_token(force_login=True)
        resp = call_model(token)

    if not resp.ok:
        print(f"Request failed: {resp.status_code} {resp.reason}", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        return 1

    print(resp.json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
