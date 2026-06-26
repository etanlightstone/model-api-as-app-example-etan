"""Caller identity + owner gating.

Domino enforces *authentication* at its app reverse proxy — everyone who reaches
the app has already passed Domino auth. We only need *identity* here to gate the
Settings page so only the project owner can change the hosted model.

The proxy forwards the authenticated user in a request header, but **the header
name varies by Domino version** and **the value's format varies** (username vs
email vs OIDC subject). So we:

* build a rich set of owner identifiers — from ``DOMINO_PROJECT_OWNER`` and, when
  reachable, the live Domino ``/v4/users/self`` profile (userName, email,
  id, idpId) — and a normalized form that treats ``etan_lightstone`` and
  ``etan.lightstone`` (and the email local-part) as equal;
* check a broad, curated list of identity headers (overridable via
  ``MODEL_APP_USER_HEADER``) and match any value against that owner set;
* expose ``probe_headers`` (``/settings/whoami``) that shows every header, the
  owner set, and exactly which header matched — so an operator can pin
  ``MODEL_APP_USER_HEADER`` for their deployment in one look.

Safety valves the owner controls at deploy time:
* ``MODEL_APP_OWNERS`` — comma-separated extra owner identifiers (usernames/emails).
* ``MODEL_APP_SETTINGS_ACCESS=all`` — treat every authenticated viewer as owner
  (use only when Domino's app sharing already restricts who can reach the app).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from starlette.requests import Request

from core import settings

# Identity headers Domino / its auth proxy (oauth2-proxy, Keycloak) have used.
# An explicit override via MODEL_APP_USER_HEADER takes precedence.
_DEFAULT_HEADERS = [
    # Domino's app proxy sends the run-as user here, WITHOUT an x- prefix
    # (confirmed on cloud-dogfood). Keep these first.
    "domino-username", "domino-user", "domino-user-name", "domino-email",
    "domino-user-id", "domino-runas-username",
    "x-domino-username", "x-domino-user", "x-domino-runas-username",
    "x-domino-user-name", "x-domino-user-id",
    "x-forwarded-user", "x-forwarded-email", "x-forwarded-preferred-username",
    "x-auth-request-user", "x-auth-request-email", "x-auth-request-preferred-username",
    "x-webauth-user", "x-remote-user", "remote-user", "x-user", "x-username",
    "oidc-claim-preferred-username", "oidc-claim-email",
]
_OVERRIDE_HEADERS = [h.strip().lower() for h in
                     os.environ.get("MODEL_APP_USER_HEADER", "").split(",") if h.strip()]
_CANDIDATE_USER_HEADERS = _OVERRIDE_HEADERS or _DEFAULT_HEADERS

_SETTINGS_ACCESS = os.environ.get("MODEL_APP_SETTINGS_ACCESS", "owner").strip().lower()


@dataclass
class Caller:
    username: str
    is_owner: bool
    source: str  # which header (or "dev"/"access=all"/"unknown") the decision came from


def _norm(value: str) -> str:
    """Normalize an identifier for comparison: lowercase, strip separators.

    Makes ``etan_lightstone`` == ``etan.lightstone`` == ``Etan-Lightstone`` and,
    for emails, lets the local-part match a username.
    """
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


# --- owner identity set (cached) --------------------------------------------

_owner_norms: set[str] | None = None
_owner_raw: list[str] = []


def _fetch_self_profile() -> dict:
    """Best-effort live profile of the app's run-as user (= owner for an app)."""
    if not (settings.DOMINO_API_HOST and settings.DOMINO_USER_API_KEY):
        return {}
    try:
        import requests

        r = requests.get(
            f"{settings.DOMINO_API_HOST}/v4/users/self",
            headers={"X-Domino-Api-Key": settings.DOMINO_USER_API_KEY},
            timeout=5,
        )
        if r.status_code == 200:
            return r.json() or {}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _build_owner_set() -> tuple[set[str], list[str]]:
    raw: list[str] = []
    for v in (settings.PROJECT_OWNER, os.environ.get("DOMINO_STARTING_USERNAME", "")):
        if v:
            raw.append(v)
    raw += [v.strip() for v in os.environ.get("MODEL_APP_OWNERS", "").split(",") if v.strip()]

    prof = _fetch_self_profile()
    for key in ("userName", "email", "id", "idpId", "canonicalName"):
        if prof.get(key):
            raw.append(str(prof[key]))

    norms: set[str] = set()
    for v in raw:
        norms.add(_norm(v))
        if "@" in v:                      # also index an email's local-part
            norms.add(_norm(v.split("@", 1)[0]))
    norms.discard("")
    return norms, sorted(set(raw))


def owner_identifiers() -> tuple[set[str], list[str]]:
    global _owner_norms, _owner_raw
    if _owner_norms is None:
        _owner_norms, _owner_raw = _build_owner_set()
    return _owner_norms, _owner_raw


def reset_owner_cache() -> None:
    global _owner_norms
    _owner_norms = None


def _candidate_identities(request: Request) -> list[tuple[str, str]]:
    """All (value, header_name) identity candidates present on the request."""
    found = []
    for name in _CANDIDATE_USER_HEADERS:
        val = request.headers.get(name)
        if val:
            found.append((val.strip(), name))
    return found


def _value_matches_owner(value: str) -> bool:
    norms, _ = owner_identifiers()
    if _norm(value) in norms:
        return True
    if "@" in value and _norm(value.split("@", 1)[0]) in norms:
        return True
    return False


def resolve_caller(request: Request) -> Caller:
    """Determine who is calling and whether they own the project."""
    candidates = _candidate_identities(request)
    username = candidates[0][0] if candidates else ""
    source = candidates[0][1] if candidates else "unknown"

    # Local/dev with no proxy in front: trust the caller as owner.
    if settings.DEV_TREAT_CALLER_AS_OWNER:
        return Caller(username or settings.PROJECT_OWNER or "dev", True, "dev")

    # Operator opt-in: anyone who can reach the app may configure it.
    if _SETTINGS_ACCESS == "all":
        return Caller(username or "viewer", True, "access=all")

    for value, header in candidates:
        if _value_matches_owner(value):
            return Caller(value, True, header)

    return Caller(username, False, source)


def probe_headers(request: Request) -> dict:
    """Diagnostic for pinning the identity header on a deployment (/settings/whoami)."""
    norms, raw = owner_identifiers()
    candidates = _candidate_identities(request)
    resolved = resolve_caller(request)
    return {
        "resolved_username": resolved.username,
        "resolved_is_owner": resolved.is_owner,
        "resolved_from": resolved.source,
        "configured_owner_env": settings.PROJECT_OWNER,
        "owner_identifiers": raw,
        "settings_access_mode": _SETTINGS_ACCESS,
        "dev_owner_mode": settings.DEV_TREAT_CALLER_AS_OWNER,
        "identity_headers_checked": _CANDIDATE_USER_HEADERS,
        "identity_headers_found": [
            {"header": h, "value": v, "matches_owner": _value_matches_owner(v)}
            for v, h in candidates
        ],
        "all_request_headers": {k: v for k, v in request.headers.items()},
        "hint": (
            "If 'identity_headers_found' is empty, your Domino app proxy isn't "
            "sending any header in the checked list — copy the real header name "
            "from 'all_request_headers' into the MODEL_APP_USER_HEADER env var. "
            "If a header is found but matches_owner is false, add its value to "
            "MODEL_APP_OWNERS. As a last resort set MODEL_APP_SETTINGS_ACCESS=all."
        ),
    }
