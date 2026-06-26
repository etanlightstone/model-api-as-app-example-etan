"""Caller identity + owner gating.

Domino enforces *authentication* at its app reverse proxy — everyone who can
reach the app has already passed Domino auth. We only need *identity* here for
one thing: gating the Settings page so that only the project owner can change
which model the app hosts.

Domino forwards the run-as user in a request header, but the exact header name
varies by Domino version (the plan flags this as a Phase 1 unknown). So we:

* check a list of candidate header names (overridable via
  ``MODEL_APP_USER_HEADER``),
* compare the resolved username to ``DOMINO_PROJECT_OWNER``,
* expose ``probe_headers`` so an operator can pin the real header name on their
  deployment from the live request (surfaced at ``/settings/whoami``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from starlette.requests import Request

from core import settings

# Header names Domino has used across versions for the run-as user. The first
# one that yields a value wins. An explicit override takes precedence.
_CANDIDATE_USER_HEADERS = [
    h.strip()
    for h in os.environ.get("MODEL_APP_USER_HEADER", "").split(",")
    if h.strip()
] or [
    "x-domino-username",
    "x-domino-user",
    "x-domino-runas-username",
    "x-forwarded-user",
    "x-forwarded-preferred-username",
    "x-auth-request-user",
    "remote-user",
]


@dataclass
class Caller:
    username: str
    is_owner: bool
    source: str  # which header (or "dev"/"unknown") the identity came from


def _header_username(request: Request) -> tuple[str, str]:
    for name in _CANDIDATE_USER_HEADERS:
        val = request.headers.get(name)
        if val:
            return val.strip(), name
    return "", ""


def resolve_caller(request: Request) -> Caller:
    """Determine who is calling and whether they own the project."""
    username, source = _header_username(request)

    if settings.DEV_TREAT_CALLER_AS_OWNER:
        # Local/dev: no Domino proxy in front, so trust the caller as owner.
        return Caller(username=username or settings.PROJECT_OWNER or "dev",
                      is_owner=True, source="dev")

    if username:
        is_owner = bool(settings.PROJECT_OWNER) and (
            username.lower() == settings.PROJECT_OWNER.lower()
        )
        return Caller(username=username, is_owner=is_owner, source=source)

    # No identifying header found and not in dev mode: treat as a viewer. The
    # operator should hit /settings/whoami to pin the header name for their
    # deployment, then set MODEL_APP_USER_HEADER.
    return Caller(username="", is_owner=False, source="unknown")


def probe_headers(request: Request) -> dict:
    """Diagnostic snapshot for pinning the identity header on a deployment."""
    resolved = resolve_caller(request)
    return {
        "configured_owner": settings.PROJECT_OWNER,
        "dev_owner_mode": settings.DEV_TREAT_CALLER_AS_OWNER,
        "candidate_headers_checked": _CANDIDATE_USER_HEADERS,
        "resolved_username": resolved.username,
        "resolved_is_owner": resolved.is_owner,
        "resolved_from": resolved.source,
        "all_request_headers": {k: v for k, v in request.headers.items()},
    }
