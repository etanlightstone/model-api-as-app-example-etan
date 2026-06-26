"""Proxy-aware URL construction.

A Domino app is served behind a reverse proxy under a path prefix like
``/<owner>/<project>/r/.../<app>/``. uvicorn's ``--proxy-headers`` plus the
forwarded headers let Starlette reconstruct the externally reachable scheme,
host, and prefix. We centralize that here so the self-doc UI and the generated
curl snippets always show the real external URL, never ``localhost:8888``.
"""

from __future__ import annotations

from starlette.requests import Request


def app_base_url(request: Request) -> str:
    """The external base URL of the app, including any proxy path prefix, no trailing slash."""
    base = str(request.base_url).rstrip("/")
    # base_url already includes root_path when Starlette knows it. Honor an
    # explicit forwarded prefix if the proxy sends one and it's not reflected.
    prefix = request.headers.get("x-forwarded-prefix", "").rstrip("/")
    if prefix and not base.endswith(prefix):
        base = base + prefix
    return base


def sync_url(base: str, slug: str) -> str:
    return f"{base}/models/{slug}/latest/model"


def async_base(base: str, slug: str) -> str:
    return f"{base}/api/modelApis/async/v1/{slug}"
