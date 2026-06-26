"""Proxy-safe URL construction.

A Domino app is served behind a reverse proxy under a path prefix like
``/apps/<app-id>/``. The proxy **strips** that prefix before forwarding, so the
app only ever sees the internal path (``/``, ``/settings``, ``/models/...``) and
has **no reliable way to know the external prefix** — uvicorn's ``--proxy-headers``
restores scheme/host but not the path. We therefore never emit root-absolute or
absolute URLs from the server. Instead:

* Pages set ``<base href>`` to a *relative* path that points back at the app
  root (derived purely from the request's own depth). Every relative link/asset
  in the page then resolves against the real external URL the browser used.
* Anywhere we need the absolute external base (the copy-paste curl snippets), we
  emit the ``APP_BASE_PLACEHOLDER`` token and substitute it client-side from
  ``document.baseURI`` — which the browser knows even though the server doesn't.
"""

from __future__ import annotations

from starlette.requests import Request

# Token the server writes wherever the absolute external base is needed; the
# browser swaps it for document.baseURI (see static/app.js).
APP_BASE_PLACEHOLDER = "__APP_BASE__"


def base_href(request: Request) -> str:
    """A relative href that resolves to the app root from the current page.

    Computed from the internal path depth, so it's correct regardless of the
    (unknown) external prefix the proxy adds. For the app's pages — ``/`` and
    ``/settings`` — this is ``./``; deeper pages get the right number of ``../``.
    """
    path = request.scope.get("path") or request.url.path or "/"
    segments = [s for s in path.split("/") if s]
    ups = max(0, len(segments) - 1)
    return "./" if ups == 0 else "../" * ups


def sync_url(base: str, slug: str) -> str:
    return f"{base}/models/{slug}/latest/model"


def async_base(base: str, slug: str) -> str:
    return f"{base}/api/modelApis/async/v1/{slug}"
