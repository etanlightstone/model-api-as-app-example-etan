"""Owner-only Settings — pick which model the app hosts.

Before configuration everyone sees "not set up yet"; the owner additionally
sees the model picker (registry models + versions, or a custom function path).
Saving writes ``app_config``, warms the adapter, and rebuilds the async pool so
the endpoints light up live.

Writes are JSON ``fetch`` calls from the page (no multipart), each re-checking
owner identity server-side — the client-side hiding of controls is cosmetic.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from core import config as config_mod
from core import identity, registry, state
from core.adapter import slugify
from services.tasks import worker as task_worker

router = APIRouter(prefix="/settings")


def _require_owner(request: Request) -> identity.Caller:
    caller = identity.resolve_caller(request)
    if not caller.is_owner:
        raise HTTPException(403, "Only the project owner can change the hosted model.")
    return caller


@router.get("/whoami")
async def whoami(request: Request):
    """Diagnostic: pin the identity header name on this deployment (plan §5.6)."""
    return identity.probe_headers(request)


@router.get("/models")
async def list_registry_models(request: Request):
    _require_owner(request)
    listing = registry.list_models()
    return {
        "available": listing.available,
        "error": listing.error,
        "models": [
            {"name": m.name, "versions": m.versions,
             "latest_version": m.latest_version, "description": m.description}
            for m in listing.models
        ],
    }


def _apply_and_reload():
    """Reload the active adapter and resize the async pool to match."""
    ls = state.reload_from_config()
    task_worker.configure()
    return ls


@router.post("/select")
async def select_model(request: Request):
    caller = _require_owner(request)
    body = await request.json()
    source_type = body.get("source_type")

    if source_type == "registry":
        model_name = (body.get("model_name") or "").strip()
        if not model_name:
            raise HTTPException(422, "model_name is required for a registry model.")
        version = (body.get("version") or "").strip() or None
        display = (body.get("display_name") or model_name).strip()
        slug = slugify(body.get("slug") or display)
        params = {"model_name": model_name, "version": version}

    elif source_type == "custom_function":
        import os

        file_path = (body.get("file_path") or "").strip()
        if not file_path:
            raise HTTPException(422, "file_path is required for a custom function.")
        func_name = (body.get("func_name") or "predict").strip()
        display = (body.get("display_name") or "").strip()
        # Default the slug to the *folder* name (e.g. weather-regressor), not the
        # full file path — a model_app.yaml `slug`/`name` still overrides on warmup.
        folder = os.path.basename(os.path.dirname(os.path.abspath(file_path))) or "model"
        slug = slugify(body.get("slug") or display or folder)
        params = {"file_path": file_path, "func_name": func_name,
                  "overrides": body.get("overrides") or {}}
        display = display or folder
    else:
        raise HTTPException(422, "source_type must be 'registry' or 'custom_function'.")

    config_mod.save_config(
        source_type=source_type, params=params, display_name=display,
        slug=slug, updated_by=caller.username,
    )
    ls = _apply_and_reload()
    if not ls.ready:
        # Roll back so we don't leave the app pointing at a model that won't load.
        config_mod.clear_config()
        _apply_and_reload()
        raise HTTPException(400, f"Model selected but failed to load: {ls.error}")
    return {"ok": True, "slug": ls.slug, "display_name": ls.display_name,
            "ready": ls.ready}


@router.post("/clear")
async def clear_model(request: Request):
    _require_owner(request)
    config_mod.clear_config()
    _apply_and_reload()
    return {"ok": True}
