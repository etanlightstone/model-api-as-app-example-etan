"""Synchronous (real-time) prediction endpoint.

Mirrors the Domino Model API sync contract exactly, only the URL prefix differs:

    POST {APP_BASE}/models/{slug}/latest/model
    body:  {"data": {...}}          (a bare record is also accepted)
    resp:  {"result": ..., "request_id": ..., "timing_ms": ...}

Inference runs in a worker thread (``asyncio.to_thread``) so a slow model never
blocks the event loop and other live requests stay responsive. Single-record
real-time calls are sub-second; the heavy, parallel path is the async engine
(Phase 6), which uses a process pool.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException, Request

from core import state
from core.predict_service import ValidationError, run_prediction

router = APIRouter()


def _require_ready(slug: str):
    st = state.get_state()
    if not st.configured:
        raise HTTPException(503, "This model endpoint is not set up yet.")
    adapter = state.adapter_for_slug(slug)
    if adapter is None:
        if not st.ready and st.error:
            raise HTTPException(503, f"Model failed to load: {st.error}")
        raise HTTPException(404, f"No model is hosted at slug '{slug}'.")
    return adapter


@router.post("/models/{slug}/latest/model")
async def predict(slug: str, request: Request):
    adapter = _require_ready(slug)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Request body must be valid JSON.")

    # Accept the Domino `{"data": {...}}` envelope or a bare record/list.
    payload = body.get("data") if isinstance(body, dict) and "data" in body else body

    request_id = uuid.uuid4().hex
    start = time.perf_counter()
    try:
        result, _ = await asyncio.to_thread(run_prediction, adapter, payload)
    except ValidationError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Inference error: {type(exc).__name__}: {exc}")
    timing_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "result": result,
        "request_id": request_id,
        "timing_ms": timing_ms,
        "model": {"slug": slug, "name": adapter.name},
    }
