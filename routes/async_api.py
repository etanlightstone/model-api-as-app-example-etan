"""Asynchronous submit/poll endpoints — mirror Domino's async Model API exactly.

    POST {APP_BASE}/api/modelApis/async/v1/{slug}
      body: {"parameters": {...}}                  (by value)
         or {"parameters": {"input_file": "<path>"}}  (by reference, §5.4)
      ->   {"asyncPredictionId": "<id>"}

    GET  {APP_BASE}/api/modelApis/async/v1/{slug}/{predId}
      ->   {"status": "queued"}
         | {"status": "succeeded", "result": {...}}
         | {"status": "failed", "errors": [...]}
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from core import identity, state
from core.predict_service import ValidationError, normalize_records, validate_records
from services.tasks import service
from services.tasks import worker as task_worker

router = APIRouter(prefix="/api/modelApis/async/v1")


def _require_ready(slug: str):
    st = state.get_state()
    if not st.configured:
        raise HTTPException(503, "This model endpoint is not set up yet.")
    adapter = state.adapter_for_slug(slug)
    if adapter is None:
        if st.error:
            raise HTTPException(503, f"Model failed to load: {st.error}")
        raise HTTPException(404, f"No model is hosted at slug '{slug}'.")
    if not task_worker.is_enabled():
        raise HTTPException(503, "Async tasks are not enabled on this deployment.")
    return adapter


@router.post("/{slug}")
async def submit(slug: str, request: Request):
    adapter = _require_ready(slug)
    await task_worker.ensure_running()
    caller = identity.resolve_caller(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Request body must be valid JSON.")
    params = body.get("parameters") if isinstance(body, dict) and "parameters" in body else body
    if not isinstance(params, (dict, list)):
        raise HTTPException(422, "Expected a 'parameters' object.")

    # By-reference: a single {"input_file": "<path>"} pointer.
    if isinstance(params, dict) and "input_file" in params and len(params) == 1:
        try:
            task_id = service.create_task(
                slug=slug, user_id=caller.username, user_name=caller.username,
                mode="reference", input_file_ref=params["input_file"],
            )
        except service.UserQuotaExceeded as exc:
            raise HTTPException(429, str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(422, str(exc))
        task_worker.nudge(task_id)
        return {"asyncPredictionId": task_id}

    # By-value: validate up front so bad input fails fast (like the sync route).
    # In passthrough mode there's no schema to validate against — forward as-is.
    passthrough = adapter.input_schema.passthrough
    try:
        records, _ = normalize_records(params, passthrough=passthrough)
        validated = records if passthrough else validate_records(adapter, records)
    except ValidationError as exc:
        raise HTTPException(422, str(exc))
    try:
        task_id = service.create_task(
            slug=slug, user_id=caller.username, user_name=caller.username,
            mode="value", records=validated,
        )
    except service.UserQuotaExceeded as exc:
        raise HTTPException(429, str(exc))
    task_worker.nudge(task_id)
    return {"asyncPredictionId": task_id}


@router.get("/{slug}/{pred_id}")
async def poll(slug: str, pred_id: str):
    row = service.get_task(pred_id)
    if row is None or row["slug"] != slug:
        raise HTTPException(404, "Prediction not found.")
    return service.to_public(row)


@router.get("/{slug}/{pred_id}/result")
async def result_stream(slug: str, pred_id: str):
    """Stream the output JSONL for a terminal by-reference task (convenience)."""
    row = service.get_task(pred_id)
    if row is None or row["slug"] != slug:
        raise HTTPException(404, "Prediction not found.")
    if row["status"] not in service.TERMINAL:
        raise HTTPException(409, f"Task not finished (status={row['status']}).")

    def _gen():
        import os

        path = row["output_file_path"]
        if path and os.path.isfile(path):
            with open(path, "rb") as fh:
                while chunk := fh.read(65536):
                    yield chunk

    return StreamingResponse(_gen(), media_type="application/x-ndjson")


@router.post("/{slug}/{pred_id}/cancel")
async def cancel(slug: str, pred_id: str):
    row = service.get_task(pred_id)
    if row is None or row["slug"] != slug:
        raise HTTPException(404, "Prediction not found.")
    task_worker.nudge(pred_id)
    return service.to_public(service.request_cancel(pred_id))
