"""Health + readiness."""

from __future__ import annotations

from fastapi import APIRouter, Request

from core import identity, state
from services.tasks import worker as task_worker

router = APIRouter()


@router.get("/health")
async def health(request: Request, verbose: bool = False):
    st = state.get_state()
    body = {
        "status": "ok",
        "configured": st.configured,
        "model_ready": st.ready,
    }
    if st.error:
        body["model_error"] = st.error
    if verbose:
        body["slug"] = st.slug
        body["display_name"] = st.display_name
        body["source_type"] = st.source_type
        body["identity"] = identity.resolve_caller(request).__dict__
        body["tasks_worker"] = task_worker.worker_state()
    return body
