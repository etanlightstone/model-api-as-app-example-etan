"""Self-documenting UI + playground (server-rendered, no build step).

Every page is a *projection of the live ``ModelAdapter`` + generated schema* the
endpoints use, so the docs can never drift from the API. The endpoints page
lists the sync + async routes with field tables and the §5.7 all-in-one curl;
the playground renders a schema-driven form that fetches the live endpoint
same-origin (the browser's Domino app-auth cookie carries auth).
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from core import identity, links, snippets, state
from core.schema import example_record, input_json_schema

router = APIRouter()
_TEMPLATES = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "templates"))


def _endpoint_descriptors(base: str, adapter) -> list[dict]:
    schema = adapter.input_schema
    slug = adapter.slug
    example = example_record(schema)
    return [
        {
            "id": "sync",
            "title": "Real-time prediction",
            "method": "POST",
            "url": links.sync_url(base, slug),
            "request_envelope": {"data": example},
            "response_example": {"result": "<model output>", "request_id": "…", "timing_ms": 12.3},
            "curl_workload": snippets.sync_curl(base, slug, schema, in_workload=True),
            "curl_offplatform": snippets.sync_curl(base, slug, schema, in_workload=False),
        },
        {
            "id": "async_submit",
            "title": "Async submit",
            "method": "POST",
            "url": links.async_base(base, slug),
            "request_envelope": {"parameters": example},
            "response_example": {"asyncPredictionId": "task_…"},
            "curl_workload": snippets.async_curl(base, slug, schema, in_workload=True),
            "curl_offplatform": snippets.async_curl(base, slug, schema, in_workload=False),
        },
        {
            "id": "async_poll",
            "title": "Async poll",
            "method": "GET",
            "url": links.async_base(base, slug) + "/{asyncPredictionId}",
            "request_envelope": None,
            "response_example": {"status": "succeeded", "result": "<model output>"},
            "curl_workload": None,
            "curl_offplatform": None,
        },
    ]


def _context(request: Request) -> dict:
    st = state.get_state()
    caller = identity.resolve_caller(request)
    # We can't know the external prefix server-side; emit a placeholder for any
    # absolute URL and let the browser fill it from document.baseURI.
    base = links.APP_BASE_PLACEHOLDER
    ctx = {
        "request": request,
        "state": st,
        "caller": caller,
        "base": base,
        "base_href": links.base_href(request),
        "app_title": "Model API (as App)",
    }
    adapter = state.get_adapter()
    if st.ready and adapter is not None:
        ctx["adapter"] = adapter
        ctx["schema"] = adapter.input_schema
        ctx["output_fields"] = adapter.input_schema.outputs
        ctx["json_schema"] = input_json_schema(adapter.input_schema)
        ctx["example_record"] = example_record(adapter.input_schema)
        ctx["endpoints"] = _endpoint_descriptors(base, adapter)
        ctx["has_image"] = adapter.input_schema.has_image_input()
        ctx["passthrough"] = adapter.input_schema.passthrough
    return ctx


@router.get("/")
async def home(request: Request):
    ctx = _context(request)
    name = "endpoints.html" if ctx["state"].ready else "not_set_up.html"
    return _TEMPLATES.TemplateResponse(request, name, ctx)


@router.get("/settings")
async def settings_page(request: Request):
    ctx = _context(request)
    return _TEMPLATES.TemplateResponse(request, "settings.html", ctx)
