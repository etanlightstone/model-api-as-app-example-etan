"""Model-API-as-App — a Domino App that hosts any project model as a REST API.

The app serves a selected model behind Domino's native app auth with a URL and
payload shape that mirror Domino Model APIs (sync ``/models/{slug}/latest/model``
and async ``/api/modelApis/async/v1/{slug}``), plus a self-documenting browser
UI and an always-on async submit/poll surface.

This module wires the pieces together and owns the lifespan: initialize the DB,
load the configured model (if any), and start the async worker + process pool.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core import db, settings, state
from routes import async_api, health, predict, settings as settings_routes, ui
from services.tasks import worker as task_worker

logging.basicConfig(level=os.environ.get("MODEL_APP_LOG_LEVEL", "INFO"))
logger = logging.getLogger("model_app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    db.init_db()

    ls = state.reload_from_config()
    if ls.configured and ls.ready:
        logger.info("Loaded model '%s' (slug=%s)", ls.display_name, ls.slug)
    elif ls.configured:
        logger.warning("Configured model failed to load: %s", ls.error)
    else:
        logger.info("No model configured yet — app is in 'not set up' state.")

    task_worker.configure()
    await task_worker.start_worker()
    try:
        yield
    finally:
        await task_worker.stop_worker()
        task_worker.shutdown_pool()


app = FastAPI(title=settings.APP_TITLE, lifespan=lifespan)

_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(async_api.router)
app.include_router(settings_routes.router)
app.include_router(ui.router)
