"""Process-wide configuration resolved from the environment.

Everything here is read once at import. The values that matter for *where* the
app keeps its durable state (the SQLite DB + async blobs) and *how* it reaches
Domino (the token proxy, the registry API) are templated from env vars rather
than hardcoded, because they vary by deployment — exactly the Phase 1 caveat in
the implementation plan.
"""

from __future__ import annotations

import os
from pathlib import Path


def _first_env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


# --- Identity ---------------------------------------------------------------
# The project owner is the only identity allowed to change Settings. Domino
# exposes it via DOMINO_PROJECT_OWNER (and the starting user as a fallback).
PROJECT_OWNER: str = _first_env("DOMINO_PROJECT_OWNER", "DOMINO_STARTING_USERNAME")
PROJECT_NAME: str = _first_env("DOMINO_PROJECT_NAME", default="model-api-as-app")
PROJECT_ID: str = os.environ.get("DOMINO_PROJECT_ID", "")

# When true, identity gating is relaxed so a developer running the app outside
# Domino (no run-as headers) is treated as the owner. Never set this in a real
# Domino deployment — there the headers/env are present and gating is real.
DEV_TREAT_CALLER_AS_OWNER: bool = os.environ.get(
    "MODEL_APP_DEV_OWNER", "" if PROJECT_OWNER else "1"
).lower() in ("1", "true", "yes")


# --- Domino service endpoints ------------------------------------------------
# The in-workload token proxy. The implementation plan calls for templating the
# port/path from an env var rather than hardcoding `localhost:8899`; Domino sets
# DOMINO_API_PROXY to exactly that base.
TOKEN_PROXY_BASE: str = _first_env(
    "MODEL_APP_TOKEN_PROXY", "DOMINO_API_PROXY", default="http://localhost:8899"
).rstrip("/")
TOKEN_PROXY_PATH: str = os.environ.get("MODEL_APP_TOKEN_PROXY_PATH", "/access-token")
TOKEN_PROXY_URL: str = f"{TOKEN_PROXY_BASE}{TOKEN_PROXY_PATH}"

# Domino REST API (used by the registry client). DOMINO_API_HOST points at the
# in-cluster Nucleus service; DOMINO_USER_API_KEY authenticates server-side use.
DOMINO_API_HOST: str = _first_env("DOMINO_API_HOST", "DOMINO_USER_HOST").rstrip("/")
DOMINO_USER_API_KEY: str = os.environ.get("DOMINO_USER_API_KEY", "")

# The externally-reachable Domino host, used to build *browser* links (e.g. the
# "view in model registry" link on the Endpoints page). Prefer DOMINO_USER_HOST
# — the in-cluster DOMINO_API_HOST may not be reachable from a user's browser —
# and fall back to it only when the user host isn't set.
DOMINO_USER_HOST: str = _first_env("DOMINO_USER_HOST", "DOMINO_API_HOST").rstrip("/")


# --- Durable state location --------------------------------------------------
def _resolve_data_dir() -> Path:
    """Where the SQLite DB + async blobs live.

    Preference order:
      1. MODEL_APP_DATA_DIR (explicit override).
      2. A `.model_app` dir inside the project's mounted Domino dataset, so
         state survives app redeploys (the whole point of putting it on a
         dataset rather than the ephemeral container fs).
      3. A local `.appdata` dir (dev fallback when no dataset is mounted).
    """
    override = os.environ.get("MODEL_APP_DATA_DIR")
    if override:
        return Path(override)

    datasets_dir = os.environ.get("DOMINO_DATASETS_DIR")
    if datasets_dir:
        base = Path(datasets_dir) / PROJECT_NAME
        # The project dataset is typically mounted at <datasets>/<project name>.
        if base.is_dir():
            return base / ".model_app"
        # Some deployments mount the dataset root directly.
        if Path(datasets_dir).is_dir():
            return Path(datasets_dir) / ".model_app"

    return Path(__file__).resolve().parent.parent / ".appdata"


DATA_DIR: Path = _resolve_data_dir()
DB_PATH: Path = DATA_DIR / "model_app.db"
TASKS_DIR: Path = DATA_DIR / "tasks"


def ensure_dirs() -> None:
    """Create the data + tasks directories (idempotent)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)


# --- App metadata ------------------------------------------------------------
APP_TITLE = "Model API (as App)"
APP_PORT = int(os.environ.get("APP_PORT", "8888"))
