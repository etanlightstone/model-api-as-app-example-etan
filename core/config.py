"""Active-model configuration (the single ``app_config`` row).

One row names the model the app currently hosts and where it came from. Empty
table = unconfigured ("not set up yet"). Only the owner may write (the route
enforces that; this module is the persistence layer).
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from typing import Any

from core import db


@dataclass
class ModelConfig:
    source_type: str            # 'registry' | 'custom_function'
    params: dict[str, Any]
    display_name: str
    slug: str
    updated_at: str
    updated_by: str


def get_config() -> ModelConfig | None:
    conn = db.get_conn()
    row = conn.execute("SELECT * FROM app_config WHERE id = 1").fetchone()
    if row is None:
        return None
    return ModelConfig(
        source_type=row["source_type"],
        params=json.loads(row["params_json"] or "{}"),
        display_name=row["display_name"],
        slug=row["slug"],
        updated_at=row["updated_at"],
        updated_by=row["updated_by"],
    )


def save_config(
    source_type: str,
    params: dict[str, Any],
    display_name: str,
    slug: str,
    updated_by: str,
) -> ModelConfig:
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO app_config (id, source_type, params_json, display_name, slug, updated_at, updated_by)
            VALUES (1, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_type=excluded.source_type,
                params_json=excluded.params_json,
                display_name=excluded.display_name,
                slug=excluded.slug,
                updated_at=excluded.updated_at,
                updated_by=excluded.updated_by
            """,
            (source_type, json.dumps(params), display_name, slug, now, updated_by),
        )
    return ModelConfig(source_type, params, display_name, slug, now, updated_by)


def clear_config() -> None:
    with db.transaction() as conn:
        conn.execute("DELETE FROM app_config WHERE id = 1")
