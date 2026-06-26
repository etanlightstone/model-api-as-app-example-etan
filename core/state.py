"""Live model state — the in-memory active ``ModelAdapter``.

Holds the adapter the app is currently serving, plus its warmup status so the
UI and routes can show a clean "not set up yet" / "failed to load" state rather
than crashing. Rebuilt from ``app_config`` on boot and whenever the owner saves
a new selection.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from core import config as config_mod
from core.adapter import CustomFunctionAdapter, ModelAdapter, RegistryAdapter


@dataclass
class LoadState:
    configured: bool = False
    ready: bool = False
    error: str = ""
    slug: str = ""
    display_name: str = ""
    source_type: str = ""


_lock = threading.Lock()
_adapter: ModelAdapter | None = None
_state = LoadState()


def build_adapter(cfg: config_mod.ModelConfig) -> ModelAdapter:
    """Construct (but don't warm) an adapter from a saved config."""
    p: dict[str, Any] = cfg.params or {}
    if cfg.source_type == "registry":
        return RegistryAdapter(
            model_name=p.get("model_name", ""),
            version=p.get("version"),
            stage=p.get("stage"),
            model_uri=p.get("model_uri"),
            name=cfg.display_name or p.get("model_name", "model"),
        )
    if cfg.source_type == "custom_function":
        return CustomFunctionAdapter(
            file_path=p["file_path"],
            func_name=p.get("func_name", "predict"),
            name=cfg.display_name or None,
            overrides=p.get("overrides"),
        )
    raise ValueError(f"Unknown source_type: {cfg.source_type}")


def reload_from_config() -> LoadState:
    """(Re)build + warm the active adapter from the persisted config."""
    global _adapter, _state
    with _lock:
        cfg = config_mod.get_config()
        if cfg is None:
            _adapter = None
            _state = LoadState(configured=False, ready=False)
            return _state

        st = LoadState(
            configured=True,
            slug=cfg.slug,
            display_name=cfg.display_name,
            source_type=cfg.source_type,
        )
        try:
            adapter = build_adapter(cfg)
            adapter.ensure_warm()
            # The adapter may refine its own slug/name during warmup (e.g. a
            # custom function's model_app.yaml pins them). That explicit override
            # is authoritative for routing, so adopt it — and persist it back so
            # the config and the live route agree.
            if adapter.slug and (adapter.slug != cfg.slug or adapter.name != cfg.display_name):
                config_mod.save_config(cfg.source_type, cfg.params,
                                       adapter.name or cfg.display_name,
                                       adapter.slug, cfg.updated_by)
            _adapter = adapter
            st.ready = True
            st.slug = adapter.slug or cfg.slug
            st.display_name = adapter.name or st.display_name
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI, not swallowed
            _adapter = None
            st.ready = False
            st.error = f"{type(exc).__name__}: {exc}"
        _state = st
        return _state


def get_adapter() -> ModelAdapter | None:
    return _adapter


def get_state() -> LoadState:
    return _state


def adapter_for_slug(slug: str) -> ModelAdapter | None:
    """Resolve an adapter by route slug (one model per app, so it's the active one)."""
    if _adapter is not None and _adapter.slug == slug:
        return _adapter
    return None
