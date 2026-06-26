"""Worker-process entrypoints (Plan A, async doc §3b).

These run **inside the ProcessPoolExecutor worker processes**, not on the event
loop. Each worker loads the model **once** via ``init_worker`` (into a module
global) and then runs pure compute in ``classify_chunk`` — no DB handle, no
dataset writes. The model is rebuilt from the same ``app_config`` the main
process uses, so the pool hosts whatever model the owner selected.
"""

from __future__ import annotations

import os

_ADAPTER = None


def init_worker(source_type: str, params: dict, display_name: str, slug: str,
                torch_threads: int = 1) -> None:
    """Runs once per pool process at startup. Loads the model into a global."""
    # Pin intra-op threads before importing torch-heavy code to avoid
    # K×T thread oversubscription (async doc §4b).
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))

    from core import config as config_mod
    from core import state

    cfg = config_mod.ModelConfig(
        source_type=source_type, params=params or {},
        display_name=display_name, slug=slug, updated_at="", updated_by="",
    )
    global _ADAPTER
    adapter = state.build_adapter(cfg)
    adapter.ensure_warm()
    _ADAPTER = adapter

    try:
        import torch

        torch.set_num_threads(torch_threads)
    except ImportError:
        pass


def classify_chunk(records: list[dict]) -> list[dict]:
    """Pure compute: validate + predict a chunk of records. Runs in a worker."""
    from core.predict_service import prepare_records

    if _ADAPTER is None:
        raise RuntimeError("Worker model not initialized")
    return _ADAPTER.predict(prepare_records(_ADAPTER, records))
