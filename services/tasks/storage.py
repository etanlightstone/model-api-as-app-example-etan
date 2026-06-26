"""Dataset-filesystem blob storage for async tasks.

Follows the patterns in ``async_arch_for_cpu_bound_background_tasks.md`` §6:
JSONL blobs on the Domino dataset, appended + fsynced chunk-by-chunk, with a
byte-offset cursor reader so a restart resumes from the last durable chunk.
Large inputs/outputs stream to/from disk and are never held whole in memory.

Layout::

    <DATA_DIR>/tasks/<task_id>/
        input.jsonl     # by-reference inputs (immutable after enqueue)
        output.jsonl    # appended chunk-by-chunk, fsynced
        meta.json       # debug snapshot
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from core import settings


def task_dir(task_id: str) -> Path:
    d = settings.TASKS_DIR / task_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def input_path(task_id: str) -> str:
    return str(task_dir(task_id) / "input.jsonl")


def output_path(task_id: str) -> str:
    return str(task_dir(task_id) / "output.jsonl")


def write_meta(task_id: str, meta: dict) -> None:
    with open(task_dir(task_id) / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)


def append_jsonl_line(path: str, obj) -> None:
    """Append one JSON object as a line, fsynced before returning.

    fsync-before-checkpoint is the durability contract: results hit disk before
    the worker advances its cursor, so a crash re-runs at most the last chunk.
    """
    line = json.dumps(obj, default=str)
    with open(path, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_records(path: str, records: list[dict]) -> int:
    """Write a list of records to a fresh JSONL file. Returns byte length."""
    total = 0
    with open(path, "w") as fh:
        for r in records:
            line = json.dumps(r, default=str) + "\n"
            fh.write(line)
            total += len(line.encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    return total


def iter_jsonl(path: str, start_byte: int = 0) -> Iterator[tuple[int, dict]]:
    """Yield ``(end_byte_offset, record)`` from ``start_byte``.

    The end offset is the byte position *after* the yielded line, suitable for
    storing as a resume cursor. Assumes an append-stable, immutable file.
    """
    if not os.path.isfile(path):
        return
    with open(path, "rb") as fh:
        fh.seek(start_byte)
        offset = start_byte
        for raw in fh:
            offset += len(raw)
            text = raw.decode("utf-8").strip()
            if not text:
                continue
            yield offset, json.loads(text)


def count_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def read_all(path: str) -> list[dict]:
    return [rec for _, rec in iter_jsonl(path)]


def resolve_reference_path(ref: str) -> str:
    """Resolve a by-reference ``input_file`` to a readable local path.

    Accepts an absolute path, or one relative to the project's dataset dir. S3
    and other remote stores are out of v1 scope (plan §5.4 risk #5); we default
    to dataset paths, which are already mounted.
    """
    if os.path.isabs(ref) and os.path.exists(ref):
        return ref
    datasets_dir = os.environ.get("DOMINO_DATASETS_DIR", "")
    if datasets_dir:
        cand = os.path.join(datasets_dir, ref)
        if os.path.exists(cand):
            return cand
        cand2 = os.path.join(datasets_dir, settings.PROJECT_NAME, ref)
        if os.path.exists(cand2):
            return cand2
    if os.path.exists(ref):
        return ref
    raise FileNotFoundError(f"input_file not found (dataset paths only): {ref}")


def load_reference_records(ref: str) -> list[dict]:
    """Load records from a by-reference input file (.jsonl or .csv)."""
    path = resolve_reference_path(ref)
    if path.endswith(".csv"):
        import pandas as pd

        return pd.read_csv(path).to_dict(orient="records")
    return read_all(path)
