"""Shared request normalization + validation for the prediction routes.

Mirrors how a Domino Model API treats the request body: values arrive (often as
strings) inside an envelope, are validated/coerced against the model's schema,
and one output object is produced per input record. Used by both the sync route
and the async worker so they apply identical rules.
"""

from __future__ import annotations

from typing import Any

from core.adapter import ModelAdapter
from core.schema import build_request_model


class ValidationError(Exception):
    """Raised when a request body doesn't match the model's input schema."""


def normalize_records(payload: Any, passthrough: bool = False) -> tuple[list[dict], bool]:
    """Turn a request payload into a list of records.

    Returns ``(records, was_single)``. Accepts:
      * a single record dict (scalars)            → 1 record
      * a columnar dict (``{"col": [v1, v2]}``)   → N records
      * a list of record dicts                    → N records

    In ``passthrough`` mode we skip columnar expansion: without a known schema we
    can't tell a multi-row columnar payload from a model that genuinely wants a
    list-valued field, so a dict is always treated as a single record.
    """
    if isinstance(payload, list):
        return [dict(r) for r in payload], False
    if isinstance(payload, dict):
        if not passthrough:
            list_vals = {k: v for k, v in payload.items() if isinstance(v, list)}
            if list_vals and len(list_vals) == len(payload):
                n = len(next(iter(list_vals.values())))
                if any(len(v) != n for v in list_vals.values()):
                    raise ValidationError("Columnar inputs have mismatched lengths.")
                return [{k: payload[k][i] for k in payload} for i in range(n)], False
        return [payload], True
    raise ValidationError("Request body must be an object or a list of objects.")


def validate_records(adapter: ModelAdapter, records: list[dict]) -> list[dict]:
    """Validate + coerce each record against the adapter's input schema."""
    model = build_request_model(adapter.input_schema)
    validated: list[dict] = []
    for i, rec in enumerate(records):
        try:
            obj = model(**rec)
        except Exception as exc:  # pydantic ValidationError et al.
            raise ValidationError(f"Record {i}: {exc}") from exc
        validated.append(obj.model_dump())
    return validated


def prepare_records(adapter: ModelAdapter, records: list[dict]) -> list[dict]:
    """Validate + coerce records, or pass them through untouched in passthrough mode."""
    if adapter.input_schema.passthrough:
        return records
    return validate_records(adapter, records)


def run_prediction(adapter: ModelAdapter, payload: Any) -> tuple[Any, int]:
    """Validate, predict, and shape the result like a Domino Model API.

    Returns ``(result, n_records)`` where ``result`` is a single dict for a
    single-record request, else a list of dicts.
    """
    passthrough = adapter.input_schema.passthrough
    records, was_single = normalize_records(payload, passthrough=passthrough)
    if not records:
        raise ValidationError("No input records provided.")
    validated = prepare_records(adapter, records)
    outputs = adapter.predict(validated)
    if was_single and len(outputs) == 1:
        return outputs[0], 1
    return outputs, len(outputs)
