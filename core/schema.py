"""Schema model + projections.

A ``Schema`` is the single source of truth for a hosted model's input/output
shape. Adapters produce it (from an MLflow signature or a function's typed
parameters); everything downstream is a *projection* of it:

* a **pydantic model** for request validation (with Domino-style string
  coercion),
* a **JSON-Schema** blob for the UI/playground,
* an **example payload** for docs and the "try it" form.

Keeping all three derived from one ``Schema`` is what lets the self-documenting
UI never drift from the live endpoint.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

from pydantic import create_model

# JSON-type vocabulary we normalize every adapter's types into.
JSON_TYPES = ("number", "integer", "string", "boolean", "object", "array")

# Map a python/numpy/mlflow type name to one of JSON_TYPES.
_TYPE_NAME_TO_JSON = {
    "float": "number", "float32": "number", "float64": "number", "double": "number",
    "int": "integer", "int32": "integer", "int64": "integer", "long": "integer",
    "str": "string", "string": "string", "object": "string",
    "bool": "boolean", "boolean": "boolean",
    "bytes": "string", "binary": "string",
    "list": "array", "dict": "object",
}

# Example values per JSON type, used when an adapter can't supply a better one.
_EXAMPLE_BY_TYPE = {
    "number": 0.0, "integer": 0, "string": "example",
    "boolean": False, "object": {}, "array": [],
}


def python_type_to_json(tp: Any) -> str:
    """Best-effort map a python type / annotation to a JSON-type string.

    Unknown or unannotated types fall back to ``string`` — matching Domino's
    verbatim string forwarding, and flagged as "type unverified" by the caller.
    """
    if tp is None:
        return "string"
    name = getattr(tp, "__name__", None) or str(tp)
    name = name.lower().strip()
    return _TYPE_NAME_TO_JSON.get(name, "string")


@dataclass
class Field:
    """One input or output field."""

    name: str
    type: str = "string"            # one of JSON_TYPES
    required: bool = True
    example: Any = None
    description: str = ""
    # True when the type was inferred without a hint (Domino-style string
    # default). The UI surfaces this as "type unverified".
    type_unverified: bool = False
    # True when the field carries a base64-encoded image (§5.5).
    image: bool = False

    def with_default_example(self) -> "Field":
        if self.example is None and not self.image:
            self.example = _EXAMPLE_BY_TYPE.get(self.type, "example")
        if self.example is None and self.image:
            self.example = ""  # base64 string, filled in via the file picker
        return self


@dataclass
class Schema:
    """Ordered input + (best-effort) output field lists."""

    inputs: list[Field] = field(default_factory=list)
    outputs: list[Field] = field(default_factory=list)
    notes: str = ""

    def input_names(self) -> list[str]:
        return [f.name for f in self.inputs]

    def has_image_input(self) -> bool:
        return any(f.image for f in self.inputs)


# --- Projections -------------------------------------------------------------

_PY_TYPE = {
    "number": float, "integer": int, "string": str,
    "boolean": bool, "object": dict, "array": list,
}


def build_request_model(schema: Schema, model_name: str = "RequestRecord"):
    """A pydantic model validating one input record.

    Field types come from the schema; optional fields get ``None`` defaults.
    pydantic v2's lax coercion turns the strings Domino forwards (``"3000.0"``)
    into the declared numeric/bool types, which is exactly the Domino behaviour
    we want to mirror.
    """
    fields: dict[str, Any] = {}
    for f in schema.inputs:
        py = _PY_TYPE.get(f.type, str)
        if f.required:
            fields[f.name] = (py, ...)
        else:
            fields[f.name] = (py | None, None)  # type: ignore[operator]
    # `extra='ignore'` keeps us lenient like Domino; unknown keys are dropped.
    return create_model(model_name, **fields)  # type: ignore[call-overload]


def input_json_schema(schema: Schema) -> dict:
    """A JSON-Schema-ish dict for the UI (not a strict draft, just enough)."""
    props = {}
    required = []
    for f in schema.inputs:
        prop: dict[str, Any] = {"type": "string" if f.image else f.type}
        if f.description:
            prop["description"] = f.description
        if f.example is not None:
            prop["example"] = f.example
        if f.image:
            prop["format"] = "image-base64"
        props[f.name] = prop
        if f.required:
            required.append(f.name)
    return {"type": "object", "properties": props, "required": required}


def example_record(schema: Schema) -> dict:
    """An example input record (one value per field) for docs + playground."""
    out = {}
    for f in schema.inputs:
        ex = f.example
        if ex is None:
            ex = "" if f.image else _EXAMPLE_BY_TYPE.get(f.type, "example")
        # Domino forwards values as strings; show numbers as strings in examples
        # so the copy/paste payload matches what a real Model API receives.
        if f.type in ("number", "integer") and not f.image:
            ex = str(ex)
        out[f.name] = ex
    return out


def example_output(schema: Schema) -> dict:
    out = {}
    for f in schema.outputs:
        ex = f.example
        if ex is None:
            ex = _EXAMPLE_BY_TYPE.get(f.type, None)
        out[f.name] = ex
    return out


def coerce_value(value: Any, json_type: str) -> Any:
    """Coerce a single (possibly string) value to a python type.

    Used by adapters that call a typed function with kwargs, so a Domino-style
    string ``"7"`` reaches an ``int`` parameter as ``7``.
    """
    if value is None:
        return None
    try:
        if json_type == "number":
            return float(value)
        if json_type == "integer":
            return int(float(value))
        if json_type == "boolean":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "y", "t")
            return bool(value)
        if json_type == "string":
            return str(value)
    except (TypeError, ValueError):
        return value
    return value


def jsonable(obj: Any) -> Any:
    """Make adapter outputs JSON-serializable (numpy/pandas/datetime scalars)."""
    # Lazily handle numpy without importing it at module load.
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return [jsonable(x) for x in obj.tolist()]
    except ImportError:
        pass
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    return obj
