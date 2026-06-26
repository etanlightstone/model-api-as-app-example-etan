"""A fixture model with an un-introspectable signature.

``predict(**kwargs)`` has no typed parameters, so the harness can't infer a
schema and must fall back to passthrough mode (arbitrary JSON in/out, no
validation). Stands in for a registry model logged without an MLflow signature.
"""

from __future__ import annotations


def predict(**kwargs) -> dict:
    # Echo the input back plus a derived field, proving arbitrary keys flow
    # through untouched.
    return {"received": dict(kwargs), "num_fields": len(kwargs)}
