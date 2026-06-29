"""``ModelAdapter`` — the single abstraction the whole app is built around.

The UI, the sync route, and the async worker all talk to one ``ModelAdapter``
so they share a single notion of "inputs, outputs, predict()". Adding support
for a new kind of model means adding an adapter, nothing else.

Two concrete adapters ship:

* ``CustomFunctionAdapter`` — point at ``file.py`` + a function name (the
  ``model_api.py`` → ``predict`` pattern). Schema is inferred from the typed
  signature; an optional ``model_app.yaml`` sidecar overrides it.
* ``RegistryAdapter`` — load a registered MLflow model; schema comes straight
  from the model's signature.

Adapters are framework-agnostic: a ``CustomFunctionAdapter`` happily wraps the
PyTorch diabetes model or the scikit-learn weather model because both expose a
typed ``predict(...)`` function.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import re
import sys
from abc import ABC, abstractmethod
from typing import Any

from core import schema as schema_mod
from core.schema import Field, Schema

# Sibling module names the example projects use; purged from the import cache
# before loading a custom function so the *right* siblings resolve even if a
# different model was loaded earlier in the process (test isolation; harmless in
# production where one app hosts one model).
_SIBLING_NAMES = ("model", "pyfunc_model")


def _purge_sibling_modules() -> None:
    """Drop cached generic-named modules so the next load re-imports its own.

    The example models bundle modules with identical names (``model.py``,
    ``pyfunc_model.py``). MLflow *prepends* a registry model's
    bundled ``code/`` dir to ``sys.path`` but never clears ``sys.modules`` — so a
    sibling cached by an earlier load shadows the one being loaded, and
    unpickling the pyfunc fails with e.g. ``Can't get attribute
    'WeatherRegressor' on module 'pyfunc_model'``. Popping the names forces the
    import machinery to re-resolve them from MLflow's freshly-prepended dir.
    """
    for name in _SIBLING_NAMES:
        sys.modules.pop(name, None)


def slugify(name: str) -> str:
    """A URL-safe slug for the route prefix, e.g. 'Weather Regressor' → 'weather-regressor'."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", str(name)).strip("-").lower()
    return s or "model"


class ModelAdapter(ABC):
    """Loads a model once and scores records against it."""

    def __init__(self) -> None:
        self.name = "model"
        self.slug = "model"
        self.schema = Schema()
        self.description = ""
        self._warm = False

    @property
    def input_schema(self) -> Schema:
        return self.schema

    @property
    def output_schema(self) -> Schema:
        return Schema(inputs=self.schema.outputs)

    @abstractmethod
    def warmup(self) -> None:
        """Load the model + resolve the schema. Idempotent; called once."""

    @abstractmethod
    def predict(self, records: list[dict]) -> list[dict]:
        """Score a list of input records, returning one output dict per record."""

    def ensure_warm(self) -> None:
        if not self._warm:
            self.warmup()
            self._warm = True


# --- Custom function adapter -------------------------------------------------

def _import_module_from_path(file_path: str):
    """Import a module from an absolute file path, with its dir on sys.path.

    The example ``model_api.py`` files do ``from model import ...``, so the
    module's own directory must be importable *and* take priority over any
    earlier-loaded namesakes.
    """
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Custom function file not found: {file_path}")
    mod_dir = os.path.dirname(file_path)
    mod_name = os.path.splitext(os.path.basename(file_path))[0]

    # Prioritize this model's directory for sibling imports.
    if mod_dir in sys.path:
        sys.path.remove(mod_dir)
    sys.path.insert(0, mod_dir)

    # Purge cached siblings (and the target) that resolve to a *different* dir.
    for cached in (*_SIBLING_NAMES, mod_name):
        existing = sys.modules.get(cached)
        if existing is not None:
            ex_file = getattr(existing, "__file__", "") or ""
            if os.path.dirname(os.path.abspath(ex_file)) != mod_dir:
                sys.modules.pop(cached, None)

    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _load_overrides(file_path: str, explicit: dict | None) -> dict:
    """Merge a ``model_app.yaml`` sidecar (next to the function) with explicit
    overrides passed in config. Explicit config wins."""
    merged: dict = {}
    sidecar = os.path.join(os.path.dirname(os.path.abspath(file_path)), "model_app.yaml")
    if os.path.isfile(sidecar):
        import yaml

        with open(sidecar) as fh:
            loaded = yaml.safe_load(fh) or {}
        if isinstance(loaded, dict):
            merged.update(loaded)
    if explicit:
        merged.update(explicit)
    return merged


class CustomFunctionAdapter(ModelAdapter):
    """Wrap a plain Python scoring function (e.g. ``model_api.py:predict``)."""

    def __init__(
        self,
        file_path: str,
        func_name: str = "predict",
        name: str | None = None,
        overrides: dict | None = None,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.func_name = func_name
        self._explicit_overrides = overrides or {}
        self._func = None
        self.name = name or slugify(os.path.basename(os.path.dirname(os.path.abspath(file_path))) or "model")
        self.slug = slugify(self.name)

    def warmup(self) -> None:
        overrides = _load_overrides(self.file_path, self._explicit_overrides)
        module = _import_module_from_path(self.file_path)
        func = getattr(module, self.func_name, None)
        if func is None or not callable(func):
            raise AttributeError(
                f"'{self.func_name}' is not a callable in {self.file_path}"
            )
        self._func = func

        if overrides.get("name"):
            self.name = str(overrides["name"])
            self.slug = slugify(overrides.get("slug") or self.name)
        self.description = overrides.get("description", "") or (func.__doc__ or "").strip().split("\n")[0]

        self.schema = self._infer_schema(func, overrides)

    def _infer_schema(self, func, overrides: dict) -> Schema:
        sig = inspect.signature(func)
        inputs: list[Field] = []
        image_fields = set(overrides.get("image_fields", []) or [])
        field_overrides = overrides.get("inputs", {}) or {}

        for pname, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            annot = param.annotation if param.annotation is not inspect._empty else None
            jtype = schema_mod.python_type_to_json(annot)
            unverified = annot is None
            required = param.default is inspect._empty
            example = None if required else param.default

            ov = field_overrides.get(pname, {}) or {}
            if "type" in ov:
                jtype = ov["type"]
                unverified = False
            if "example" in ov:
                example = ov["example"]
            is_image = pname in image_fields or bool(ov.get("image"))

            fld = Field(
                name=pname,
                type="string" if is_image else jtype,
                required=required,
                example=example,
                description=ov.get("description", ""),
                type_unverified=unverified and not is_image,
                image=is_image,
            ).with_default_example()
            inputs.append(fld)

        schema = Schema(inputs=inputs, notes=overrides.get("notes", ""))
        if not inputs:
            # No typed parameters to introspect (e.g. predict(**kwargs)) — accept
            # arbitrary JSON and forward it unvalidated.
            schema.passthrough = True
            schema.notes = (schema.notes or
                            "Schema could not be inferred from the signature; this "
                            "endpoint accepts arbitrary JSON (validation skipped).")
            return schema

        # Output schema: prefer an explicit override; else probe with one example
        # call so the docs show the real output field names/types.
        out_override = overrides.get("outputs")
        if out_override:
            schema.outputs = self._fields_from_override(out_override)
        else:
            schema.outputs = self._probe_outputs(func, schema)
        return schema

    @staticmethod
    def _fields_from_override(spec) -> list[Field]:
        fields = []
        if isinstance(spec, dict):
            for k, v in spec.items():
                v = v or {}
                fields.append(Field(name=k, type=v.get("type", "string"),
                                    example=v.get("example"),
                                    description=v.get("description", "")).with_default_example())
        elif isinstance(spec, list):
            for k in spec:
                fields.append(Field(name=str(k)).with_default_example())
        return fields

    def _probe_outputs(self, func, schema: Schema) -> list[Field]:
        """Call the function once with example inputs to learn the output shape.

        Best-effort: a model that can't be invoked at warmup (missing artifact,
        etc.) just yields an empty output schema rather than failing the load.
        """
        if schema.has_image_input():
            return []  # don't fabricate image bytes just to probe
        try:
            example = schema_mod.example_record(schema)
            kwargs = {f.name: schema_mod.coerce_value(example[f.name], f.type)
                      for f in schema.inputs}
            result = func(**kwargs)
            record = self._normalize_one(result)
            fields = []
            for k, v in record.items():
                jt = schema_mod.python_type_to_json(type(v))
                fields.append(Field(name=k, type=jt, required=False,
                                    example=schema_mod.jsonable(v)))
            return fields
        except Exception:
            return []

    @staticmethod
    def _normalize_one(result: Any) -> dict:
        """Coerce a single function return into a flat dict."""
        result = schema_mod.jsonable(result)
        if isinstance(result, dict):
            return result
        # pandas DataFrame / Series
        try:
            import pandas as pd

            if isinstance(result, pd.DataFrame):
                return result.iloc[0].to_dict() if len(result) else {}
            if isinstance(result, pd.Series):
                return result.to_dict()
        except ImportError:
            pass
        if isinstance(result, (list, tuple)):
            return {"result": list(result)}
        return {"result": result}

    def predict(self, records: list[dict]) -> list[dict]:
        if self._func is None:
            raise RuntimeError("Adapter not warmed up")
        out = []
        for rec in records:
            if self.schema.passthrough:
                # Forward the whole record as kwargs; we don't know the types.
                kwargs = dict(rec)
            else:
                kwargs = {}
                for f in self.schema.inputs:
                    if f.name in rec:
                        kwargs[f.name] = schema_mod.coerce_value(rec[f.name], f.type)
            result = self._func(**kwargs)
            out.append(self._normalize_one(result))
        return out


# --- Registry adapter --------------------------------------------------------

class RegistryAdapter(ModelAdapter):
    """Load a registered MLflow model; schema from its signature."""

    def __init__(
        self,
        model_name: str,
        version: str | None = None,
        stage: str | None = None,
        model_uri: str | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.version = version
        self.stage = stage
        self._explicit_uri = model_uri
        self._model = None
        self.name = name or model_name
        self.slug = slugify(self.name)

    def _uri(self) -> str:
        if self._explicit_uri:
            return self._explicit_uri
        if self.version:
            return f"models:/{self.model_name}/{self.version}"
        if self.stage:
            return f"models:/{self.model_name}/{self.stage}"
        return f"models:/{self.model_name}/latest"

    def warmup(self) -> None:
        import mlflow.pyfunc

        uri = self._uri()
        # Make MLflow import THIS model's bundled code, not a same-named module
        # cached by a previously loaded model (see _purge_sibling_modules).
        _purge_sibling_modules()
        self._model = mlflow.pyfunc.load_model(uri)
        self.description = f"Registered MLflow model ({uri})"
        self.schema = self._schema_from_signature()

    def _schema_from_signature(self) -> Schema:
        meta = getattr(self._model, "metadata", None)
        sig = getattr(meta, "signature", None) if meta else None
        inputs: list[Field] = []
        outputs: list[Field] = []
        if sig is not None:
            inputs = self._fields_from_mlflow_schema(sig.inputs, default_required=True)
            if sig.outputs is not None:
                outputs = self._fields_from_mlflow_schema(sig.outputs, default_required=False)
        schema = Schema(inputs=inputs, outputs=outputs)
        if not inputs:
            # No signature (or an inputs-less one): forward arbitrary JSON to the
            # pyfunc unvalidated rather than breaking the endpoint.
            schema.passthrough = True
            schema.notes = ("This model has no MLflow input signature; the endpoint "
                            "accepts arbitrary JSON and forwards it to the model "
                            "(validation skipped).")
        return schema

    @staticmethod
    def _fields_from_mlflow_schema(ml_schema, default_required: bool) -> list[Field]:
        fields: list[Field] = []
        try:
            cols = ml_schema.inputs  # mlflow Schema.inputs -> list[ColSpec/TensorSpec]
        except Exception:
            return fields
        for i, col in enumerate(cols):
            cname = getattr(col, "name", None) or f"input_{i}"
            ctype = getattr(col, "type", None)
            type_name = getattr(ctype, "name", None) or str(ctype)
            jtype = schema_mod.python_type_to_json(type_name)
            fields.append(
                Field(name=str(cname), type=jtype, required=default_required).with_default_example()
            )
        return fields

    def predict(self, records: list[dict]) -> list[dict]:
        if self._model is None:
            raise RuntimeError("Adapter not warmed up")
        import pandas as pd

        df = pd.DataFrame(records)
        result = self._model.predict(df)
        return self._normalize_result(result, n=len(records))

    def _normalize_result(self, result: Any, n: int) -> list[dict]:
        import numpy as np
        import pandas as pd

        if isinstance(result, pd.DataFrame):
            return [schema_mod.jsonable(r) for r in result.to_dict(orient="records")]
        if isinstance(result, pd.Series):
            return [{result.name or "result": schema_mod.jsonable(v)} for v in result.tolist()]
        arr = np.asarray(result)
        out_names = [f.name for f in self.schema.outputs]
        rows = []
        if arr.ndim == 1:
            for v in arr.tolist():
                rows.append({(out_names[0] if out_names else "result"): schema_mod.jsonable(v)})
        else:
            for row in arr.tolist():
                if out_names and len(out_names) == len(row):
                    rows.append({out_names[j]: schema_mod.jsonable(v) for j, v in enumerate(row)})
                else:
                    rows.append({f"output_{j}": schema_mod.jsonable(v) for j, v in enumerate(row)})
        return rows
