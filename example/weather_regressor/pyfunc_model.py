"""Self-contained MLflow ``pyfunc`` wrapper for the weather regressor.

This is the model that gets logged to the registry and **deployed as a Model
API directly from the registry entry** (no custom scoring code). It bundles the
fitted scikit-learn pipeline (preprocessing + ``RandomForestRegressor``) and
returns the predicted temperatures as **named columns**.

Because ``train.py`` logs this model **with a signature** (inferred from a
named-feature example), Domino knows the input/output schema and auto-wraps it
as a REST endpoint: the request body is the named features, the response is the
predicted temperature(s).

Contrast with ``model_api.py``, the *custom-code* path, where you point Domino
at a file/function and write the glue yourself.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

import mlflow.pyfunc

from model import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class WeatherRegressor(mlflow.pyfunc.PythonModel):
    """Pyfunc that owns the full scikit-learn pipeline + output naming."""

    def load_context(self, context):
        """Load the bundled pipeline once at endpoint startup."""
        bundle = joblib.load(context.artifacts["bundle"])
        self._pipeline = bundle["pipeline"]
        self._features = list(bundle["feature_columns"])
        self._targets = list(bundle["target_columns"])

    def predict(self, context, model_input, params=None):
        """Score one or more samples.

        Args:
            model_input: A pandas ``DataFrame`` with the feature columns (MLflow
                hands the request body in as a DataFrame), or a dict /
                list-of-dicts when called directly in Python.

        Returns:
            A ``DataFrame`` with one column per predicted temperature target,
            one row per input sample.
        """
        df = self._coerce(model_input).copy()

        missing = [c for c in self._features if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing feature columns: {missing}")

        # Domino forwards request values verbatim, and they typically arrive as
        # strings (e.g. "7"). The model's signature is declared as strings to
        # accept them; coerce each column to the type the pipeline expects.
        for col in NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col])
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].astype(str)

        preds = np.asarray(self._pipeline.predict(df[self._features]))
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        return pd.DataFrame(np.round(preds, 2), columns=self._targets)

    @staticmethod
    def _coerce(model_input) -> pd.DataFrame:
        """Accept a DataFrame, a single record dict, or a list of records."""
        if isinstance(model_input, pd.DataFrame):
            return model_input
        if isinstance(model_input, dict):
            is_columnar = any(
                isinstance(v, (list, tuple, np.ndarray)) for v in model_input.values()
            )
            return pd.DataFrame(model_input if is_columnar else [model_input])
        return pd.DataFrame(model_input)
