"""Self-contained MLflow ``pyfunc`` wrapper for the diabetes classifier.

This is the model that gets logged to the registry and **deployed as a Model
API directly from the registry entry** (no custom scoring code). It bundles
everything inference needs -- the fitted ``StandardScaler`` statistics, the
trained ``DiabetesNet`` weights and the decision threshold -- so it accepts the
*raw* feature columns used in training and returns calibrated probabilities
plus a label.

Because ``train.py`` logs this model **with a signature** (see
``mlflow.pyfunc.log_model(..., signature=...)``), Domino knows the exact input
and output schema and can auto-wrap it as a REST endpoint. The request body is
the named feature columns; the response is the probability + verdict.

Contrast with ``model_api.py``, which is the *custom-code* path: there you
point Domino at a file/function and write the glue yourself. Here the glue
lives inside the model artifact itself.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import torch

import mlflow.pyfunc

from model import DiabetesNet


class DiabetesClassifier(mlflow.pyfunc.PythonModel):
    """Pyfunc that owns preprocessing + the network + post-processing."""

    def load_context(self, context: Any) -> None:
        """Rebuild the model and scaler from the bundled checkpoint.

        Called once when the model is loaded (at endpoint startup), not per
        request. ``context.artifacts["checkpoint"]`` resolves to the ``.pt``
        file that was bundled via ``artifacts={"checkpoint": ...}`` at log time.
        """
        # weights_only=False: the checkpoint holds plain python objects
        # (config dict, feature list, scaler stats) alongside the tensors.
        ckpt = torch.load(
            context.artifacts["checkpoint"], map_location="cpu", weights_only=False
        )

        self._model = DiabetesNet.from_config(ckpt["config"])
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()

        self._mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
        self._scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
        self._features = list(ckpt["feature_columns"])
        # Threshold can be tuned per-deployment without re-logging the model.
        self._threshold = float(os.environ.get("MODEL_THRESHOLD", "0.5"))

    def predict(self, context, model_input, params=None):
        """Score one or more samples.

        Args:
            model_input: A pandas ``DataFrame`` with the training feature
                columns (MLflow hands the request body in as a DataFrame), or a
                dict / list-of-dicts when called directly in Python.

        Returns:
            A ``DataFrame`` with ``diabetes_probability`` (float) and
            ``is_diabetic`` (bool), one row per input sample.
        """
        df = self._coerce(model_input)

        missing = [c for c in self._features if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing feature columns: {missing}")

        X = df[self._features].to_numpy(dtype=np.float32)
        X_scaled = (X - self._mean) / self._scale

        with torch.no_grad():
            logits = self._model(torch.from_numpy(X_scaled.astype(np.float32)))
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()

        labels = (probs >= self._threshold).astype(bool)
        return pd.DataFrame(
            {"diabetes_probability": np.round(probs, 4), "is_diabetic": labels}
        )

    @staticmethod
    def _coerce(model_input: Any) -> pd.DataFrame:
        """Accept a DataFrame, a single record dict, or a list of records."""
        if isinstance(model_input, pd.DataFrame):
            return model_input
        if isinstance(model_input, dict):
            # {"col": [v1, v2]} (columnar) vs {"col": v} (single record).
            is_columnar = any(
                isinstance(v, (list, tuple, np.ndarray)) for v in model_input.values()
            )
            return pd.DataFrame(model_input if is_columnar else [model_input])
        return pd.DataFrame(model_input)
