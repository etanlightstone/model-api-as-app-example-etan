"""Domino Model API entrypoint for the diabetes classifier.

This is the file/function to point a Domino Model API at when hosting the model
*without* the automated model-registry deployment. Configure the published
model with:

    File:     model_api.py
    Function: predict

Domino wraps this function in a web server: each request's JSON body is passed
as keyword arguments, and the returned dict is serialized back as the JSON
response.

The model binary is loaded **once at import time** (not per request) so it is
warm for every prediction.

Example request body::

    {
      "calories_wk": 8000,
      "hrs_exercise_wk": 2.5,
      "exercise_intensity": 0.6,
      "annual_income": 60000,
      "num_children": 1,
      "weight": 180
    }

Example response::

    {
      "is_diabetic": true,
      "probability": 0.6283,
      "threshold": 0.5
    }

Quick local test::

    python model_api.py
"""

from __future__ import annotations

import os

import torch

from model import FEATURE_COLUMNS
from predict import DEFAULT_MODEL_PATH, load_model, predict as _predict


# Decision threshold for the positive class. Override with the MODEL_THRESHOLD
# environment variable when publishing the Model API if you want a different cut.
THRESHOLD = float(os.environ.get("MODEL_THRESHOLD", "0.5"))

# ---- Load once at import (kept warm across requests) ----
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL, _SCALER_MEAN, _SCALER_SCALE, _FEATURE_COLUMNS = load_model(
    DEFAULT_MODEL_PATH, _DEVICE
)


def predict(
    calories_wk: float,
    hrs_exercise_wk: float,
    exercise_intensity: float,
    annual_income: float,
    num_children: float,
    weight: float,
) -> dict:
    """Score a single patient and return the diabetes prediction.

    Args mirror the training feature columns. Domino passes the request JSON
    fields in as these keyword arguments.

    Returns a JSON-serializable dict with the boolean verdict, the probability
    of diabetes, and the threshold used.
    """
    # Assemble the feature row in the exact order the model expects.
    features = {
        "calories_wk": calories_wk,
        "hrs_exercise_wk": hrs_exercise_wk,
        "exercise_intensity": exercise_intensity,
        "annual_income": annual_income,
        "num_children": num_children,
        "weight": weight,
    }
    row = [[float(features[col]) for col in _FEATURE_COLUMNS]]

    probs, labels = _predict(
        _MODEL, _SCALER_MEAN, _SCALER_SCALE, row, _DEVICE, THRESHOLD
    )

    return {
        "is_diabetic": bool(labels[0]),
        "probability": round(float(probs[0]), 4),
        "threshold": THRESHOLD,
    }


if __name__ == "__main__":
    # Local smoke test of the scoring function (same call shape Domino uses).
    print("Feature order:", FEATURE_COLUMNS)
    print(predict(
        calories_wk=8000,
        hrs_exercise_wk=2.5,
        exercise_intensity=0.6,
        annual_income=60000,
        num_children=1,
        weight=180,
    ))
