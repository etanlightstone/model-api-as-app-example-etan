"""Domino Model API entrypoint for the weather temperature regressor.

This is the *custom-code* path: point a Domino Model API at this file/function
and Domino wraps it in a web server. Each request's JSON body is passed as
keyword arguments, and the returned dict is serialized as the JSON response.

    File:     example/weather_regressor/model_api.py
    Function: predict

The fitted pipeline is loaded **once at import time** (kept warm across
requests). For the no-code alternative, register the signed pyfunc that
``train.py`` logs and deploy it straight from the registry (see README).

Example request body::

    {
      "month": 7, "week_of": 28, "state": "Alabama",
      "precipitation": 0.1, "wind_speed": 5.0, "wind_direction": 20
    }

Example response::

    { "avg_temp": 82.7, "max_temp": 93.0, "min_temp": 72.1 }

Quick local test::

    python model_api.py
"""

from __future__ import annotations

import os
import sys

# Ensure the sibling modules (model.py, predict.py) import cleanly regardless of
# the working directory Domino loads this Model API file from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from model import FEATURE_COLUMNS
from predict import DEFAULT_MODEL_PATH, load_model, predict as _predict


# ---- Load once at import (kept warm across requests) ----
_PIPELINE, _FEATURE_COLUMNS, _TARGETS = load_model(DEFAULT_MODEL_PATH)


def predict(
    month: int,
    week_of: int,
    state: str,
    precipitation: float,
    wind_speed: float,
    wind_direction: float,
) -> dict:
    """Predict the temperature metric(s) for a single sample.

    Args mirror the feature columns; Domino passes the request JSON fields in as
    these keyword arguments. Returns a dict mapping each target to its predicted
    value in degrees Fahrenheit.
    """
    # Domino passes request fields through as-is, and values may arrive as
    # strings (e.g. "7"); coerce to the numeric types the pipeline expects.
    df = pd.DataFrame([{
        "month": int(float(month)),
        "week_of": int(float(week_of)),
        "precipitation": float(precipitation),
        "wind_speed": float(wind_speed),
        "wind_direction": int(float(wind_direction)),
        "state": str(state),
    }])[FEATURE_COLUMNS]

    preds = _predict(_PIPELINE, _FEATURE_COLUMNS, _TARGETS, df)
    return {t: round(float(preds.iloc[0][t]), 1) for t in _TARGETS}


if __name__ == "__main__":
    # Local smoke test of the scoring function (same call shape Domino uses).
    print("Feature order:", FEATURE_COLUMNS)
    print(predict(
        month=7,
        week_of=28,
        state="Alabama",
        precipitation=0.1,
        wind_speed=5.0,
        wind_direction=20,
    ))
