"""Domino Model API entrypoint for the weather temperature regressor.

Point a Domino Model API at this file/function when hosting the model *without*
the automated model-registry deployment:

    File:     model_api.py
    Function: predict

Domino wraps this function in a web server: each request's JSON body is passed
as keyword arguments, and the returned dict is serialized back as the JSON
response. The model binary is loaded **once at import time** (not per request).

Example request body::

    {
      "month": 7,
      "week_of": 28,
      "state": "Alabama",
      "precipitation": 0.1,
      "wind_speed": 5.0,
      "wind_direction": 20
    }

Example response (depends on which targets the model was trained on)::

    {
      "Data.Temperature.Avg Temp": 82.7,
      "Data.Temperature.Max Temp": 93.0,
      "Data.Temperature.Min Temp": 72.1
    }

Quick local test::

    python model_api.py
"""

from __future__ import annotations

import pandas as pd
import torch

from predict import DEFAULT_MODEL_PATH, load_model, predict as _predict


# ---- Load once at import (kept warm across requests) ----
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_BUNDLE = load_model(DEFAULT_MODEL_PATH, _DEVICE)


def predict(
    month: int,
    week_of: int,
    state: str,
    precipitation: float,
    wind_speed: float,
    wind_direction: float,
) -> dict:
    """Predict the temperature metric(s) for a single sample.

    Args mirror the training feature columns. Domino passes the request JSON
    fields in as these keyword arguments.

    Returns a JSON-serializable dict mapping each trained target column to its
    predicted value (degrees Fahrenheit).
    """
    df = pd.DataFrame([{
        "Date.Month": month,
        "Date.Week of": week_of,
        "Data.Precipitation": precipitation,
        "Data.Wind.Speed": wind_speed,
        "Data.Wind.Direction": wind_direction,
        "Station.State": state,
    }])

    preds = _predict(_BUNDLE, df, _DEVICE)[0]
    return {
        target: round(float(value), 1)
        for target, value in zip(_BUNDLE["targets"], preds)
    }


if __name__ == "__main__":
    # Local smoke test of the scoring function (same call shape Domino uses).
    print(predict(
        month=7,
        week_of=28,
        state="Alabama",
        precipitation=0.1,
        wind_speed=5.0,
        wind_direction=20,
    ))
