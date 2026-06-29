"""Weather temperature regressor — Domino Model API entrypoint *and* local CLI.

Two ways to run the same scoring code, one file:

* **As a Domino Model API (custom-code path).** Point the published model at:

      File:     model_api.py
      Function: predict

  Domino wraps ``predict(...)`` in a web server: each request's JSON fields are
  passed in as keyword arguments and the returned dict is the JSON response.

* **From the terminal.** Score a CSV of samples, a single sample, or the
  built-in demos::

      python model_api.py                                   # built-in demo samples
      python model_api.py --month 7 --week-of 28 --state Alabama \
          --precipitation 0.1 --wind-speed 5.0 --wind-direction 20
      python model_api.py --input new_weather.csv --output predictions.csv

The fitted pipeline (written by ``train.py``) is loaded **once at import time**,
so it is warm for every Model API request and reused as-is by the CLI. For the
no-code alternative, register the signed pyfunc that ``train.py`` logs and deploy
it straight from the registry (see README).

Example request body::

    {
      "month": 7, "week_of": 28, "state": "Alabama",
      "precipitation": 0.1, "wind_speed": 5.0, "wind_direction": 20
    }

Example response::

    {"avg_temp": 82.7, "max_temp": 93.0, "min_temp": 72.1}
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the sibling module (model.py) imports cleanly regardless of the working
# directory Domino loads this Model API file from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

from model import FEATURE_COLUMNS


DEFAULT_MODEL_PATH = os.path.join(
    os.environ.get("DOMINO_ARTIFACTS_DIR", "/mnt/artifacts"),
    "weather_model",
    "weather_model.joblib",
)


def load_model(model_path: str):
    """Load the fitted pipeline bundle (pipeline + column names + targets)."""
    bundle = joblib.load(model_path)
    return bundle["pipeline"], bundle["feature_columns"], bundle["target_columns"]


# ---- Load once at import (warm for every request, reused by the CLI) ----
_PIPELINE, _FEATURE_COLUMNS, _TARGETS = load_model(DEFAULT_MODEL_PATH)


def _score(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one predicted column per target."""
    preds = np.asarray(_PIPELINE.predict(df[_FEATURE_COLUMNS]))
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    return pd.DataFrame(np.round(preds, 2), columns=_TARGETS, index=df.index)


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

    preds = _score(df)
    return {t: round(float(preds.iloc[0][t]), 1) for t in _TARGETS}


def _single_sample_df(args) -> pd.DataFrame:
    """Build a one-row frame from the named CLI flags (errors if any are missing)."""
    values = {
        "month": args.month,
        "week_of": args.week_of,
        "precipitation": args.precipitation,
        "wind_speed": args.wind_speed,
        "wind_direction": args.wind_direction,
        "state": args.state,
    }
    missing = [k for k, v in values.items() if v is None]
    if missing:
        raise ValueError(f"Missing feature flags: {missing}")
    return pd.DataFrame([values])[FEATURE_COLUMNS]


def main() -> None:
    """Score from the terminal: a CSV batch, a single sample, or built-in demos."""
    p = argparse.ArgumentParser(
        description="Score samples with the trained weather temperature regressor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=None,
                   help="CSV of samples to score (must contain the feature columns).")
    p.add_argument("--output", default=None,
                   help="Optional path to write predictions as CSV.")

    # Single-sample named flags (mirror the feature columns).
    p.add_argument("--month", type=int, help="Month (1-12).")
    p.add_argument("--week-of", type=int, help="Week of the year.")
    p.add_argument("--state", type=str, help="State, e.g. 'Alabama'.")
    p.add_argument("--precipitation", type=float, help="Precipitation.")
    p.add_argument("--wind-speed", type=float, help="Wind speed.")
    p.add_argument("--wind-direction", type=int, help="Wind direction (degrees).")
    args = p.parse_args()

    print(f"Targets: {_TARGETS}")

    if args.input:
        df = pd.read_csv(args.input)
        missing = [c for c in _FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Input CSV is missing feature columns: {missing}")
    elif args.month is not None or args.state is not None:
        df = _single_sample_df(args)
    else:
        print("No input provided -- scoring two built-in demo samples.")
        df = pd.DataFrame([
            {"month": 7, "week_of": 28, "precipitation": 0.1,
             "wind_speed": 5.0, "wind_direction": 20, "state": "Alabama"},
            {"month": 1, "week_of": 3, "precipitation": 0.0,
             "wind_speed": 9.7, "wind_direction": 320, "state": "Minnesota"},
        ])[FEATURE_COLUMNS]

    preds = _score(df)
    result = df.reset_index(drop=True).join(preds.reset_index(drop=True))

    print("\nPredictions (degrees F):")
    for i, row in preds.reset_index(drop=True).iterrows():
        desc = ", ".join(f"{t}={row[t]:.1f}" for t in _TARGETS)
        print(f"  Sample {i + 1}: {desc}")

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nWrote detailed predictions to: {args.output}")


if __name__ == "__main__":
    main()
