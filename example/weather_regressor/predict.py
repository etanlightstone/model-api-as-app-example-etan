"""Run inference with a trained weather temperature regressor (scikit-learn).

Loads the fitted pipeline bundle written by ``train.py`` from disk and predicts
the temperature metric(s) for new samples.

Input can come from a CSV file (one row per sample, with the friendly feature
columns) or from a single sample passed via named flags.

Examples
--------
Predict for every row in a CSV and write the results::

    python predict.py --input new_weather.csv --output predictions.csv

Predict for one sample given as named feature values::

    python predict.py --month 7 --week-of 28 --state Alabama \
        --precipitation 0.1 --wind-speed 5.0 --wind-direction 20

If no input is given, a couple of built-in demo samples are scored.
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from model import FEATURE_COLUMNS


DEFAULT_MODEL_PATH = os.path.join(
    os.environ.get("DOMINO_ARTIFACTS_DIR", "/mnt/artifacts"),
    "weather_model",
    "weather_model.joblib",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference with a trained weather temperature regressor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                   help="Path to the trained model bundle (.joblib).")
    p.add_argument("--input", default=None,
                   help="CSV file of samples to score (must contain the "
                        "feature columns).")
    p.add_argument("--output", default=None,
                   help="Optional path to write predictions as CSV.")

    # Single-sample named flags (mirror the feature columns).
    p.add_argument("--month", type=int, help="Month (1-12).")
    p.add_argument("--week-of", type=int, help="Week of the year.")
    p.add_argument("--state", type=str, help="State, e.g. 'Alabama'.")
    p.add_argument("--precipitation", type=float, help="Precipitation.")
    p.add_argument("--wind-speed", type=float, help="Wind speed.")
    p.add_argument("--wind-direction", type=int, help="Wind direction (degrees).")
    return p.parse_args()


def load_model(model_path: str):
    """Load the pipeline bundle (pipeline + column names)."""
    bundle = joblib.load(model_path)
    return bundle["pipeline"], bundle["feature_columns"], bundle["target_columns"]


def predict(pipeline, feature_columns, targets, df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one predicted column per target."""
    preds = np.asarray(pipeline.predict(df[feature_columns]))
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    return pd.DataFrame(np.round(preds, 2), columns=targets, index=df.index)


def _single_sample_df(args) -> pd.DataFrame:
    """Build a one-row frame from the named flags (errors if any are missing)."""
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
    args = parse_args()

    pipeline, feature_columns, targets = load_model(args.model_path)
    print(f"Loaded model from: {args.model_path}")
    print(f"Targets: {targets}")

    if args.input:
        df = pd.read_csv(args.input)
        missing = [c for c in feature_columns if c not in df.columns]
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

    preds = predict(pipeline, feature_columns, targets, df)

    result = df.reset_index(drop=True).join(preds.reset_index(drop=True))

    print("\nPredictions (degrees F):")
    for i, row in preds.reset_index(drop=True).iterrows():
        desc = ", ".join(f"{t}={row[t]:.1f}" for t in targets)
        print(f"  Sample {i + 1}: {desc}")

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nWrote detailed predictions to: {args.output}")


if __name__ == "__main__":
    main()
