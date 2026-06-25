"""Run inference with a trained weather temperature regressor.

Loads the model binary written by ``train.py`` from disk and predicts the
continuous temperature metric(s) for new samples. Uses the GPU automatically
when one is available, and falls back to CPU otherwise.

Input can come from a CSV file (one row per sample, with the same feature
columns used in training) or from a single sample passed via named flags.

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

import numpy as np
import pandas as pd
import torch

from model import NUMERIC_FEATURES, CATEGORICAL_FEATURES, WeatherNet


DEFAULT_MODEL_PATH = os.path.join(
    os.environ.get("DOMINO_ARTIFACTS_DIR", "/mnt/artifacts"),
    "weather_model",
    "weather_model.pt",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference with a trained weather temperature regressor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                   help="Path to the trained model binary (.pt).")
    p.add_argument("--input", default=None,
                   help="CSV file of samples to score (must contain the "
                        "feature columns used in training).")
    p.add_argument("--output", default=None,
                   help="Optional path to write predictions as CSV.")

    # Single-sample named flags (mirror the training feature columns).
    p.add_argument("--month", type=int, help="Date.Month (1-12).")
    p.add_argument("--week-of", type=int, help="Date.Week of.")
    p.add_argument("--state", type=str, help="Station.State, e.g. 'Alabama'.")
    p.add_argument("--precipitation", type=float, help="Data.Precipitation.")
    p.add_argument("--wind-speed", type=float, help="Data.Wind.Speed.")
    p.add_argument("--wind-direction", type=float, help="Data.Wind.Direction.")
    return p.parse_args()


def load_model(model_path: str, device: torch.device):
    """Load checkpoint and rebuild the model + preprocessing objects."""
    # weights_only=False because the checkpoint stores python objects
    # (config, the fitted sklearn preprocessor and target scaler).
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = WeatherNet.from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return {
        "model": model,
        "preprocessor": checkpoint["preprocessor"],
        "target_scaler": checkpoint["target_scaler"],
        "targets": checkpoint["target_columns"],
    }


def predict(bundle, df: pd.DataFrame, device):
    """Preprocess a feature DataFrame and return predictions in real units."""
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = bundle["preprocessor"].transform(df[feature_cols])
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
    tensor = torch.from_numpy(X).to(device)

    with torch.no_grad():
        out = bundle["model"](tensor).cpu().numpy()

    # Inverse-transform from standardized space back to degrees / real units.
    preds = bundle["target_scaler"].inverse_transform(out)
    return preds


def _single_sample_df(args) -> pd.DataFrame:
    """Build a one-row DataFrame from the named single-sample flags."""
    return pd.DataFrame([{
        "Date.Month": args.month,
        "Date.Week of": args.week_of,
        "Data.Precipitation": args.precipitation,
        "Data.Wind.Speed": args.wind_speed,
        "Data.Wind.Direction": args.wind_direction,
        "Station.State": args.state,
    }])


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bundle = load_model(args.model_path, device)
    targets = bundle["targets"]
    print(f"Loaded model from: {args.model_path}")
    print(f"Predicting target(s): {targets}")

    single_flags = [args.month, args.week_of, args.state,
                    args.precipitation, args.wind_speed, args.wind_direction]

    # ----- Assemble the input -----
    if args.input:
        df = pd.read_csv(args.input)
    elif any(v is not None for v in single_flags):
        if any(v is None for v in single_flags):
            raise ValueError(
                "For a single sample, provide all of: --month --week-of "
                "--state --precipitation --wind-speed --wind-direction."
            )
        df = _single_sample_df(args)
    else:
        # Built-in demo samples (summer in Alabama vs. winter in Alaska).
        print("No input provided -- scoring two built-in demo samples.")
        df = pd.DataFrame([
            {"Date.Month": 7, "Date.Week of": 28, "Data.Precipitation": 0.1,
             "Data.Wind.Speed": 5.0, "Data.Wind.Direction": 20,
             "Station.State": "Alabama"},
            {"Date.Month": 1, "Date.Week of": 3, "Data.Precipitation": 0.5,
             "Data.Wind.Speed": 9.0, "Data.Wind.Direction": 30,
             "Station.State": "Alaska"},
        ])
        print(
            "\nTry it yourself -- copy/paste and edit the values:\n"
            "\n    python predict.py --month 7 --week-of 28 --state Alabama "
            "--precipitation 0.1 --wind-speed 5.0 --wind-direction 20\n"
        )

    preds = predict(bundle, df, device)

    # Clear, human-readable output per sample.
    print("\nPredictions:")
    for i in range(len(df)):
        prefix = f"Sample {i + 1}: " if len(df) > 1 else ""
        parts = [f"{t.split('.')[-1]} = {preds[i, j]:.1f}°F"
                 for j, t in enumerate(targets)]
        print(f"  {prefix}" + "  |  ".join(parts))

    if args.output:
        result = df.copy()
        for j, t in enumerate(targets):
            result[f"pred[{t}]"] = preds[:, j]
        result.to_csv(args.output, index=False)
        print(f"\nWrote detailed predictions to: {args.output}")


if __name__ == "__main__":
    main()
