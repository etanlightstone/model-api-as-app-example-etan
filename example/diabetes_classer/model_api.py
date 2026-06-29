"""Diabetes classifier — Domino Model API entrypoint *and* local CLI, one file.

Two ways to run the same scoring code:

* **As a Domino Model API (custom-code path).** Point the published model at:

      File:     model_api.py
      Function: predict

  Domino wraps ``predict(...)`` in a web server: each request's JSON fields are
  passed in as keyword arguments and the returned dict is the JSON response.

* **From the terminal.** Score a CSV of patients, a single sample, or the
  built-in demos::

      python model_api.py                                   # built-in demo samples
      python model_api.py --features 8000 2.5 0.6 60000 1 180
      python model_api.py --input new_patients.csv --output predictions.csv

The model binary (written by ``train.py``) is loaded **once at import time**, so
it is warm for every Model API request and reused as-is by the CLI. Uses the GPU
automatically when one is available, otherwise CPU.

Example request body::

    {
      "calories_wk": 8000, "hrs_exercise_wk": 2.5, "exercise_intensity": 0.6,
      "annual_income": 60000, "num_children": 1, "weight": 180
    }

Example response::

    {"is_diabetic": true, "probability": 0.6283, "threshold": 0.5}
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the sibling module (model.py) imports cleanly regardless of the working
# directory Domino loads this Model API file from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

from model import DiabetesNet


DEFAULT_MODEL_PATH = os.path.join(
    os.environ.get("DOMINO_ARTIFACTS_DIR", "/mnt/artifacts"),
    "diabetes_model",
    "diabetes_model.pt",
)

# Decision threshold for the positive class. Override with the MODEL_THRESHOLD
# environment variable when publishing the Model API (or --threshold on the CLI).
THRESHOLD = float(os.environ.get("MODEL_THRESHOLD", "0.5"))


def load_model(model_path: str, device: torch.device):
    """Load the checkpoint and rebuild the model + scaler statistics."""
    # weights_only=False because the checkpoint stores plain python objects
    # (config dict, feature list, scaler stats) alongside the tensors.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = DiabetesNet.from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return (
        model,
        np.asarray(checkpoint["scaler_mean"], dtype=np.float32),
        np.asarray(checkpoint["scaler_scale"], dtype=np.float32),
        list(checkpoint["feature_columns"]),
    )


# ---- Load once at import (warm for every request, reused by the CLI) ----
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL, _SCALER_MEAN, _SCALER_SCALE, _FEATURE_COLUMNS = load_model(
    DEFAULT_MODEL_PATH, _DEVICE
)


def _score(X, threshold: float = THRESHOLD):
    """Standardize raw feature rows and return (probabilities, labels).

    The scaler stats were fit during training and saved in the checkpoint, so we
    apply the exact same standardization the model was trained on.
    """
    X = np.asarray(X, dtype=np.float32)
    X_scaled = (X - _SCALER_MEAN) / _SCALER_SCALE
    tensor = torch.from_numpy(X_scaled.astype(np.float32)).to(_DEVICE)

    with torch.no_grad():
        logits = _MODEL(tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    labels = (probs >= threshold).astype(int)
    return probs, labels


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
    fields in as these keyword arguments and serializes the returned dict back as
    the JSON response.
    """
    features = {
        "calories_wk": calories_wk,
        "hrs_exercise_wk": hrs_exercise_wk,
        "exercise_intensity": exercise_intensity,
        "annual_income": annual_income,
        "num_children": num_children,
        "weight": weight,
    }
    # Assemble the feature row in the exact order the model expects.
    row = [[float(features[col]) for col in _FEATURE_COLUMNS]]
    probs, labels = _score(row)

    return {
        "is_diabetic": bool(labels[0]),
        "probability": round(float(probs[0]), 4),
        "threshold": THRESHOLD,
    }


def main() -> None:
    """Score from the terminal: a CSV batch, a single sample, or built-in demos."""
    p = argparse.ArgumentParser(
        description="Score patients with the trained diabetes classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=None,
                   help="CSV of samples to score (must contain the feature columns).")
    p.add_argument("--features", type=float, nargs="+", default=None,
                   help="A single sample's feature values, in column order.")
    p.add_argument("--output", default=None,
                   help="Optional path to write predictions as CSV.")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help="Probability threshold for the positive class.")
    args = p.parse_args()

    print(f"Using device: {_DEVICE}")
    print(f"Expected feature order: {_FEATURE_COLUMNS}")

    # ----- Assemble the input matrix -----
    if args.input:
        df = pd.read_csv(args.input)
        missing = [c for c in _FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Input CSV is missing feature columns: {missing}")
    elif args.features:
        if len(args.features) != len(_FEATURE_COLUMNS):
            raise ValueError(
                f"Expected {len(_FEATURE_COLUMNS)} feature values "
                f"{_FEATURE_COLUMNS}, got {len(args.features)}."
            )
        df = pd.DataFrame([args.features], columns=_FEATURE_COLUMNS)
    else:
        # Built-in demo samples (active lifestyle vs. sedentary).
        print("No input provided -- scoring two built-in demo samples.")
        df = pd.DataFrame(
            [[3000, 5.0, 0.8, 120000, 0, 150.0],
             [18000, 0.3, 0.05, 25000, 3, 290.0]],
            columns=_FEATURE_COLUMNS,
        )

    probs, labels = _score(df[_FEATURE_COLUMNS].values, args.threshold)

    result = df.copy()
    result["diabetes_probability"] = probs
    result["predicted_is_diabetic"] = labels

    # Clear, human-readable verdict per sample.
    print("\nPredictions:")
    for i, (prob, label) in enumerate(zip(probs, labels)):
        verdict = "DIABETIC" if label == 1 else "NOT diabetic"
        prefix = f"Sample {i + 1}: " if len(probs) > 1 else ""
        print(f"  {prefix}{verdict}  ({prob * 100:.1f}% probability of diabetes)")

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nWrote detailed predictions to: {args.output}")


if __name__ == "__main__":
    main()
