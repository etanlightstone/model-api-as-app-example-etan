"""Run inference with a trained diabetes classifier.

Loads the model binary written by ``train.py`` from disk and predicts diabetes
risk for new samples. Uses the GPU automatically when one is available, and
falls back to CPU otherwise.

Input can come from a CSV file (one row per sample, with the same feature
columns used in training) or from a single sample passed on the command line.

Examples
--------
Predict for every row in a CSV and write the results::

    python predict.py --input new_patients.csv --output predictions.csv

Predict for one sample given as feature values (in column order)::

    python predict.py --features 8000 2.5 0.6 60000 1 180

If no input is given, a couple of built-in demo samples are scored.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from model import DiabetesNet


DEFAULT_MODEL_PATH = os.path.join(
    os.environ.get("DOMINO_ARTIFACTS_DIR", "/mnt/artifacts"),
    "diabetes_model",
    "diabetes_model.pt",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference with a trained diabetes classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                   help="Path to the trained model binary (.pt).")
    p.add_argument("--input", default=None,
                   help="CSV file of samples to score (must contain the "
                        "feature columns used in training).")
    p.add_argument("--features", type=float, nargs="+", default=None,
                   help="A single sample's feature values, in column order.")
    p.add_argument("--output", default=None,
                   help="Optional path to write predictions as CSV.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability threshold for the positive class.")
    return p.parse_args()


def load_model(model_path: str, device: torch.device):
    """Load checkpoint and rebuild the model + scaler parameters."""
    # weights_only=False because the checkpoint stores plain python objects
    # (config dict, feature list) alongside the tensors.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = DiabetesNet.from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    scaler_mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    feature_columns = checkpoint["feature_columns"]
    return model, scaler_mean, scaler_scale, feature_columns


def predict(model, scaler_mean, scaler_scale, X, device, threshold):
    """Standardize inputs and return (probabilities, predicted labels)."""
    X = np.asarray(X, dtype=np.float32)
    X_scaled = (X - scaler_mean) / scaler_scale
    tensor = torch.from_numpy(X_scaled.astype(np.float32)).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    labels = (probs >= threshold).astype(int)
    return probs, labels


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, scaler_mean, scaler_scale, feature_columns = load_model(
        args.model_path, device
    )
    print(f"Loaded model from: {args.model_path}")
    print(f"Expected feature order: {feature_columns}")

    # ----- Assemble the input matrix -----
    if args.input:
        df = pd.read_csv(args.input)
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input CSV is missing feature columns: {missing}")
        X = df[feature_columns].values
        source_df = df
    elif args.features:
        if len(args.features) != len(feature_columns):
            raise ValueError(
                f"Expected {len(feature_columns)} feature values "
                f"{feature_columns}, got {len(args.features)}."
            )
        X = np.array([args.features])
        source_df = pd.DataFrame(X, columns=feature_columns)
    else:
        # Built-in demo samples (active lifestyle vs. sedentary).
        print("No input provided -- scoring two built-in demo samples.")
        X = np.array([
            [3000, 5.0, 0.8, 120000, 0, 150.0],
            [18000, 0.3, 0.05, 25000, 3, 290.0],
        ])
        source_df = pd.DataFrame(X, columns=feature_columns)

    probs, labels = predict(
        model, scaler_mean, scaler_scale, X, device, args.threshold
    )

    result = source_df.copy()
    result["diabetes_probability"] = probs
    result["predicted_is_diabetic"] = labels

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\nPredictions:")
    print(result.to_string(index=False))

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nWrote predictions to: {args.output}")


if __name__ == "__main__":
    main()
