"""Train the weather temperature regressor (scikit-learn).

Fits a scikit-learn ``Pipeline`` (preprocessing + ``RandomForestRegressor``) to
predict the average / max / min temperature from date, location and weather
features. Uses Domino's built-in MLflow to track params and metrics, writes the
fitted pipeline to disk for the custom-code Model API, and logs a **signed
pyfunc** so the model can be registered and deployed straight from the registry.

Examples
--------
Train with defaults (predict avg/max/min temperature)::

    python train.py

More boosting iterations, predicting only the average temperature::

    python train.py --targets avg_temp --max-iter 800 --learning-rate 0.05
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature

from model import (
    FEATURE_COLUMNS,
    RAW_TO_FEATURE,
    RAW_TO_TARGET,
    TARGET_COLUMNS,
    build_pipeline,
)
from pyfunc_model import WeatherRegressor


DEFAULT_DATA_PATH = os.path.join(
    os.environ.get("DOMINO_DATASETS_DIR", "/mnt/data"),
    os.environ.get("DOMINO_PROJECT_NAME", ""),
    "weather.csv",
)

DEFAULT_OUTPUT_DIR = os.path.join(
    os.environ.get("DOMINO_ARTIFACTS_DIR", "/mnt/artifacts"), "weather_model"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a scikit-learn weather temperature regressor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / IO
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                   help="Path to the weather CSV file.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Directory to write the fitted pipeline and metadata.")
    p.add_argument("--targets", nargs="+", default=TARGET_COLUMNS,
                   choices=TARGET_COLUMNS,
                   help="One or more temperature targets to predict.")

    # Model hyperparameters (vary the boosting here)
    p.add_argument("--max-iter", type=int, default=400,
                   help="Boosting iterations (trees) per target.")
    p.add_argument("--learning-rate", type=float, default=0.08,
                   help="Boosting learning rate.")
    p.add_argument("--max-depth", type=int, default=None,
                   help="Max tree depth (None = no limit).")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data held out for validation.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    default_experiment = "weather-sklearn"
    project = os.environ.get("DOMINO_PROJECT_NAME")
    if project:
        default_experiment = f"{default_experiment}-{project}"
    p.add_argument("--experiment-name", default=default_experiment,
                   help="MLflow experiment name.")

    return p.parse_args()


def load_data(data_path: str, targets: list[str]):
    """Load the CSV and return friendly-named feature / target frames."""
    df = pd.read_csv(data_path)

    # Keep only the columns we use, renamed to the friendly snake_case names.
    feats = df[list(RAW_TO_FEATURE)].rename(columns=RAW_TO_FEATURE)
    tgts = df[list(RAW_TO_TARGET)].rename(columns=RAW_TO_TARGET)

    X = feats[FEATURE_COLUMNS]
    y = tgts[targets].astype(float)
    return X, y


def evaluate(pipeline, X_val, y_val, targets):
    """Per-target MAE / RMSE / R^2 in real units (degrees F)."""
    preds = np.asarray(pipeline.predict(X_val))
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    y_true = y_val.to_numpy()

    metrics = {}
    for i, name in enumerate(targets):
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], preds[:, i])))
        metrics[f"mae_{name}"] = float(mean_absolute_error(y_true[:, i], preds[:, i]))
        metrics[f"rmse_{name}"] = rmse
        metrics[f"r2_{name}"] = float(r2_score(y_true[:, i], preds[:, i]))
    return metrics


def main() -> None:
    args = parse_args()

    X, y = load_data(args.data_path, args.targets)
    print(f"Predicting target(s): {args.targets}")
    print(f"Loaded {len(X)} rows, {X.shape[1]} features.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=args.seed
    )

    pipeline = build_pipeline(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=args.seed,
    )

    # ----- Domino built-in MLflow tracking -----
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "targets": ", ".join(args.targets),
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "val_split": args.val_split,
            "seed": args.seed,
            "n_features": X.shape[1],
            "model": "MultiOutputRegressor[HistGradientBoostingRegressor]",
        })

        print("Fitting pipeline ...")
        pipeline.fit(X_train, y_train)

        val_metrics = evaluate(pipeline, X_val, y_val, args.targets)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        mae_avg = np.mean([val_metrics[f"mae_{t}"] for t in args.targets])
        r2_avg = np.mean([val_metrics[f"r2_{t}"] for t in args.targets])
        print(f"Validation: mean MAE={mae_avg:.3f} F | mean R^2={r2_avg:.4f}")

        # ----- Save the fitted pipeline bundle to disk -----
        # One file holds the pipeline plus the column names inference needs.
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "weather_model.joblib")
        bundle = {
            "pipeline": pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": list(args.targets),
        }
        joblib.dump(bundle, model_path)
        print(f"Saved model bundle to: {model_path}")

        meta_path = os.path.join(args.output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "feature_columns": FEATURE_COLUMNS,
                    "target_columns": list(args.targets),
                    "final_val_metrics": val_metrics,
                    "mlflow_run_id": run.info.run_id,
                },
                f,
                indent=2,
            )

        # ----- Log the model to MLflow as a signed pyfunc -----
        # The pyfunc (see pyfunc_model.py) bundles the whole pipeline and is
        # logged WITH A SIGNATURE inferred from a named-feature example, so
        # Domino can auto-wrap it as a Model API directly from the registry.
        here = os.path.dirname(os.path.abspath(__file__))

        # Compute the output schema from a real (numeric) prediction, but log
        # the INPUT example as strings so the deployed signature accepts the
        # string-valued payloads Domino forwards. The pyfunc coerces them back.
        numeric_example = X_val.head(2).reset_index(drop=True)
        output_example = pd.DataFrame(
            np.round(pipeline.predict(numeric_example), 2), columns=args.targets
        )
        input_example = numeric_example.astype(str)
        signature = infer_signature(input_example, output_example)

        mlflow.pyfunc.log_model(
            name="model",
            python_model=WeatherRegressor(),
            artifacts={"bundle": model_path},
            code_paths=[
                os.path.join(here, "model.py"),
                os.path.join(here, "pyfunc_model.py"),
            ],
            signature=signature,
            input_example=input_example,
            pip_requirements=[
                "scikit-learn", "pandas", "numpy", "mlflow", "cloudpickle", "joblib",
            ],
        )

        mlflow.log_artifact(meta_path)

        print(f"MLflow run id: {run.info.run_id}")
        print("Final validation metrics:")
        print(json.dumps(val_metrics, indent=2))


if __name__ == "__main__":
    main()
