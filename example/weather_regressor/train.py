"""Train the weather temperature regressor.

Trains a configurable PyTorch MLP (see ``model.py``) to predict one or more
continuous temperature metrics from date / location / precipitation / wind
features. All architecture and training hyperparameters are exposed as
command-line arguments with sensible defaults.

Like the diabetes example, this uses Domino's built-in MLflow to track params
and metrics, and *also* writes the trained model binary (weights + config +
fitted preprocessing) to disk so the inference script can load it directly.

Examples
--------
Train with defaults (predict Avg/Max/Min temperature)::

    python train.py

Predict only the average temperature, with a larger network::

    python train.py --targets "Data.Temperature.Avg Temp" \
        --hidden-sizes 128 64 32 --epochs 80 --learning-rate 5e-4
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.pytorch

from model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMNS,
    WeatherNet,
)


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
        description="Train a configurable PyTorch weather temperature regressor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / IO
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                   help="Path to the weather CSV file.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Directory to write the model binary and metadata.")
    p.add_argument("--targets", nargs="+", default=TARGET_COLUMNS,
                   help="One or more continuous columns to predict.")

    # Model architecture (vary size / shape here)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 32],
                   help="Hidden layer widths; number of values sets the depth.")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout probability after each hidden layer.")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="Adam LR.")
    p.add_argument("--weight-decay", type=float, default=0.0, help="L2 reg.")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data held out for validation.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    default_experiment = "weather-pytorch"
    project = os.environ.get("DOMINO_PROJECT_NAME")
    if project:
        default_experiment = f"{default_experiment}-{project}"
    p.add_argument("--experiment-name", default=default_experiment,
                   help="MLflow experiment name.")

    return p.parse_args()


def load_data(data_path: str, targets, val_split: float, seed: int):
    """Load the CSV, split, and fit feature + target preprocessing.

    Numeric features are standardized and categoricals one-hot encoded via a
    ``ColumnTransformer``. Targets are standardized too (helps multi-output
    training); predictions are inverse-transformed back to real units for
    metrics and inference.

    Returns train/val tensors plus the fitted preprocessor and target scaler.
    """
    df = pd.read_csv(data_path)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[targets].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    # OneHotEncoder may return a sparse matrix; densify for torch.
    X_train_t = np.asarray(X_train_t.todense() if hasattr(X_train_t, "todense")
                           else X_train_t, dtype=np.float32)
    X_val_t = np.asarray(X_val_t.todense() if hasattr(X_val_t, "todense")
                         else X_val_t, dtype=np.float32)

    target_scaler = StandardScaler()
    y_train_s = target_scaler.fit_transform(y_train).astype(np.float32)
    y_val_s = target_scaler.transform(y_val).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train_t), torch.from_numpy(y_train_s))
    val_ds = TensorDataset(torch.from_numpy(X_val_t), torch.from_numpy(y_val_s))
    return train_ds, val_ds, preprocessor, target_scaler, X_train_t.shape[1]


def evaluate(model, loader, device, target_scaler, targets):
    """Compute MAE, RMSE and R^2 per target in real (un-scaled) units."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss, n = 0.0, 0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
            all_targets.append(yb.cpu().numpy())
            all_preds.append(out.cpu().numpy())

    y_true_s = np.concatenate(all_targets)
    y_pred_s = np.concatenate(all_preds)

    # Back to real units before scoring.
    y_true = target_scaler.inverse_transform(y_true_s)
    y_pred = target_scaler.inverse_transform(y_pred_s)

    metrics = {"loss": total_loss / max(n, 1)}
    for i, name in enumerate(targets):
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        metrics[f"mae[{name}]"] = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        metrics[f"rmse[{name}]"] = rmse
        metrics[f"r2[{name}]"] = float(r2_score(y_true[:, i], y_pred[:, i]))
    return metrics


def _mlflow_safe(name: str) -> str:
    """MLflow metric keys can't contain some chars -- sanitize target names."""
    return name.replace(" ", "_").replace("[", "_").replace("]", "")


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Predicting target(s): {args.targets}")

    train_ds, val_ds, preprocessor, target_scaler, input_dim = load_data(
        args.data_path, args.targets, args.val_split, args.seed
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = WeatherNet(
        input_dim=input_dim,
        output_dim=len(args.targets),
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
    ).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # ----- Domino built-in MLflow tracking -----
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "targets": ", ".join(args.targets),
            "hidden_sizes": "-".join(map(str, args.hidden_sizes)),
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "val_split": args.val_split,
            "seed": args.seed,
            "input_dim": input_dim,
            "device": device.type,
        })

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss, n = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
                n += xb.size(0)

            train_loss = running_loss / max(n, 1)
            val_metrics = evaluate(model, val_loader, device, target_scaler, args.targets)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{_mlflow_safe(k)}", v, step=epoch)

            if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
                mae_avg = np.mean([val_metrics[f"mae[{t}]"] for t in args.targets])
                print(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"mean_val_MAE={mae_avg:.3f}"
                )

        # ----- Save the model binary to disk -----
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "weather_model.pt")

        # One checkpoint holds everything inference needs: weights, arch config,
        # the fitted feature preprocessor, the target scaler and column names.
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": model.config,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target_columns": args.targets,
            "preprocessor": preprocessor,
            "target_scaler": target_scaler,
        }
        torch.save(checkpoint, model_path)
        print(f"Saved model binary to: {model_path}")

        meta_path = os.path.join(args.output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "config": model.config,
                    "numeric_features": NUMERIC_FEATURES,
                    "categorical_features": CATEGORICAL_FEATURES,
                    "targets": args.targets,
                    "final_val_metrics": val_metrics,
                    "mlflow_run_id": run.info.run_id,
                },
                f,
                indent=2,
            )

        mlflow.log_artifact(model_path, artifact_path="model_binary")
        mlflow.log_artifact(meta_path, artifact_path="model_binary")
        mlflow.pytorch.log_model(model, name="model")

        print(f"MLflow run id: {run.info.run_id}")
        print("Final validation metrics:")
        print(json.dumps(val_metrics, indent=2))


if __name__ == "__main__":
    main()
