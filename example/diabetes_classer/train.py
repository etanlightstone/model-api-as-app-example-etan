"""Train the diabetes classifier.

Trains a configurable PyTorch MLP (see ``model.py``) on the diabetes dataset.
All architecture and training hyperparameters are exposed as command-line
arguments with sensible defaults, so you can vary the size/shape of the network
and the training regime without editing code.

The script uses Domino's built-in MLflow to track parameters, metrics and the
model artifact, and *also* writes the trained model binary (weights + config +
the fitted feature scaler) to disk so the inference script can load it directly.

Examples
--------
Train with defaults::

    python train.py

Train a larger/deeper network for longer::

    python train.py --hidden-sizes 128 64 32 --epochs 50 --batch-size 256 \
        --learning-rate 5e-4 --dropout 0.3
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.pytorch

from model import FEATURE_COLUMNS, TARGET_COLUMN, DiabetesNet


# Default location of the diabetes CSV inside this Domino project's dataset.
DEFAULT_DATA_PATH = os.path.join(
    os.environ.get("DOMINO_DATASETS_DIR", "/mnt/data"),
    os.environ.get("DOMINO_PROJECT_NAME", ""),
    "diabetes_dataset.csv",
)

# Default output directory for the model binary (persisted Domino artifacts).
DEFAULT_OUTPUT_DIR = os.path.join(
    os.environ.get("DOMINO_ARTIFACTS_DIR", "/mnt/artifacts"), "diabetes_model"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a configurable PyTorch diabetes classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / IO
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                   help="Path to the diabetes CSV file.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Directory to write the model binary and metadata.")

    # Model architecture (vary size / shape here)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[32, 16],
                   help="Hidden layer widths; number of values sets the depth.")
    p.add_argument("--dropout", type=float, default=0.2,
                   help="Dropout probability after each hidden layer.")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="Adam LR.")
    p.add_argument("--weight-decay", type=float, default=0.0, help="L2 reg.")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data held out for validation.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    # Default experiment name is namespaced by the Domino project so it does not
    # collide with experiments of the same name owned by other projects.
    default_experiment = "diabetes-pytorch"
    project = os.environ.get("DOMINO_PROJECT_NAME")
    if project:
        default_experiment = f"{default_experiment}-{project}"
    p.add_argument("--experiment-name", default=default_experiment,
                   help="MLflow experiment name.")

    return p.parse_args()


def load_data(data_path: str, val_split: float, seed: int):
    """Load the CSV, split, and standardize features.

    Returns train/val tensors and the fitted ``StandardScaler``.
    """
    df = pd.read_csv(data_path)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    return train_ds, val_ds, scaler


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Compute loss, accuracy, F1 and ROC-AUC over a data loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_targets.append(yb.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    targets = np.concatenate(all_targets)
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)

    return {
        "loss": total_loss / max(n, 1),
        "accuracy": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds, zero_division=0),
        "roc_auc": roc_auc_score(targets, probs),
    }


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, scaler = load_data(args.data_path, args.val_split, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = DiabetesNet(
        input_dim=len(FEATURE_COLUMNS),
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
    ).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # ----- Domino built-in MLflow tracking -----
    # The Domino workspace pre-configures the MLflow tracking URI, so we only
    # need to set the experiment and start a run.
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "hidden_sizes": "-".join(map(str, args.hidden_sizes)),
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "val_split": args.val_split,
            "seed": args.seed,
            "n_features": len(FEATURE_COLUMNS),
            "device": device.type,
        })

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss, n = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
                n += xb.size(0)

            train_loss = running_loss / max(n, 1)
            val_metrics = evaluate(model, val_loader, device)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)

            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"val_auc={val_metrics['roc_auc']:.4f}"
            )

        # ----- Save the model binary to disk -----
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "diabetes_model.pt")

        # A single checkpoint holds everything inference needs: the weights, the
        # architecture config (to rebuild the net), the fitted scaler stats, and
        # the feature ordering.
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": model.config,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }
        torch.save(checkpoint, model_path)
        print(f"Saved model binary to: {model_path}")

        # Also write a small human-readable metadata file next to the binary.
        meta_path = os.path.join(args.output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "config": model.config,
                    "feature_columns": FEATURE_COLUMNS,
                    "final_val_metrics": val_metrics,
                    "mlflow_run_id": run.info.run_id,
                },
                f,
                indent=2,
            )

        # ----- Log artifacts + model to MLflow -----
        mlflow.log_artifact(model_path, artifact_path="model_binary")
        mlflow.log_artifact(meta_path, artifact_path="model_binary")
        mlflow.pytorch.log_model(model, name="model")

        print(f"MLflow run id: {run.info.run_id}")
        print("Final validation metrics:", json.dumps(val_metrics, indent=2))


if __name__ == "__main__":
    main()
