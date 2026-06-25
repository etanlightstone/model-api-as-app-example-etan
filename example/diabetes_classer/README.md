# Diabetes Classifier (PyTorch) — Example

A small, end-to-end example of training and serving a PyTorch neural network
that predicts diabetes from lifestyle/demographic features, integrated with
Domino's built-in MLflow experiment tracking.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | Configurable MLP architecture (`DiabetesNet`) + feature definitions. |
| `train.py` | Training script — configurable arch & hyperparameters, MLflow tracking, saves model binary to disk. |
| `predict.py` | Inference script — loads the binary from disk, uses GPU if available else CPU. |
| `requirements.txt` | Python dependencies. |

## Data

Uses `diabetes_dataset.csv` from this project's dataset
(`$DOMINO_DATASETS_DIR/$DOMINO_PROJECT_NAME/diabetes_dataset.csv`).

Features: `calories_wk`, `hrs_exercise_wk`, `exercise_intensity`,
`annual_income`, `num_children`, `weight`. Target: `is_diabetic` (0/1).

## Model

`DiabetesNet` is a feed-forward MLP. The default is a small 2-hidden-layer
network (`[32, 16]`) with dropout. Depth and width are set by `--hidden-sizes`,
so the same architecture scales from tiny to large. Features are standardized
with a `StandardScaler` that is fit during training and saved inside the model
checkpoint, so inference applies the exact same scaling.

## Train

```bash
# Defaults: small net [32, 16], 30 epochs
python train.py

# Vary the size/shape of the network and the training regime
python train.py \
  --hidden-sizes 128 64 32 \
  --dropout 0.3 \
  --epochs 50 \
  --batch-size 256 \
  --learning-rate 5e-4 \
  --weight-decay 1e-5
```

Training logs params/metrics to Domino's MLflow (experiment defaults to
`diabetes-pytorch-<project>`) **and** writes the model binary to
`$DOMINO_ARTIFACTS_DIR/diabetes_model/diabetes_model.pt` (override with
`--output-dir`). The checkpoint bundles the weights, architecture config,
feature ordering, and the fitted scaler. A `metadata.json` is written alongside.

Run `python train.py --help` for all options.

## Predict

The inference script loads the on-disk binary and **automatically uses a GPU
when one is available, otherwise CPU**.

```bash
# Demo: scores two built-in sample patients
python predict.py

# Single patient (feature values in column order)
python predict.py --features 8000 2.5 0.6 60000 1 180

# Batch from a CSV (must contain the feature columns), write results
python predict.py --input new_patients.csv --output predictions.csv

# Point at a specific model binary / adjust decision threshold
python predict.py --model-path /path/to/diabetes_model.pt --threshold 0.4
```

Output adds `diabetes_probability` and `predicted_is_diabetic` columns.

## Setup

```bash
# CPU-only torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy scikit-learn mlflow

# For a GPU hardware tier, install the default CUDA build instead:
pip install torch
```
