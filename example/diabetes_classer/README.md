# Diabetes Classifier (PyTorch) — Example

A small, end-to-end example of training and serving a PyTorch neural network
that predicts diabetes from lifestyle/demographic features, integrated with
Domino's built-in MLflow experiment tracking.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | Configurable MLP architecture (`DiabetesNet`) + feature definitions. |
| `train.py` | Training script — configurable arch & hyperparameters, MLflow tracking, saves model binary to disk, logs a **signed pyfunc** to the registry. |
| `pyfunc_model.py` | `mlflow.pyfunc` wrapper (scaler + net + threshold) — the model deployed **from the registry entry**. |
| `model_api.py` | Custom-code Model API entrypoint (`predict`) — the model deployed **from a file/function**, and also a local CLI for inference (uses GPU if available else CPU). |
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

## Predict (local CLI)

`model_api.py` doubles as a CLI: run it directly to score on disk. It loads the
on-disk binary and **automatically uses a GPU when one is available, otherwise
CPU** — the same warm model the hosted endpoint uses.

```bash
# Demo: scores two built-in sample patients
python model_api.py

# Single patient (feature values in column order)
python model_api.py --features 8000 2.5 0.6 60000 1 180

# Batch from a CSV (must contain the feature columns), write results
python model_api.py --input new_patients.csv --output predictions.csv

# Adjust the decision threshold
python model_api.py --features 8000 2.5 0.6 60000 1 180 --threshold 0.4
```

Output adds `diabetes_probability` and `predicted_is_diabetic` columns.

## Deploy as a Model API

There are two ways to serve this model in Domino. Both are included here.

### Option A — Custom code (`model_api.py`)

You point Domino at a file and function and write the scoring glue yourself.

1. **Publish** → **Model APIs** → **New Model**
2. File: `example/diabetes_classer/model_api.py`, Function: `predict`
3. Pick an environment that has the deps in `requirements.txt` (including `uwsgi`).

Domino wraps the request in a `data` object whose fields become the `predict()`
keyword arguments. Values may be sent as strings — the function coerces them.

Request:

```json
{
  "data": {
    "calories_wk": "3000.0",
    "hrs_exercise_wk": "5.0",
    "exercise_intensity": "0.8",
    "annual_income": "120000.0",
    "num_children": "0.0",
    "weight": "150.0"
  }
}
```

Response (the function's returned dict, under the response `result`):

```json
{ "is_diabetic": false, "probability": 0.0067, "threshold": 0.5 }
```

Call it with `curl` (find the URL + token on the endpoint's **Overview** tab):

```bash
curl -X POST "$MODEL_API_URL" \
  -H "Authorization: Bearer $MODEL_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"data": {"calories_wk": "3000.0", "hrs_exercise_wk": "5.0", "exercise_intensity": "0.8", "annual_income": "120000.0", "num_children": "0.0", "weight": "150.0"}}'
```

This path loads the binary from `$DOMINO_ARTIFACTS_DIR/diabetes_model/diabetes_model.pt`
at import time. Tune the cut-off with the `MODEL_THRESHOLD` env var.

### Option B — From the registry entry (auto-wrapped, no scoring code)

`train.py` logs a **signed `pyfunc`** (see `pyfunc_model.py`) that bundles the
scaler + network + threshold. Because it is logged **with a signature**
(inferred from a named-feature example), Domino reads the input/output schema
and **auto-wraps** it as an endpoint — you don't write any scoring code.

1. **Experiments** → open the training run → register the **`model`** logged model.
   (Register `model`, *not* a raw artifact folder — only the signed pyfunc carries
   the schema + a `requirements.txt`.)
2. **Model Registry** → the registered version → **Deploy** as a Model API.

Schema Domino auto-wraps to (inputs carry their real numeric types — see note
below):

```
inputs : calories_wk, hrs_exercise_wk, exercise_intensity,
         annual_income, num_children, weight   (all double)
outputs: diabetes_probability (float), is_diabetic (boolean)
```

Domino fronts the registry endpoint with the **same `{"data": {...}}` envelope**
as the custom-code path, so the identical request works on both:

```json
{
  "data": {
    "calories_wk": 3000.0,
    "hrs_exercise_wk": 5.0,
    "exercise_intensity": 0.8,
    "annual_income": 120000.0,
    "num_children": 0,
    "weight": 150.0
  }
}
```

Response (Domino adds `release` / `timing` / `request_id` metadata around it):

```json
{ "diabetes_probability": 0.0067, "is_diabetic": false }
```

> **Numbers or strings?** The model is logged with a **numeric** signature, which
> is the more permissive contract: MLflow coerces quoted strings (`"3000.0"`) to
> the declared numeric type, so the endpoint accepts **both** native JSON numbers
> *and* string-valued payloads, and the pyfunc coerces once more before scoring.
> (Declaring the columns as `string` instead would make MLflow *reject* native
> numbers and force every caller to quote — which is why the numeric signature is
> the better default.)

> **Why pyfunc and not `mlflow.pytorch.log_model`?** Logging the bare network
> would serve only the `DiabetesNet` — no scaler and no schema — so it would
> expect a pre-scaled tensor and would not auto-wrap to the named features.

## Setup

```bash
# CPU-only torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy scikit-learn mlflow

# For a GPU hardware tier, install the default CUDA build instead:
pip install torch
```
