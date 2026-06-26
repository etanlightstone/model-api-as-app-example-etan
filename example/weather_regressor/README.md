# Weather Temperature Regressor (scikit-learn) — Example

An end-to-end example of training and serving a **scikit-learn** model that
predicts temperature (average / max / min) from date, location and weather
features, integrated with Domino's built-in MLflow tracking and registry.

It mirrors the `diabetes_classer` example but is a **multi-output regression**
problem built on a scikit-learn `Pipeline` instead of PyTorch.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | Feature/target definitions + `build_pipeline()` (preprocessing + `HistGradientBoostingRegressor`). |
| `train.py` | Training script — MLflow tracking, saves the fitted pipeline to disk, logs a **signed pyfunc** to the registry. |
| `predict.py` | Inference script — loads the pipeline bundle from disk. |
| `pyfunc_model.py` | `mlflow.pyfunc` wrapper — the model deployed **from the registry entry**. |
| `model_api.py` | Custom-code Model API entrypoint (`predict`) — the model deployed **from a file/function**. |
| `requirements.txt` | Python dependencies. |

## Data

Uses `weather.csv` from this project's dataset
(`$DOMINO_DATASETS_DIR/$DOMINO_PROJECT_NAME/weather.csv`).

The raw dataset columns (`Date.Month`, `Data.Temperature.Avg Temp`, …) are
renamed to clean snake_case so the served APIs expose a friendly JSON schema:

| Role | Columns |
| --- | --- |
| Features | `month`, `week_of`, `precipitation`, `wind_speed`, `wind_direction`, `state` |
| Targets | `avg_temp`, `max_temp`, `min_temp` (degrees F) |

## Model

A single scikit-learn `Pipeline`:

```
ColumnTransformer( StandardScaler(numeric) + OneHotEncoder(state) )
  -> MultiOutputRegressor( HistGradientBoostingRegressor )
```

Gradient-boosted trees give strong accuracy (val R² ≈ 0.89) while keeping the
serialized model small (~3 MB) — which matters when it is bundled into the
MLflow model and deployed. The preprocessing lives *inside* the pipeline, so the
fitted estimator accepts raw feature values directly.

## Train

```bash
# Defaults: predict avg/max/min temperature
python train.py

# More boosting iterations, only the average temperature
python train.py --targets avg_temp --max-iter 800 --learning-rate 0.05
```

Training logs params/metrics to Domino's MLflow (experiment defaults to
`weather-sklearn-<project>`), writes the fitted pipeline to
`$DOMINO_ARTIFACTS_DIR/weather_model/weather_model.joblib`, and logs a signed
pyfunc named `model` for registry deployment. Run `python train.py --help` for
all options.

## Predict (local CLI)

```bash
# Demo: scores two built-in samples
python predict.py

# Single sample via named flags
python predict.py --month 7 --week-of 28 --state Alabama \
    --precipitation 0.1 --wind-speed 5.0 --wind-direction 20

# Batch from a CSV (must contain the feature columns), write results
python predict.py --input new_weather.csv --output predictions.csv
```

## Deploy as a Model API

Two ways to serve this model in Domino. Both are included here.

### Option A — Custom code (`model_api.py`)

You point Domino at a file and function and write the scoring glue yourself.

1. **Publish** → **Model APIs** → **New Model**
2. File: `example/weather_regressor/model_api.py`, Function: `predict`
3. Pick an environment with the deps in `requirements.txt` (including `uwsgi`).

Domino wraps the request in a `data` object whose fields become the `predict()`
keyword arguments. Values may be sent as strings — the function coerces them.

Request:

```json
{
  "data": {
    "month": "7",
    "week_of": "28",
    "state": "Alabama",
    "precipitation": "0.1",
    "wind_speed": "5.0",
    "wind_direction": "20"
  }
}
```

Response (the function's returned dict, under the response `result`):

```json
{ "avg_temp": 83.2, "max_temp": 93.3, "min_temp": 72.6 }
```

Call it with `curl` (find the URL + token on the endpoint's **Overview** tab):

```bash
curl -X POST "$MODEL_API_URL" \
  -H "Authorization: Bearer $MODEL_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"data": {"month": "7", "week_of": "28", "state": "Alabama", "precipitation": "0.1", "wind_speed": "5.0", "wind_direction": "20"}}'
```

This path loads the pipeline from
`$DOMINO_ARTIFACTS_DIR/weather_model/weather_model.joblib` at import time.

### Option B — From the registry entry (auto-wrapped, no scoring code)

`train.py` logs a **signed `pyfunc`** (see `pyfunc_model.py`) that bundles the
whole pipeline. Because it is logged **with a signature**, Domino reads the
input/output schema and **auto-wraps** it as an endpoint — no scoring code.

1. **Experiments** → open the training run → register the **`model`** logged
   model (the signed pyfunc — it carries the schema + a `requirements.txt`).
2. **Model Registry** → the registered version → **Deploy** as a Model API.

Schema Domino auto-wraps to:

```
inputs : month (long), week_of (long), wind_direction (long),
         precipitation (double), wind_speed (double), state (string)
outputs: avg_temp (double), max_temp (double), min_temp (double)
```

The auto-wrapped model speaks MLflow's standard scoring format. Request:

```json
{
  "dataframe_records": [
    { "month": 7, "week_of": 28, "wind_direction": 20,
      "precipitation": 0.1, "wind_speed": 5.0, "state": "Alabama" }
  ]
}
```

Response:

```json
{ "predictions": [ { "avg_temp": 83.24, "max_temp": 93.28, "min_temp": 72.6 } ] }
```

> **Confirm the request envelope for your Domino version.** The above is the
> native MLflow scoring schema; depending on how Domino fronts registry models,
> the body may instead be wrapped (e.g. under `data`). Check the endpoint's
> **Overview**/sample-request tab once deployed.

> **Match the signature types.** MLflow enforces the schema strictly: send the
> integer features (`month`, `week_of`, `wind_direction`) as integers and the
> rest (`precipitation`, `wind_speed`) as numbers with a decimal point.

> **Why pyfunc and not `mlflow.sklearn.log_model`?** The pyfunc returns the
> predictions as **named columns** (`avg_temp`/`max_temp`/`min_temp`) rather
> than a bare array, so the endpoint response is self-describing.

## Setup

```bash
pip install scikit-learn pandas numpy joblib mlflow
# uwsgi is required by Domino for serving Model API endpoints:
pip install uwsgi
```
