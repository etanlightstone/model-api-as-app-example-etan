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
| `pyfunc_model.py` | `mlflow.pyfunc` wrapper — the model deployed **from the registry entry**. |
| `model_api.py` | Custom-code Model API entrypoint (`predict`) — the model deployed **from a file/function**, and also a local CLI for inference. |
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

`model_api.py` doubles as a CLI: run it directly to score on disk, using the
same warm pipeline the hosted endpoint uses.

```bash
# Demo: scores two built-in samples
python model_api.py

# Single sample via named flags
python model_api.py --month 7 --week-of 28 --state Alabama \
    --precipitation 0.1 --wind-speed 5.0 --wind-direction 20

# Batch from a CSV (must contain the feature columns), write results
python model_api.py --input new_weather.csv --output predictions.csv
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
inputs : month, week_of, precipitation, wind_speed, wind_direction (double),
         state (string)
outputs: avg_temp (double), max_temp (double), min_temp (double)
```

Each input carries its real type — the numeric features as numbers, `state` as a
string (see note below). Domino fronts the registry endpoint with the **same
`{"data": {...}}` envelope** as the custom-code path, so the identical request
works on both:

```json
{
  "data": {
    "month": 7,
    "week_of": 28,
    "state": "Alabama",
    "precipitation": 0.1,
    "wind_speed": 5.0,
    "wind_direction": 20
  }
}
```

Response (Domino adds `release` / `timing` / `request_id` metadata around it):

```json
{ "avg_temp": 83.24, "max_temp": 93.28, "min_temp": 72.6 }
```

> **Numbers or strings?** The numeric features are logged with their real numeric
> types, which is the more permissive contract: MLflow coerces quoted strings
> (`"5.0"`) to the declared numeric type, so the endpoint accepts **both** native
> JSON numbers *and* string-valued payloads, and the pyfunc coerces once more
> before scoring. (Casting the numerics to `string` instead would make MLflow
> *reject* native numbers and force every caller to quote.) `state` is a genuine
> string and stays one.

> **Why pyfunc and not `mlflow.sklearn.log_model`?** The pyfunc returns the
> predictions as **named columns** (`avg_temp`/`max_temp`/`min_temp`) rather
> than a bare array, so the endpoint response is self-describing.

## Setup

```bash
pip install scikit-learn pandas numpy joblib mlflow
# uwsgi is required by Domino for serving Model API endpoints:
pip install uwsgi
```
