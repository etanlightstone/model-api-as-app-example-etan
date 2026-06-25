# Weather Temperature Regressor (PyTorch) — Example

A small, end-to-end example of training and serving a PyTorch neural network
that **predicts continuous temperature metrics** from weather features. It is
the regression counterpart to the `diabetes_classer` classification example.

**Classification vs. regression:** the diabetes example predicts a discrete
label (diabetic / not). This example predicts continuous numbers (temperature
in °F) — and it's **multi-output**, predicting Avg / Max / Min temperature
together by default. Train on a single target with `--targets` if you prefer.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | Configurable MLP regressor (`WeatherNet`) + feature/target definitions. |
| `train.py` | Training script — configurable arch & hyperparameters, MLflow tracking, saves model binary to disk. |
| `predict.py` | Inference script — loads the binary from disk, uses GPU if available else CPU. |
| `model_api.py` | Domino Model API entrypoint (`predict` function) for manual hosting. |
| `requirements.txt` | Python dependencies. |

## Data

Uses `weather.csv` from this project's dataset
(`$DOMINO_DATASETS_DIR/$DOMINO_PROJECT_NAME/weather.csv`) — weekly US city
weather for 2016–17.

- **Features:** `Date.Month`, `Date.Week of`, `Data.Precipitation`,
  `Data.Wind.Speed`, `Data.Wind.Direction` (numeric, standardized) and
  `Station.State` (categorical, one-hot encoded).
- **Targets:** `Data.Temperature.Avg/Max/Min Temp` (continuous).

Numeric features are standardized and the state is one-hot encoded via a
`ColumnTransformer`; targets are standardized for training and inverse-transformed
back to °F for metrics and predictions. The fitted preprocessing is saved inside
the model checkpoint so inference reproduces it exactly.

## Train

```bash
# Defaults: net [64, 32], 60 epochs, predicts Avg/Max/Min temp
python train.py

# Predict a single metric with a larger network, trained longer
python train.py --targets "Data.Temperature.Avg Temp" \
  --hidden-sizes 128 64 32 --epochs 80 --learning-rate 5e-4
```

Training logs params and per-epoch metrics (loss, plus MAE / RMSE / R² per
target) to Domino's MLflow (experiment `weather-pytorch-<project>`) **and**
writes the model binary to
`$DOMINO_ARTIFACTS_DIR/weather_model/weather_model.pt` (override with
`--output-dir`), with a `metadata.json` alongside.

Run `python train.py --help` for all options.

## Predict

Loads the on-disk binary and **uses a GPU when available, otherwise CPU**.

```bash
# Demo: scores two built-in samples
python predict.py

# Single sample (named flags)
python predict.py --month 7 --week-of 28 --state Alabama \
  --precipitation 0.1 --wind-speed 5.0 --wind-direction 20

# Batch from a CSV (must contain the feature columns), write results
python predict.py --input new_weather.csv --output predictions.csv
```

## Host as a Domino Model API (manual)

Publish a Model API pointed at:

- **File:** `model_api.py`
- **Function:** `predict`

Request body:

```json
{"month": 7, "week_of": 28, "state": "Alabama", "precipitation": 0.1, "wind_speed": 5.0, "wind_direction": 20}
```

Response maps each trained target to its predicted °F value. As with the
diabetes example, ensure the `.pt` binary is reachable at `DEFAULT_MODEL_PATH`
in the endpoint's environment (commit it to the repo or mount a dataset, since
`/mnt/artifacts` is per-run).

## Setup

```bash
# CPU-only torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy scikit-learn mlflow

# For a GPU hardware tier, install the default CUDA build instead:
pip install torch
```
