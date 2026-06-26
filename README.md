# Model API as App — Examples

Two small, self-contained examples that train a model with Domino's built-in
MLflow tracking and then serve it as a **Domino Model API** — shown two ways:
as **custom scoring code** and as a **registry model that Domino auto-wraps**.

The samples are deliberately parallel: same project layout, same two deployment
paths, different framework and ML task. Pick whichever is closer to your use
case, or read both to compare PyTorch vs. scikit-learn.

## The samples

| | [`diabetes_classer`](example/diabetes_classer/) | [`weather_regressor`](example/weather_regressor/) |
| --- | --- | --- |
| Framework | PyTorch | scikit-learn |
| Task | Binary classification (is diabetic?) | Multi-output regression (avg/max/min temp) |
| Model | `DiabetesNet` MLP + `StandardScaler` | `Pipeline`: preprocessing → `HistGradientBoostingRegressor` |
| Output | `is_diabetic`, `probability` | `avg_temp`, `max_temp`, `min_temp` |
| Data | `diabetes_dataset.csv` | `weather.csv` |

Each sample's `README.md` has the full details, data schema, and commands.

## Shared layout

Every sample uses the same set of files:

| File | Purpose |
| --- | --- |
| `model.py` | Model / pipeline definition + feature & target columns. |
| `train.py` | Trains, logs params/metrics to MLflow, writes the model to disk, **logs a signed pyfunc** to the registry. |
| `predict.py` | Local CLI inference from the on-disk model. |
| `pyfunc_model.py` | `mlflow.pyfunc` wrapper used for the **registry** deployment. |
| `model_api.py` | Entrypoint (`predict`) for the **custom-code** deployment. |
| `requirements.txt` | Dependencies (includes `uwsgi`, required to serve Model APIs). |

## Two ways to deploy as a Model API

Both paths are implemented in each sample.

**A — Custom code (`model_api.py` → `predict`).** You point Domino at a file
and function and write the scoring glue. The function loads the model binary
from `$DOMINO_ARTIFACTS_DIR` and returns a JSON-friendly dict.

**B — From the registry (no scoring code).** `train.py` logs a `pyfunc`
**with a signature** (inferred from a named-feature example). Because the
registered model carries its input/output schema (and its own
`requirements.txt`), Domino reads the schema and **auto-wraps** it as an
endpoint. Register the logged model named `model` and deploy it.

> When registering, choose the logged model named **`model`** — it is the
> signed pyfunc that carries the schema and dependencies. A bare artifact
> folder has neither and cannot be auto-deployed.

## Typical workflow

```bash
cd example/<sample>
python train.py        # train + track + log the registry model
python predict.py      # sanity-check predictions locally
python model_api.py    # smoke-test the custom-code scoring function
```

Then publish via **Publish → Model APIs** (custom code) or register the logged
model and deploy from the **Model Registry** (auto-wrapped). See each sample's
README for the exact request/response shapes.
