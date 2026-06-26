# What if: R-backed models

## Short answer: not supported today

The app has two model-loading paths, both Python-only:

1. **Custom-function path** (`CustomFunctionAdapter`, `adapter.py:148`) — uses `importlib` + `inspect.signature` to import a `.py` file and call a named function. R cannot use this path at all.
2. **Registry path** (`RegistryAdapter`, `adapter.py:313`) — calls `mlflow.pyfunc.load_model(uri)`, which requires a `python_function` flavor entry in the `MLmodel` file. R models logged via `mlflow::mlflow_save_model()` in R don't have one, so loading fails (visible as a load error in Settings, not a silent passthrough).

The README's claim that "any MLflow flavor" works applies only to Python-backed MLflow flavors (sklearn, XGBoost, TensorFlow, CatBoost, etc.). The R `r_function`/`crate` flavor is not Python-backed and fails at load time.

The rest of the stack (`predict_service.py`, `schema.py`) wraps inputs and outputs in pandas/numpy — an R bridge would need to speak those types too.

---

## Workarounds (outside this app)

### Option A — Python pyfunc wrapper via `rpy2`

Write a `mlflow.pyfunc.PythonModel` subclass in Python that calls into R via `rpy2`, log it with a `python_function` flavor, and register it. The registry path loads it like any other Python model.

Requires:
- R + `rpy2` installed in the app's Compute Environment
- A Python wrapper module per R model

### Option B — Python shim with the custom-function path

Write a `model_api.py:predict()` function in Python that shells out to `Rscript` (via `subprocess`) and returns a plain dict. Point Settings at that file.

Requires:
- R installed in the app's Compute Environment
- A Python shim per R model

---

## Summary

Neither workaround is a one-liner and both require changes outside this app (environment setup + per-model glue code). There is no path to drop in an R model and have it just work.
