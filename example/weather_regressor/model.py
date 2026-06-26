"""Model definition for the weather temperature regressor (scikit-learn).

This example mirrors the diabetes classifier but uses **scikit-learn** instead
of PyTorch and solves a **multi-output regression** problem: predict the
average, max and min temperature from a handful of date / location / weather
features.

Everything inference needs lives in a single scikit-learn ``Pipeline``:

    ColumnTransformer( StandardScaler(numeric) + OneHotEncoder(categorical) )
      -> MultiOutputRegressor( HistGradientBoostingRegressor )

Gradient-boosted trees give strong accuracy on this tabular data while keeping
the serialized model tiny (~3 MB), which matters when it gets bundled into the
MLflow model and deployed as an endpoint. ``MultiOutputRegressor`` fits one
booster per temperature target.

Because the preprocessing is *inside* the pipeline, the fitted estimator
accepts raw feature values directly -- no separate scaler to ship around.

The raw dataset columns have awkward names (``Date.Month``,
``Data.Temperature.Avg Temp`` ...). We rename them to clean snake_case so both
the custom-code Model API and the registry-deployed model expose a friendly
JSON schema.
"""

from __future__ import annotations

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---- Friendly (snake_case) feature / target names used everywhere ----
NUMERIC_FEATURES: List[str] = [
    "month",
    "week_of",
    "precipitation",
    "wind_speed",
    "wind_direction",
]
CATEGORICAL_FEATURES: List[str] = ["state"]
FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMNS: List[str] = ["avg_temp", "max_temp", "min_temp"]

# ---- Mapping from the raw CSV column names to the friendly names ----
RAW_TO_FEATURE = {
    "Date.Month": "month",
    "Date.Week of": "week_of",
    "Data.Precipitation": "precipitation",
    "Data.Wind.Speed": "wind_speed",
    "Data.Wind.Direction": "wind_direction",
    "Station.State": "state",
}
RAW_TO_TARGET = {
    "Data.Temperature.Avg Temp": "avg_temp",
    "Data.Temperature.Max Temp": "max_temp",
    "Data.Temperature.Min Temp": "min_temp",
}


def build_pipeline(
    max_iter: int = 400,
    learning_rate: float = 0.08,
    max_depth: int | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build the (unfitted) preprocessing + regression pipeline.

    Numeric features are standardized and the single categorical feature
    (``state``) is one-hot encoded; unknown categories seen at inference time
    are ignored rather than erroring. ``HistGradientBoostingRegressor`` is a
    single-output learner, so it is wrapped in ``MultiOutputRegressor`` to
    predict all temperature targets (one booster per target).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ]
    )

    booster = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("regressor", MultiOutputRegressor(booster)),
    ])
