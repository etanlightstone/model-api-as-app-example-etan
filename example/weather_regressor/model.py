"""Neural network architecture for weather (temperature) regression.

A small, configurable feed-forward (MLP) regression model. Unlike the diabetes
*classifier* (which outputs class logits), this is a *regressor*: the final
layer is linear and produces one continuous value per target metric. It supports
multi-output regression, so it can predict one or more metrics at once
(by default: average, max and min temperature).

The set of input features is split into numeric columns (standardized) and
categorical columns (one-hot encoded) at training time; the fitted preprocessing
is saved in the model checkpoint so inference reproduces it exactly.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# Raw weather feature columns used as model inputs.
NUMERIC_FEATURES: List[str] = [
    "Date.Month",
    "Date.Week of",
    "Data.Precipitation",
    "Data.Wind.Speed",
    "Data.Wind.Direction",
]
CATEGORICAL_FEATURES: List[str] = [
    "Station.State",
]

# Continuous metrics to predict. Multi-output regression by default; you can
# train on a single target (e.g. just Avg Temp) via the training script.
TARGET_COLUMNS: List[str] = [
    "Data.Temperature.Avg Temp",
    "Data.Temperature.Max Temp",
    "Data.Temperature.Min Temp",
]


class WeatherNet(nn.Module):
    """Configurable MLP for (multi-output) regression.

    Args:
        input_dim: Number of input features *after* preprocessing
            (numeric + one-hot expanded categoricals).
        output_dim: Number of continuous targets to predict.
        hidden_sizes: List of hidden layer widths. The number of entries
            controls the depth; each value controls the width of that layer.
        dropout: Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = len(TARGET_COLUMNS),
        hidden_sizes: List[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        # Persist the config so an identical architecture can be rebuilt at
        # inference time from the saved checkpoint.
        self.config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_sizes": list(hidden_sizes),
            "dropout": dropout,
        }

        layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden_sizes:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        # Linear output head -- no activation -- one unit per target metric.
        layers.append(nn.Linear(prev, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @classmethod
    def from_config(cls, config: dict) -> "WeatherNet":
        """Rebuild a model from a saved config dict."""
        return cls(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            hidden_sizes=config["hidden_sizes"],
            dropout=config.get("dropout", 0.0),
        )
