"""Neural network architecture for diabetes classification.

A small, configurable feed-forward (MLP) binary classifier. The defaults give
a compact model with two hidden layers; the shape can be varied at construction
time so the same architecture can be scaled up or down from the training script.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# The diabetes dataset feature columns, in the order the model expects them.
FEATURE_COLUMNS: List[str] = [
    "calories_wk",
    "hrs_exercise_wk",
    "exercise_intensity",
    "annual_income",
    "num_children",
    "weight",
]
TARGET_COLUMN: str = "is_diabetic"


class DiabetesNet(nn.Module):
    """Configurable MLP for binary diabetes classification.

    Args:
        input_dim: Number of input features.
        hidden_sizes: List of hidden layer widths. The number of entries
            controls the depth; each value controls the width of that layer.
        dropout: Dropout probability applied after each hidden activation.
        num_classes: Number of output logits (2 for binary classification).
    """

    def __init__(
        self,
        input_dim: int = len(FEATURE_COLUMNS),
        hidden_sizes: List[int] | None = None,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [32, 16]

        # Persist the config so it can be saved alongside the weights and used
        # to rebuild an identical architecture at inference time.
        self.config = {
            "input_dim": input_dim,
            "hidden_sizes": list(hidden_sizes),
            "dropout": dropout,
            "num_classes": num_classes,
        }

        layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden_sizes:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @classmethod
    def from_config(cls, config: dict) -> "DiabetesNet":
        """Rebuild a model from a saved config dict."""
        return cls(
            input_dim=config["input_dim"],
            hidden_sizes=config["hidden_sizes"],
            dropout=config.get("dropout", 0.0),
            num_classes=config.get("num_classes", 2),
        )
