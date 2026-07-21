"""Model definition and shared artifact helpers.

Kept separate from training and serving so both ``train.py`` and ``app.py``
build the exact same architecture and load artifacts the same way.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

MODEL_WEIGHTS = "model.pt"
SCALER_FILE = "scaler.npz"
META_FILE = "meta.json"


class RegressionMLP(nn.Module):
    """A small multi-layer perceptron for tabular regression."""

    def __init__(self, in_features: int, hidden=(64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class Artifacts:
    """Everything needed to run inference, loaded from a model directory."""

    model: RegressionMLP
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    feature_names: list[str]
    device: torch.device

    def predict(self, rows: np.ndarray) -> np.ndarray:
        """Predict targets for a 2D array of raw (unscaled) feature rows."""
        x = (rows - self.x_mean) / self.x_std
        tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            scaled = self.model(tensor).cpu().numpy()
        return scaled * self.y_std + self.y_mean


def select_device() -> torch.device:
    """Return CUDA when available (GPU passthrough), otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_artifacts(
    model_dir: str,
    model: RegressionMLP,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: float,
    y_std: float,
    feature_names: list[str],
) -> None:
    """Persist weights, feature scaling, and metadata to ``model_dir``."""
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, MODEL_WEIGHTS))
    np.savez(
        os.path.join(model_dir, SCALER_FILE),
        x_mean=x_mean,
        x_std=x_std,
        y_mean=np.array([y_mean]),
        y_std=np.array([y_std]),
    )
    with open(os.path.join(model_dir, META_FILE), "w") as fh:
        json.dump({"feature_names": feature_names}, fh, indent=2)


def load_artifacts(model_dir: str, device: torch.device | None = None) -> Artifacts:
    """Load a trained model and its scaling parameters from ``model_dir``."""
    device = device or select_device()
    with open(os.path.join(model_dir, META_FILE)) as fh:
        feature_names = json.load(fh)["feature_names"]

    scaler = np.load(os.path.join(model_dir, SCALER_FILE))
    model = RegressionMLP(in_features=len(feature_names))
    state = torch.load(
        os.path.join(model_dir, MODEL_WEIGHTS), map_location=device
    )
    model.load_state_dict(state)
    model.to(device)

    return Artifacts(
        model=model,
        x_mean=scaler["x_mean"],
        x_std=scaler["x_std"],
        y_mean=float(scaler["y_mean"][0]),
        y_std=float(scaler["y_std"][0]),
        feature_names=feature_names,
        device=device,
    )
