"""FastAPI inference service for the California Housing regression model.

Run locally:
    uvicorn app:app --reload

The model directory is read from the MODEL_DIR env var (default: ``models``).
Train the model first with ``python train.py`` so the artifacts exist.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import Artifacts, load_artifacts

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

# Populated on startup; stays None if no trained artifacts are present yet.
_state: dict[str, Artifacts | None] = {"artifacts": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["artifacts"] = load_artifacts(MODEL_DIR)
        print(f"Loaded model from {MODEL_DIR} on {_state['artifacts'].device}")
    except FileNotFoundError:
        _state["artifacts"] = None
        print(f"No model found in {MODEL_DIR}; train it with `python train.py`.")
    yield


app = FastAPI(title="California Housing Regression API", lifespan=lifespan)


class HousingFeatures(BaseModel):
    """One California district's features (units per the sklearn dataset)."""

    MedInc: float = Field(..., description="Median income (10k USD)")
    HouseAge: float = Field(..., description="Median house age (years)")
    AveRooms: float = Field(..., description="Average rooms per household")
    AveBedrms: float = Field(..., description="Average bedrooms per household")
    Population: float = Field(..., description="District population")
    AveOccup: float = Field(..., description="Average household occupancy")
    Latitude: float = Field(..., description="District latitude")
    Longitude: float = Field(..., description="District longitude")

    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.9841,
                "AveBedrms": 1.0238,
                "Population": 322.0,
                "AveOccup": 2.5556,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted median house value (100k USD)")


def _require_model() -> Artifacts:
    artifacts = _state["artifacts"]
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train it first with `python train.py`.",
        )
    return artifacts


@app.get("/health")
def health() -> dict:
    artifacts = _state["artifacts"]
    return {
        "status": "ok",
        "model_loaded": artifacts is not None,
        "device": str(artifacts.device) if artifacts else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures) -> PredictionResponse:
    artifacts = _require_model()
    row = np.array(
        [[getattr(features, name) for name in artifacts.feature_names]],
        dtype=np.float32,
    )
    value = float(artifacts.predict(row)[0])
    return PredictionResponse(prediction=value)
