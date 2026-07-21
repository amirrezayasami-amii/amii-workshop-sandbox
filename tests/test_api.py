"""Tests for the FastAPI inference service.

These need torch + fastapi installed, so they run inside the Docker image
(Python 3.12), not the local 3.14 venv. A tiny random-weight model is saved
to a temp dir and pointed at via MODEL_DIR, so no real training is required.
"""

import importlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import model as model_module  # noqa: E402

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

EXAMPLE = {
    "MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.9841, "AveBedrms": 1.0238,
    "Population": 322.0, "AveOccup": 2.5556, "Latitude": 37.88, "Longitude": -122.23,
}


@pytest.fixture
def trained_model_dir(tmp_path):
    """Save a small untrained model + identity-ish scalers to a temp dir."""
    net = model_module.RegressionMLP(in_features=len(FEATURES))
    model_module.save_artifacts(
        model_dir=str(tmp_path),
        model=net,
        x_mean=np.zeros(len(FEATURES), dtype=np.float32),
        x_std=np.ones(len(FEATURES), dtype=np.float32),
        y_mean=2.0,
        y_std=1.0,
        feature_names=FEATURES,
    )
    return str(tmp_path)


def make_client(monkeypatch, model_dir):
    """Reload the app so it picks up MODEL_DIR, and return a TestClient."""
    monkeypatch.setenv("MODEL_DIR", model_dir)
    import app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.app), app_module


def test_health_reports_model_loaded(monkeypatch, trained_model_dir):
    client, _ = make_client(monkeypatch, trained_model_dir)
    with client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_a_number(monkeypatch, trained_model_dir):
    client, _ = make_client(monkeypatch, trained_model_dir)
    with client:
        resp = client.post("/predict", json=EXAMPLE)
    assert resp.status_code == 200
    assert isinstance(resp.json()["prediction"], float)


def test_predict_rejects_missing_field(monkeypatch, trained_model_dir):
    client, _ = make_client(monkeypatch, trained_model_dir)
    bad = {k: v for k, v in EXAMPLE.items() if k != "MedInc"}
    with client:
        resp = client.post("/predict", json=bad)
    assert resp.status_code == 422  # pydantic validation error


def test_predict_without_model_returns_503(monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    client, _ = make_client(monkeypatch, str(empty))
    with client:
        resp = client.post("/predict", json=EXAMPLE)
    assert resp.status_code == 503
