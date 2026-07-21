# Module 06 — Serving Predictions with FastAPI

[← Regression Model](05-regression-model.md) · [Index](README.md) · [Next: Volumes & GPU →](07-docker-volumes-gpu.md)

## Learning objectives

- Build a REST API with **FastAPI**.
- Validate request bodies automatically with **Pydantic**.
- Load the model **once** at startup using a **lifespan** handler.
- Return proper HTTP status codes (e.g. 503 when the model isn't trained yet).
- Test the API with FastAPI's **TestClient**.

## Prerequisites

- Module 05 complete; a trained model exists in `models/`.

---

## 1. Why FastAPI?

FastAPI turns Python functions into HTTP endpoints with:

- **Automatic validation** from type hints (via Pydantic) — malformed requests
  are rejected with a helpful 422 before your code runs.
- **Auto-generated interactive docs** at `/docs` (Swagger UI).
- **High performance** (async, built on Starlette/uvicorn).

---

## 2. `app.py` walkthrough

### Load the model once, at startup (lifespan)

Loading the model on *every* request would be painfully slow. A **lifespan**
handler runs setup code once when the server boots and teardown when it stops:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from model import load_artifacts

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
_state = {"artifacts": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["artifacts"] = load_artifacts(MODEL_DIR)
        print(f"Loaded model from {MODEL_DIR}")
    except FileNotFoundError:
        _state["artifacts"] = None          # server still starts; /predict 503s
        print(f"No model in {MODEL_DIR}; train it first.")
    yield                                    # <-- app runs here

app = FastAPI(title="California Housing Regression API", lifespan=lifespan)
```

> **Design choice:** if no model is present, we still let the server start (so
> `/health` works and ops tooling can probe it) but make `/predict` return a
> clear error. Failing to boot would be worse for observability.

Reading `MODEL_DIR` from an **environment variable** (not a hardcoded path) is
what lets Docker and the tests point the app at different directories.

### Validate input with a Pydantic model

Each field becomes a validated, documented part of the request body:

```python
from pydantic import BaseModel, Field

class HousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income (10k USD)")
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
```

If a client omits `MedInc` or sends a string, FastAPI responds **422
Unprocessable Entity** automatically — you write zero validation code.

### The endpoints

```python
from fastapi import HTTPException
import numpy as np

@app.get("/health")
def health():
    a = _state["artifacts"]
    return {"status": "ok", "model_loaded": a is not None,
            "device": str(a.device) if a else None}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures):
    artifacts = _state["artifacts"]
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    # Build the row in the exact feature order the model expects:
    row = np.array([[getattr(features, name) for name in artifacts.feature_names]],
                   dtype=np.float32)
    value = float(artifacts.predict(row)[0])
    return PredictionResponse(prediction=value)
```

> **Subtle but important:** we assemble the input row by iterating
> `artifacts.feature_names` (saved in `meta.json`), *not* by hardcoding order.
> This guarantees the JSON fields line up with the columns the model trained on,
> even if someone reorders the Pydantic class.

- `/health` → liveness + whether a model is loaded. Standard for
  load-balancers and orchestration.
- `/predict` → the actual inference. `503` when no model; `200` with a number
  otherwise.

---

## 3. Run the API

```bash
docker compose up
```

Then, in another terminal:

```bash
curl -s http://localhost:8000/health
# {"status":"ok","model_loaded":true,"device":"cpu"}

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc":8.3252,"HouseAge":41,"AveRooms":6.9841,"AveBedrms":1.0238,"Population":322,"AveOccup":2.5556,"Latitude":37.88,"Longitude":-122.23}'
# {"prediction":4.28...}
```

That input is the dataset's first district (true value ≈ 4.53, in units of
$100k). A prediction of ~4.28 is a reasonable estimate.

**Explore interactively:** open <http://localhost:8000/docs> for Swagger UI —
you can fire requests from the browser, and the example values come straight
from the Pydantic schema.

---

## 4. Test the API with `TestClient`

FastAPI ships a `TestClient` that exercises the app in-process — no server, no
network. Our tests build a throwaway model in a `tmp_path` and point the app at
it via `MODEL_DIR`:

```python
import importlib, pytest
torch = pytest.importorskip("torch")       # skip if torch absent (local 3.14)
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
import model as model_module

def make_client(monkeypatch, model_dir):
    monkeypatch.setenv("MODEL_DIR", model_dir)
    import app as app_module
    importlib.reload(app_module)           # re-read MODEL_DIR
    return TestClient(app_module.app), app_module

def test_predict_returns_a_number(monkeypatch, trained_model_dir):
    client, _ = make_client(monkeypatch, trained_model_dir)
    with client:                            # `with` triggers lifespan (model load)
        resp = client.post("/predict", json=EXAMPLE)
    assert resp.status_code == 200
    assert isinstance(resp.json()["prediction"], float)
```

Techniques on display:

- `pytest.importorskip(...)` — cleanly **skip** these tests where torch/fastapi
  aren't installed (your local 3.14 venv), while still running them in Docker.
- `monkeypatch.setenv` + `importlib.reload` — make the app pick up a test-only
  `MODEL_DIR`.
- `with client:` — entering the context manager fires the **lifespan**, so the
  model actually loads before the request.
- We also assert the **503 path** (no model) and the **422 path** (missing
  field) — testing failure modes, not just the happy path.

Run them in the container:

```bash
docker compose run --rm api pytest -v
# ...
# 54 passed in 2.4s
```

(36 preprocessing + 14 model (Module 05) + 4 API tests = **54 total**.)

---

## Checkpoint ✅

- [ ] `app.py` exposes `/health` and `/predict`.
- [ ] `docker compose up` serves the API; `curl` returns a prediction.
- [ ] `/docs` shows interactive documentation.
- [ ] `tests/test_api.py` passes in the container (54 tests total).

## Exercises

1. Add a `POST /predict/batch` endpoint that accepts a list of feature objects
   and returns a list of predictions.
2. Add response timing: log how long each prediction takes.
3. Return the model's device and a request id in the `/predict` response for
   observability.

[← Regression Model](05-regression-model.md) · [Index](README.md) · [Next: Volumes & GPU →](07-docker-volumes-gpu.md)
