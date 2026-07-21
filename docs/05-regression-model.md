# Module 05 — The Regression Model (PyTorch)

[← Docker](04-docker.md) · [Index](README.md) · [Next: Serving with FastAPI →](06-fastapi-inference.md)

## Learning objectives

- Load a **public dataset** (California Housing) via scikit-learn.
- Build a small **PyTorch** multi-layer perceptron (MLP) for regression.
- Train it with a standard loop (batching, loss, optimizer).
- Properly **scale** features and targets using *training* statistics only.
- **Persist** the model and everything needed to reproduce inference.
- Write code that runs on **CPU or GPU** automatically.

## Prerequisites

- Module 04 (we run training inside Docker, because torch has no Python 3.14
  wheel — the container uses 3.12).

Update `requirements.txt`:

```
numpy
pandas
torch
scikit-learn
joblib
fastapi
uvicorn[standard]
pytest
httpx
```

---

## 1. The dataset

**California Housing** is a classic public regression benchmark: given 8
features about a district (median income, house age, average rooms, location,
etc.), predict the median house value. scikit-learn downloads and caches it:

```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target       # 20,640 rows × 8 features
```

---

## 2. Separating concerns: `model.py` vs `train.py`

We split model code from training so that **training and serving build the exact
same architecture and load artifacts identically**. `app.py` (Module 06) imports
from `model.py` too — no duplicated network definition that could drift.

### `model.py` — architecture + artifact helpers

**The network** — a small MLP. `squeeze(-1)` turns the `(batch, 1)` output into
`(batch,)` to match the target shape:

```python
import torch.nn as nn

class RegressionMLP(nn.Module):
    def __init__(self, in_features, hidden=(64, 32)):
        super().__init__()
        layers = []
        prev = in_features
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
```

**Device selection** — this one function is why the same code runs on CPU and
GPU (Module 07):

```python
import torch

def select_device():
    """CUDA when available (GPU passthrough), otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Saving artifacts** — a model alone is not enough to predict. You must persist
*how the data was scaled*, or inference will feed the model differently-scaled
numbers than it trained on. We save three things:

```python
import json, os
import numpy as np
import torch

def save_artifacts(model_dir, model, x_mean, x_std, y_mean, y_std, feature_names):
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    np.savez(os.path.join(model_dir, "scaler.npz"),
             x_mean=x_mean, x_std=x_std,
             y_mean=np.array([y_mean]), y_std=np.array([y_std]))
    with open(os.path.join(model_dir, "meta.json"), "w") as fh:
        json.dump({"feature_names": feature_names}, fh, indent=2)
```

| Artifact | Contains | Why it's needed |
| -------- | -------- | --------------- |
| `model.pt` | network weights (`state_dict`) | the learned parameters |
| `scaler.npz` | feature/target mean & std | to scale inputs and un-scale outputs identically to training |
| `meta.json` | ordered feature names | so the API knows which JSON field maps to which input column |

**Loading** reverses it, wrapping everything in an `Artifacts` object with a
`.predict()` that scales inputs, runs the model, and un-scales the output. See
the full `model.py` in the repo.

---

## 3. `train.py` — the training script

The full script is in the repo; here are the parts worth teaching.

### Scale using TRAIN statistics only

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
x_std[x_std == 0] = 1.0                       # guard constant columns
y_mean, y_std = float(y_train.mean()), float(y_train.std()) or 1.0
```

> **Critical ML principle — no data leakage.** We compute scaling from the
> *training* set only, then apply it to test data. Computing statistics over the
> whole dataset would leak information about the test set into training and
> inflate your scores.

### The training loop

```python
model = RegressionMLP(in_features=X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(n, device=device)          # shuffle each epoch
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        optimizer.zero_grad()                          # reset gradients
        pred = model(Xtr[idx])
        loss = loss_fn(pred, ytr[idx])
        loss.backward()                                # compute gradients
        optimizer.step()                               # update weights
```

The five-line inner loop is the heart of virtually all PyTorch training:
**zero grads → forward → loss → backward → step.** Worth pausing on in a
workshop.

### Evaluate in real units

We report **RMSE in the original target units** (not the scaled space) so the
number is interpretable:

```python
test_pred = model(Xte) * y_std + y_mean     # un-scale predictions
rmse = torch.sqrt(torch.mean((test_pred - test_true) ** 2))
```

---

## 4. Run training (in Docker)

```bash
docker compose build
docker compose run --rm api python train.py --epochs 30
```

**Expected output:**

```
Using device: cpu
Loaded California Housing: 20640 rows, 8 features
epoch   1 | test RMSE = 0.7748
epoch  10 | test RMSE = 0.5999
epoch  20 | test RMSE = 0.5582
epoch  30 | test RMSE = 0.5449
Saved model artifacts to /app/models/
```

The falling RMSE means the model is learning. Because the project is
volume-mounted (Module 07), the `models/` directory now exists **on your host**:

```bash
ls models/
# meta.json  model.pt  scaler.npz
```

> `models/` is a build artifact, so it's git-ignored — never commit trained
> weights to the source repo.

---

## 5. Unit-testing the model

The model layer is pure logic (given weights and scalers, produce a number), so
it's very testable — no dataset download or server required. We add
`tests/test_model.py` with **14 tests** covering four concerns. Like the API
tests, they start with `pytest.importorskip("torch")` so they run in Docker but
*skip* cleanly on a torch-less local 3.14 venv.

### a) Architecture

```python
def test_forward_output_is_1d_of_batch_size():
    net = RegressionMLP(in_features=8)
    out = net(torch.zeros(5, 8))
    assert out.shape == (5,)          # forward() squeezes (5, 1) -> (5,)

def test_hidden_layer_shapes_match_config():
    net = RegressionMLP(in_features=4, hidden=(16, 8))
    linears = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(4, 16), (16, 8), (8, 1)]
```

### b) Device selection — testing the GPU path *without* a GPU

You can't rely on CI/laptops having CUDA, so we **mock** `torch.cuda.is_available`
to exercise both branches deterministically:

```python
def test_select_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: False)
    assert select_device().type == "cpu"

def test_select_device_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: True)
    assert select_device().type == "cuda"
```

> This is the payoff of the mocking skills from Module 02: the CUDA code path is
> now tested on a CPU-only machine.

### c) The scaling math (the part most likely to hide bugs)

`Artifacts.predict` scales inputs, runs the model, and un-scales the output. We
verify each half with tiny **stub models** whose behavior we control:

```python
def test_predict_applies_target_unscaling():
    class ZeroModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0])
    art = _artifacts(ZeroModel(), np.zeros(8), np.ones(8), y_mean=3.5, y_std=2.0)
    # model outputs 0 -> prediction must equal y_mean exactly
    assert np.allclose(art.predict(np.random.randn(6, 8).astype("float32")), 3.5)
```

A companion test uses a `FirstFeatureModel` (returns `x[:, 0]`) to confirm the
`(x - mean) / std` feature scaling is applied *before* the model sees the data.

### d) Artifact save/load round-trip

The most important integration guarantee: **what you save is what you load**, and
predictions are byte-for-byte reproducible.

```python
def test_roundtrip_predictions_are_identical(tmp_path):
    net = RegressionMLP(len(FEATURES))
    save_artifacts(str(tmp_path), net, np.zeros(8), np.ones(8), 2.0, 1.0, FEATURES)
    loaded = load_artifacts(str(tmp_path), device=CPU)
    rows = np.random.RandomState(2).randn(3, 8).astype("float32")
    # ... compute expected from `net` directly ...
    assert np.allclose(loaded.predict(rows), expected, atol=1e-5)

def test_load_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifacts(str(tmp_path / "does-not-exist"))
```

### e) A learning sanity check (no dataset needed)

Finally, one test proves the *training recipe itself* works — on a synthetic
linear task, the loop must drive loss well below its starting value:

```python
def test_model_learns_a_linear_relationship():
    torch.manual_seed(0)
    X = torch.randn(512, 3)
    y = X @ torch.tensor([2.0, -1.0, 0.5]) + 0.1
    net = RegressionMLP(3, hidden=(16,))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    first = None
    for _ in range(200):
        opt.zero_grad(); loss = loss_fn(net(X), y)
        first = first or loss.item()
        loss.backward(); opt.step()
    assert loss.item() < first * 0.1     # converged
```

Run them in the container:

```bash
docker compose run --rm api pytest tests/test_model.py -v
# ...
# 14 passed
```

The full suite is now **50 tests** (36 preprocessing + 14 model); the API tests
in Module 06 bring it to 54.

---

## Checkpoint ✅

- [ ] `model.py` defines `RegressionMLP`, `select_device`, and save/load helpers.
- [ ] `train.py` trains and writes `models/`.
- [ ] Training RMSE decreases across epochs.
- [ ] `models/{model.pt,scaler.npz,meta.json}` exist on the host.
- [ ] `tests/test_model.py` passes (14 tests) in the container.

## Exercises

1. Change `hidden=(64, 32)` to `(128, 64, 32)` and see if RMSE improves.
2. Add early stopping: stop training when test RMSE hasn't improved for N epochs.
3. Swap the dataset for scikit-learn's `fetch_openml("diabetes")` and adjust the
   feature list. What else has to change? (Hint: `meta.json` and the API schema.)

[← Docker](04-docker.md) · [Index](README.md) · [Next: Serving with FastAPI →](06-fastapi-inference.md)
