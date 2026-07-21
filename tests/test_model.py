"""Unit tests for the ML model layer (`model.py`).

Covers the network architecture, device selection, the inference-time
scaling/un-scaling math, artifact save/load round-trips, and a small
learning-convergence check. Like the API tests these require torch, so they
run in Docker (Python 3.12), not the local 3.14 venv.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import model as model_module  # noqa: E402
from model import (  # noqa: E402
    Artifacts,
    RegressionMLP,
    load_artifacts,
    save_artifacts,
    select_device,
)

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
CPU = torch.device("cpu")


# --------------------------------------------------------------------------
# RegressionMLP architecture
# --------------------------------------------------------------------------

def test_forward_output_is_1d_of_batch_size():
    net = RegressionMLP(in_features=8)
    out = net(torch.zeros(5, 8))
    # forward() squeezes the trailing dim: (5, 1) -> (5,)
    assert out.shape == (5,)


def test_hidden_layer_shapes_match_config():
    net = RegressionMLP(in_features=4, hidden=(16, 8))
    linears = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    shapes = [(m.in_features, m.out_features) for m in linears]
    assert shapes == [(4, 16), (16, 8), (8, 1)]


def test_default_hidden_produces_expected_depth():
    net = RegressionMLP(in_features=8)  # default hidden=(64, 32)
    linears = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [
        (8, 64), (64, 32), (32, 1)
    ]


def test_same_seed_gives_identical_weights():
    torch.manual_seed(0)
    a = RegressionMLP(8)
    torch.manual_seed(0)
    b = RegressionMLP(8)
    x = torch.randn(3, 8)
    assert torch.allclose(a(x), b(x))


# --------------------------------------------------------------------------
# select_device
# --------------------------------------------------------------------------

def test_select_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: False)
    assert select_device().type == "cpu"


def test_select_device_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: True)
    assert select_device().type == "cuda"


# --------------------------------------------------------------------------
# Artifacts.predict — scaling / un-scaling math
# --------------------------------------------------------------------------

def _artifacts(net, x_mean, x_std, y_mean, y_std):
    return Artifacts(
        model=net,
        x_mean=np.asarray(x_mean, dtype=np.float32),
        x_std=np.asarray(x_std, dtype=np.float32),
        y_mean=float(y_mean),
        y_std=float(y_std),
        feature_names=FEATURES,
        device=CPU,
    )


def test_predict_with_identity_scalers_matches_raw_model():
    net = RegressionMLP(8)
    art = _artifacts(net, np.zeros(8), np.ones(8), 0.0, 1.0)
    rows = np.random.RandomState(0).randn(4, 8).astype(np.float32)

    net.eval()
    with torch.no_grad():
        expected = net(torch.tensor(rows)).numpy()

    assert np.allclose(art.predict(rows), expected, atol=1e-5)


def test_predict_applies_target_unscaling():
    """A model that always outputs 0 must yield exactly y_mean after un-scaling."""

    class ZeroModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0])

    art = _artifacts(ZeroModel(), np.zeros(8), np.ones(8), y_mean=3.5, y_std=2.0)
    out = art.predict(np.random.randn(6, 8).astype(np.float32))
    assert np.allclose(out, 3.5)


def test_predict_applies_feature_scaling():
    """Model returns feature 0; verify (x - mean)/std is applied before it."""

    class FirstFeatureModel(torch.nn.Module):
        def forward(self, x):
            return x[:, 0]

    x_mean = np.arange(8, dtype=np.float32)
    x_std = np.full(8, 2.0, dtype=np.float32)
    art = _artifacts(FirstFeatureModel(), x_mean, x_std, y_mean=1.0, y_std=10.0)

    rows = np.random.RandomState(1).randn(5, 8).astype(np.float32)
    scaled_feat0 = (rows[:, 0] - x_mean[0]) / x_std[0]
    expected = scaled_feat0 * 10.0 + 1.0

    assert np.allclose(art.predict(rows), expected, atol=1e-5)


# --------------------------------------------------------------------------
# save_artifacts / load_artifacts round-trip
# --------------------------------------------------------------------------

def test_save_writes_all_three_files(tmp_path):
    net = RegressionMLP(len(FEATURES))
    save_artifacts(
        str(tmp_path), net, np.zeros(8), np.ones(8), 0.0, 1.0, FEATURES
    )
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "scaler.npz").exists()
    assert (tmp_path / "meta.json").exists()


def test_load_restores_scaling_and_metadata(tmp_path):
    net = RegressionMLP(len(FEATURES))
    x_mean = np.arange(8, dtype=np.float32)
    x_std = np.full(8, 2.0, dtype=np.float32)
    save_artifacts(str(tmp_path), net, x_mean, x_std, 1.5, 0.7, FEATURES)

    loaded = load_artifacts(str(tmp_path), device=CPU)

    assert loaded.feature_names == FEATURES
    assert np.allclose(loaded.x_mean, x_mean)
    assert np.allclose(loaded.x_std, x_std)
    assert loaded.y_mean == pytest.approx(1.5)
    assert loaded.y_std == pytest.approx(0.7)


def test_roundtrip_predictions_are_identical(tmp_path):
    net = RegressionMLP(len(FEATURES))
    x_mean = np.zeros(8, dtype=np.float32)
    x_std = np.ones(8, dtype=np.float32)
    save_artifacts(str(tmp_path), net, x_mean, x_std, 2.0, 1.0, FEATURES)
    loaded = load_artifacts(str(tmp_path), device=CPU)

    rows = np.random.RandomState(2).randn(3, 8).astype(np.float32)
    net.eval()
    with torch.no_grad():
        expected = net(torch.tensor(rows)).numpy() * 1.0 + 2.0

    assert np.allclose(loaded.predict(rows), expected, atol=1e-5)


def test_load_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifacts(str(tmp_path / "does-not-exist"))


# --------------------------------------------------------------------------
# Learning sanity check (no network / dataset needed)
# --------------------------------------------------------------------------

def test_model_learns_a_linear_relationship():
    """The training loop should drive loss well below its starting value."""
    torch.manual_seed(0)
    X = torch.randn(512, 3)
    true_w = torch.tensor([2.0, -1.0, 0.5])
    y = X @ true_w + 0.1

    net = RegressionMLP(3, hidden=(16,))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    first_loss = None
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(net(X), y)
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()

    assert loss.item() < first_loss * 0.1
