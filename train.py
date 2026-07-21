"""Train a PyTorch regression model on the California Housing dataset.

The dataset is a public regression benchmark fetched via scikit-learn
(downloaded once and cached). The trained model, feature scaling, and
metadata are written to the model directory (default: ``models/``), which is
the same directory the FastAPI service reads at inference time.

Usage:
    python train.py                 # sensible defaults
    python train.py --epochs 50     # override
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import config
from model import RegressionMLP, save_artifacts, select_device


def parse_args() -> argparse.Namespace:
    # Defaults live in config.py (the single source of truth); flags override.
    parser = argparse.ArgumentParser(description="Train the regression model.")
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "models"))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device()
    print(f"Using device: {device}")

    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)
    feature_names = list(data.feature_names)
    print(f"Loaded California Housing: {X.shape[0]} rows, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    # Standardize features and target using TRAIN statistics only.
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean, y_std = float(y_train.mean()), float(y_train.std()) or 1.0

    def scale_x(a: np.ndarray) -> np.ndarray:
        return (a - x_mean) / x_std

    Xtr = torch.tensor(scale_x(X_train), device=device)
    ytr = torch.tensor((y_train - y_mean) / y_std, device=device)
    Xte = torch.tensor(scale_x(X_test), device=device)
    yte = torch.tensor((y_test - y_mean) / y_std, device=device)

    model = RegressionMLP(in_features=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    n = Xtr.shape[0]
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            optimizer.zero_grad()
            pred = model(Xtr[idx])
            loss = loss_fn(pred, ytr[idx])
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # Report RMSE in the original target units.
                test_pred = model(Xte) * y_std + y_mean
                test_true = yte * y_std + y_mean
                rmse = torch.sqrt(torch.mean((test_pred - test_true) ** 2))
            print(f"epoch {epoch:3d} | test RMSE = {rmse.item():.4f}")

    save_artifacts(
        args.model_dir, model, x_mean, x_std, y_mean, y_std, feature_names
    )
    print(f"Saved model artifacts to {args.model_dir}/")


if __name__ == "__main__":
    main()
