"""Baseline — EEGNet regressor trained end-to-end (torch, GPU-ready).

The model owns its device (batches already live on ``meta['device']``);
single-output head trained with MSE on latencies normalized to ``[0, 1]``,
predictions mapped back to seconds and clamped to ``[0, CAP_S]``.
"""

import torch
from torch import nn
from braindecode.models import EEGNet

from benchmark_utils.base_solver import CompetSolver

CAP_S = 600.0


class EEGNetRegressor:
    """Torch regressor (``fit``/``predict``) that owns its device."""

    def __init__(self, n_chans, n_times, device, lr=1e-3, n_epochs=20):
        self.device = device
        self.n_epochs = n_epochs
        self.lr = lr
        self.net = EEGNet(
            n_chans=n_chans, n_outputs=1, n_times=n_times,
        ).to(device)

    def fit(self, train_loader):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.net.train()
        for _ in range(self.n_epochs):
            for X, y, _info in train_loader:
                # X, y already live on self.device (moved by the loader).
                opt.zero_grad()
                pred = self.net(X).squeeze(-1)
                loss = loss_fn(pred, y.float() / CAP_S)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        self.net.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        pred = self.net(X).squeeze(-1) * CAP_S
        return pred.clamp(0.0, CAP_S)  # (B,) seconds


class Solver(CompetSolver):

    name = "EEGNet"

    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        model = EEGNetRegressor(
            n_chans=meta["n_chans"],
            n_times=meta["n_times"],
            device=self.device,
        )

        # A submitted network ships the state dict written by ``save_model``;
        # without it (a local training run) the net starts from scratch.
        weights = meta["submission_dir"] / "weights.pt"
        if weights.exists():
            model.net.load_state_dict(
                torch.load(weights, map_location=self.device))
        return model

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        torch.save(model.net.state_dict(), path / "weights.pt")
