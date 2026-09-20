"""Reference baseline: a linear network trained in plain PyTorch.

The smallest end-to-end torch example of the submission contract — a model
built in ``load_model`` from ``meta``, an Adam loop in ``fit``, and
``save_model`` writing the checkpoint ``load_model`` reads back. Swap the
network for your own architecture and the rest still holds.

It is ``mean_ridge`` written in torch instead of scikit-learn: each window
collapses to one value per channel, then a linear layer maps those to a
latency. Latencies are regressed in ``[0, 1]`` (seconds over ``CAP_S``) so
the loss is well scaled, and mapped back to seconds in ``predict``.
"""

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver

CAP_S = 600.0


class _MeanOverTime(nn.Module):
    """(B, C, T) -> (B, C)."""

    def forward(self, X):
        return X.mean(dim=-1)


class TorchLinear:
    """Torch regressor (``fit``/``predict``) that owns its device."""

    def __init__(self, n_chans, device, lr=1e-2, n_epochs=30):
        self.device = device
        self.lr = lr
        self.n_epochs = n_epochs
        # Mean over time, then one weight per channel. A model that reads
        # the raw (B, C, T) window instead would just take (C, T) here.
        self.net = nn.Sequential(
            _MeanOverTime(),
            nn.Linear(n_chans, 1),
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
        return (self.net(X).squeeze(-1) * CAP_S).clamp(0.0, CAP_S)


class Solver(CompetSolver):

    name = "Torch-Linear"

    def load_model(self, meta):
        model = TorchLinear(n_chans=meta["n_chans"], device=self.device)
        # A submitted model ships the state dict written by save_model;
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
