"""Minimal trainable PyTorch solver for sleep-onset regression.

This example demonstrates the complete Benchopt path: initialize a model,
train it locally through ``fit``, save its weights, and export an upload-ready
Codabench ZIP. Codabench itself only uses ``load_model`` and ``predict``.
"""

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver


CAP_S = 600.0


class LinearSleepRegressor(nn.Module):
    """Regress sleep-onset latency from one global signal feature."""

    def __init__(self, device, lr=1e-3, n_epochs=20):
        super().__init__()
        self.device = device
        self.lr = lr
        self.n_epochs = n_epochs
        self.linear = nn.Linear(1, 1)
        self.to(device)

    def forward(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        feature = X.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        return torch.sigmoid(self.linear(feature).squeeze(-1)) * CAP_S

    def fit(self, train_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.train()
        for _ in range(self.n_epochs):
            for X, y, _info in train_loader:
                optimizer.zero_grad()
                prediction = self(X)
                loss = loss_fn(prediction / CAP_S, y.float() / CAP_S)
                loss.backward()
                optimizer.step()
        return self

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return self(X)


class Solver(CompetSolver):

    name = "Linear"
    requirements = []

    def load_model(self, meta):
        model = LinearSleepRegressor(device=meta["device"])
        weights = meta["submission_dir"] / "weights.pt"
        if weights.is_file():
            model.load_state_dict(torch.load(
                weights, map_location=meta["device"], weights_only=True
            ))
        elif self.train_loader is None:
            raise FileNotFoundError(
                "Inference requires weights.pt at the submission ZIP root. "
                "Run this solver with training=True to create it."
            )
        return model.eval()

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        torch.save(model.state_dict(), path / "weights.pt")
