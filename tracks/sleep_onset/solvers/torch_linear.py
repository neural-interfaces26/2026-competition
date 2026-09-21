"""Reference baseline: a linear network trained in plain PyTorch.

The smallest end-to-end torch example of the submission contract: an
``nn.Module`` built in ``load_model`` from ``meta``, the Adam loop right in
``Solver.fit``, and ``save_model`` writing the checkpoint ``load_model`` reads
back. Swap the network for your own architecture and the rest still holds.

Each window collapses to one value per channel (mean over time), then a linear
layer maps those to a latency. Latencies are regressed in ``[0, 1]`` (seconds
over ``CAP_S``) so the loss is well scaled, and mapped back to seconds in
``predict``.
"""

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver

CAP_S = 600.0


class TorchLinear(nn.Module):
    """(B, C, T) -> mean over time -> linear -> one latency in ``[0, 1]``."""

    def __init__(self, n_chans):
        super().__init__()
        self.linear = nn.Linear(n_chans, 1)

    def forward(self, X):
        return self.linear(X.mean(dim=-1)).squeeze(-1)  # (B,)

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        return (self(X) * CAP_S).clamp(0.0, CAP_S)  # (B,) seconds


class Solver(CompetSolver):

    name = "Torch-Linear"

    def load_model(self, meta):
        model = TorchLinear(n_chans=meta["n_chans"]).to(self.device)
        weights = meta["submission_dir"] / "weights.pt"
        if weights.exists():
            model.load_state_dict(
                torch.load(weights, map_location=self.device))
        return model

    def fit(self, model, train_loader):
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(30):  # epochs
            for X, y, _info in train_loader:
                # X, y already sit on the model's device (loader moved them).
                opt.zero_grad()
                loss = loss_fn(model(X), y.float() / CAP_S)
                loss.backward()
                opt.step()

    def save_model(self, model, path):
        torch.save(model.state_dict(), path / "weights.pt")
