"""Reference baseline: a linear network in plain PyTorch.

The smallest end-to-end torch example of the submission contract, and the
same model as ``mean_ridge`` in PyTorch with its own Adam loop. Inference,
all the platform runs, needs only ``load_model`` (build the ``nn.Module``
from ``meta``, load its weights) and the module's ``predict``.

Each window collapses to one value per channel (mean over time); a linear
layer maps those to a ``D``-dim embedding, trained with MSE against the
target embeddings and ranked by cosine similarity in the objective. Swap the
network for your own architecture and the rest still holds.
"""

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver


class TorchLinear(nn.Module):
    """(B, C, T) -> mean over time -> linear -> an embedding (B, D)."""

    def __init__(self, n_chans, n_outputs):
        super().__init__()
        self.linear = nn.Linear(n_chans, n_outputs)

    def forward(self, X):
        return self.linear(X.mean(dim=-1))  # (B, D)

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        return self(X)  # (B, D)


class Solver(CompetSolver):

    name = "Torch-Linear"

    def load_model(self, meta):
        model = TorchLinear(
            n_chans=meta["n_chans"], n_outputs=meta["n_outputs"],
        ).to(self.device)
        weights = meta["submission_dir"] / "weights.pt"
        if weights.exists():
            print(f"[loading] {weights} into {self.name}")
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
                loss = loss_fn(model(X), y.float())
                loss.backward()
                opt.step()

    def save_model(self, model, path):
        torch.save(model.state_dict(), path / "weights.pt")
