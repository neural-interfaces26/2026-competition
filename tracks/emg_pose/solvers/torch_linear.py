"""Reference baseline: a per-time-step linear network in plain PyTorch.

The smallest end-to-end torch example of the submission contract, and the
same model as ``ridge_pose`` in PyTorch with its own Adam loop. The Simulated
EMG is an instantaneous linear mixture of the joint angles, so a linear layer
applied at each time step maps ``(B, C, T) -> (B, n_joints, T)``.

Angles are regressed in a standardized space (train mean/std kept as buffers,
so a submitted state dict restores the scale), mirroring the EEGNet baseline.
Swap the network for your own architecture and the rest still holds.
"""

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver


class TorchLinearPose(nn.Module):
    """Per-time-step map ``(B, C, T) -> (B, n_joints, T)``, in radians."""

    def __init__(self, n_chans, n_joints):
        super().__init__()
        self.linear = nn.Linear(n_chans, n_joints)
        # Target scale, learned in fit; identity until then so an untrained
        # net still predicts in radians.
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def forward(self, X):
        # (B, C, T) -> (B, T, C) -> linear -> (B, T, J) -> (B, J, T)
        return self.linear(X.transpose(1, 2)).transpose(1, 2)

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        return self(X) * self.y_std + self.y_mean       # (B, J, T) radians


class Solver(CompetSolver):

    name = "Torch-Linear"

    def load_model(self, meta):
        model = TorchLinearPose(
            n_chans=meta["n_chans"], n_joints=meta["n_joints"],
        ).to(self.device)
        weights = meta["submission_dir"] / "weights.pt"
        if weights.exists():
            print(f"[loading] {weights} into {self.name}")
            model.load_state_dict(
                torch.load(weights, map_location=self.device))
        return model

    def fit(self, model, train_loader):
        # One pass to set the target scale, so the loss is well conditioned
        # whatever the angle range of the study.
        total = sq = count = 0.0
        for _X, y, _info in train_loader:
            y = y.float()
            total += y.sum()
            sq += (y ** 2).sum()
            count += y.numel()
        mean = total / count
        model.y_mean.fill_(float(mean))
        model.y_std.fill_(
            float((sq / count - mean ** 2).clamp_min(1e-8).sqrt()))

        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(30):  # epochs
            for X, y, _info in train_loader:
                # X, y already sit on the model's device (loader moved them).
                opt.zero_grad()
                target = (y.float() - model.y_mean) / model.y_std
                loss = loss_fn(model(X), target)
                loss.backward()
                opt.step()

    def save_model(self, model, path):
        torch.save(model.state_dict(), path / "weights.pt")
