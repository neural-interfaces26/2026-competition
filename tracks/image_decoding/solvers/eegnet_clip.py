"""Baseline — EEGNet trained with a CLIP-style retrieval loss.

Maps each EEG window to the target embedding space with an EEGNet backbone
(``n_outputs = D``) trained with an asymmetric InfoNCE / CLIP loss over the
batch (mirrors the official neuralbench baseline: targets L2-normalized,
fixed temperature, no symmetric term). Same device pattern as the other
EEGNet baselines.
"""

import torch
from torch.nn import functional as F
from braindecode.models import EEGNet

import benchmark_utils  # noqa: F401 — locates compet_core
from compet_core.base_solver import CompetSolver


def clip_loss(pred, target, temperature=0.07):
    """Asymmetric CLIP loss: classify each window among the batch targets."""
    target = F.normalize(target, dim=-1)
    logits = (F.normalize(pred, dim=-1) @ target.T) / temperature
    return F.cross_entropy(logits, torch.arange(len(pred),
                                                device=pred.device))


class EEGNetClip:
    """Torch embedding model (``fit``/``predict``) that owns its device."""

    def __init__(self, n_chans, n_times, n_outputs, device,
                 lr=1e-3, n_epochs=20):
        self.device = device
        self.n_epochs = n_epochs
        self.lr = lr
        self.net = EEGNet(
            n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
        ).to(device)

    def fit(self, train_loader):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for _ in range(self.n_epochs):
            for X, y, _info in train_loader:
                # X, y already live on self.device (moved by the loader).
                opt.zero_grad()
                loss = clip_loss(self.net(X), y.float())
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        self.net.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        return self.net(X)  # (B, D)


class Solver(CompetSolver):

    name = "EEGNet-CLIP"

    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        return EEGNetClip(
            n_chans=meta["n_chans"],
            n_times=meta["n_times"],
            n_outputs=meta["n_outputs"],
            device=self.device,
        )

    def fit(self, model, train_loader):
        model.fit(train_loader)
