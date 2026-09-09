"""Baseline — EEGNet trained end-to-end (torch, GPU-ready).

The reference for *how a submission handles the device*: the contract only
requires a model exposing ``predict(X)`` (plus the solver's optional ``fit``),
so device placement is entirely internal to the model.

The pattern (mirror it in your own submission):

- batches already arrive on ``meta["device"]`` (the dataset's loaders move
  ``X``/``y`` there), so the network just has to live on the same device —
  build it and call ``.to(device)`` in ``load_model``;
- ``fit`` runs a plain torch training loop over the loader; ``X``/``y`` are
  already on-device, nothing else to move;
- ``predict`` moves its input to the device defensively (cheap no-op when the
  caller already did) and returns per-window labels.

braindecode is a hard requirement: it is imported at module top so that, if
it is missing, benchopt reports the solver as *not installed* rather than a
fallback silently hiding the missing dependency.
"""

import torch
from torch import nn
from braindecode.models import EEGNetv4

import benchmark_utils  # noqa: F401 — locates compet_core
from compet_core.base_solver import CompetSolver


class EEGNetModel:
    """Torch model (``fit``/``predict``) that owns its device.

    Trains an EEGNet for ``n_epochs`` with Adam + cross-entropy. ``X`` is
    ``(B, C, T)`` and predictions are ``(B,)`` labels.
    """

    def __init__(self, n_chans, n_times, n_classes, device,
                 lr=1e-3, n_epochs=20):
        self.device = device
        self.n_epochs = n_epochs
        self.lr = lr
        self.net = EEGNetv4(
            n_chans=n_chans, n_outputs=n_classes, n_times=n_times,
        ).to(device)

    def fit(self, train_loader):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        self.net.train()
        for _ in range(self.n_epochs):
            for X, y, _info in train_loader:
                # X, y already live on self.device (moved by the loader).
                opt.zero_grad()
                loss = loss_fn(self.net(X), y)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        self.net.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        return self.net(X).argmax(dim=1)  # (B,)


class Solver(CompetSolver):

    name = "EEGNet"

    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        return EEGNetModel(
            n_chans=meta["n_chans"],
            n_times=meta["n_times"],
            n_classes=meta["n_classes"],
            device=self.device,
        )

    def fit(self, model, train_loader):
        model.fit(train_loader)
