"""Dummy PyTorch submission for Track 03, Sleep Onset.

Submission contract
-------------------
* Codabench imports ``class Solver(CompetSolver)`` from this file.
* ``load_model(meta)`` reconstructs the model and loads shipped weights.
* ``meta["submission_dir"]`` is read-only and contains the uploaded files.
* ``meta["device"]`` is the device used by both the model and input batches.
* ``predict(X)`` receives ``X: (B, C, T)`` and returns ``(B,)`` seconds.
* Server evaluation is inference-only. Training never runs on Codabench.

Upload ``submission.py``, ``weights.pt``, and ``config.json`` together at the
ZIP root. The included checkpoint is intentionally untrained and only
validates the full submission path. Replace it with your trained state dict.
"""

import json

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver


class MinimalSleepCNN(nn.Module):
    """Small channel-agnostic CNN returning one latency per EEG window."""

    def __init__(self, hidden_channels=8, max_latency_s=600.0):
        super().__init__()
        self.max_latency_s = max_latency_s
        # The same temporal filters process every channel. The checkpoint is
        # therefore independent of the number of channels and window length.
        self.temporal = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=25,
                      stride=5, padding=12),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Linear(hidden_channels, 1)

    def forward(self, X):
        if X.ndim != 3:
            raise ValueError(
                f"Expected X with shape (B, C, T), got {tuple(X.shape)}"
            )

        # This example deliberately infers B, C, and T from every input batch.
        # Applying the same temporal filters to each channel and then averaging
        # over channels lets one checkpoint accept different channel counts and
        # window lengths without knowing them when the model is constructed.
        batch_size, n_chans, n_times = X.shape
        X = X.to(dtype=torch.float32)
        X = X - X.mean(dim=-1, keepdim=True)
        scale = X.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        X = (X / scale).reshape(
            batch_size * n_chans, 1, n_times
        )
        features = self.temporal(X).squeeze(-1)
        features = features.reshape(batch_size, n_chans, -1).mean(dim=1)
        latency = self.regressor(features).squeeze(-1)
        return latency.clamp(0.0, self.max_latency_s)

    @torch.inference_mode()
    def predict(self, X):
        # The benchmark already puts X on meta["device"].
        self.eval()
        return self(X)


class Solver(CompetSolver):
    """Entry point discovered by the Codabench ingestion program."""

    name = "Dummy-Sleep-CNN"
    # The worker already provides PyTorch. It installs nothing at submission
    # time, so a submission must only import packages present in its image.
    requirements = []

    def load_model(self, meta):
        device = meta["device"]
        config_path = meta["submission_dir"] / "config.json"
        weights = meta["submission_dir"] / "weights.pt"
        if not config_path.is_file() or not weights.is_file():
            raise FileNotFoundError(
                "config.json and weights.pt must be included beside "
                "submission.py"
            )

        # Any additional shipped file can be read from submission_dir.
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)

        # MinimalSleepCNN is shape-agnostic, so it does not need n_chans or
        # n_times here. A fixed-size architecture would instead use, for
        # example:
        #
        # model = MyModel(
        #     n_chans=meta["n_chans"], n_times=meta["n_times"]
        # )
        model = MinimalSleepCNN(
            max_latency_s=config["max_latency_s"]
        ).to(device)
        state_dict = torch.load(
            weights, map_location=device, weights_only=True
        )
        model.load_state_dict(state_dict)
        return model.eval()
