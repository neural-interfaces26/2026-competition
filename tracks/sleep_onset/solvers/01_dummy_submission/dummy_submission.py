"""Dummy PyTorch submission for Track 03, Sleep Onset.

Submission contract
-------------------
* Codabench imports ``class Solver(CompetSolver)`` from this file.
* ``load_model(meta)`` reconstructs the model and loads shipped weights.
* ``meta["submission_dir"]`` is read-only and contains the uploaded files.
* ``meta["device"]`` is the device used by both the model and input batches.
* ``predict(X)`` receives ``X: (B, C, T)`` and returns ``(B,)`` seconds.
* Server evaluation is inference-only. Training never runs on Codabench.

Run ``make_zip.py`` to package this file as ``submission.py`` beside
``dummy_weights.pt`` and ``dummy_config.json``. The included checkpoint is
intentionally untrained and only validates the full submission path.
"""

import json

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver


class DummySleepLinear(nn.Module):
    """Dummy linear model returning one latency per EEG window."""

    def __init__(self, max_latency_s=600.0):
        super().__init__()
        self.max_latency_s = max_latency_s
        self.linear = nn.Linear(1, 1)

    def forward(self, X):
        if X.ndim != 3:
            raise ValueError(
                f"Expected X with shape (B, C, T), got {tuple(X.shape)}"
            )

        # The single feature is the mean of each complete signal window. This
        # keeps the dummy checkpoint independent of channel count and length.
        X = X.to(dtype=torch.float32)
        feature = X.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        latency = torch.sigmoid(self.linear(feature).squeeze(-1))
        return latency * self.max_latency_s

    @torch.inference_mode()
    def predict(self, X):
        # The benchmark already puts X on meta["device"].
        self.eval()
        return self(X)


class Solver(CompetSolver):
    """Entry point discovered by the Codabench ingestion program."""

    name = "Dummy-Sleep-Linear"
    # The worker already provides PyTorch. It installs nothing at submission
    # time, so a submission must only import packages present in its image.
    requirements = []

    def load_model(self, meta):
        device = meta["device"]
        config_path = meta["submission_dir"] / "dummy_config.json"
        weights = meta["submission_dir"] / "dummy_weights.pt"
        if not config_path.is_file() or not weights.is_file():
            raise FileNotFoundError(
                "dummy_config.json and dummy_weights.pt must be included "
                "at the submission ZIP root"
            )

        # Any additional shipped file can be read from submission_dir.
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)

        # DummySleepLinear is shape-agnostic, so it does not need n_chans or
        # n_times here. A fixed-size architecture would instead use, for
        # example:
        #
        # model = MyModel(
        #     n_chans=meta["n_chans"], n_times=meta["n_times"]
        # )
        model = DummySleepLinear(
            max_latency_s=config["max_latency_s"]
        ).to(device)
        state_dict = torch.load(
            weights, map_location=device, weights_only=True
        )
        model.load_state_dict(state_dict)
        return model.eval()
