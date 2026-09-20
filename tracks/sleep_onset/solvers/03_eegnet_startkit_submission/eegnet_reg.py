"""Inference-only EEGNet submission for Track 03.

This is the model-loading half of the future ready-to-upload EEGNet example.
It deliberately contains no training loop. Add a compatible ``weights.pt``
exported from the official start kit before packaging it for Codabench.
"""

import torch
from braindecode.models import EEGNet

from benchmark_utils.base_solver import CompetSolver

CAP_S = 600.0


class EEGNetRegressor:
    """Braindecode EEGNet exposing the competition ``predict`` method."""

    def __init__(self, n_chans, n_times, device):
        self.device = device
        self.net = EEGNet(
            n_chans=n_chans, n_outputs=1, n_times=n_times,
        ).to(device)

    @torch.inference_mode()
    def predict(self, X):
        self.net.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        pred = self.net(X).squeeze(-1) * CAP_S
        return pred.clamp(0.0, CAP_S)  # (B,) seconds


class Solver(CompetSolver):

    name = "EEGNet"

    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        device = meta["device"]
        weights = meta["submission_dir"] / "weights.pt"
        if not weights.is_file():
            raise FileNotFoundError(
                "This inference example requires a compatible weights.pt "
                "at the submission ZIP root."
            )

        model = EEGNetRegressor(
            n_chans=meta["n_chans"],
            n_times=meta["n_times"],
            device=device,
        )
        state_dict = torch.load(
            weights, map_location=device, weights_only=True
        )
        model.net.load_state_dict(state_dict)
        model.net.eval()
        return model
