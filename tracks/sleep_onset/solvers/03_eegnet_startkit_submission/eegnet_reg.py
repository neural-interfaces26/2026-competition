"""Trained, inference-only EEGNet submission for Track 03 warm-up.

The shipped weights come from the official NeuralBench sleep-onset start kit
trained on Sleep-EDF. Codabench only reconstructs the model and runs inference.
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
        # NeuralBench trains this regressor directly on latency in seconds.
        pred = self.net(X).squeeze(-1)
        return pred.clamp(0.0, CAP_S)  # (B,) seconds


class Solver(CompetSolver):

    name = "EEGNet"

    requirements = ["pip::braindecode==1.8.1"]

    def load_model(self, meta):
        device = meta["device"]
        weights = meta["submission_dir"] / "weights.pt"
        if not weights.is_file():
            raise FileNotFoundError(
                "This submission requires its trained weights.pt "
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
