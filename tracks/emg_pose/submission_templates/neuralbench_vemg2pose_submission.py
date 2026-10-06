"""Package a VEMG2Pose checkpoint trained by the NeuralBench pose start kit.

Copy this file to ``submission.py``, copy the retained NeuralBench checkpoint
to ``weights.pt``, then ZIP those two files. The loader accepts a Lightning
``best.ckpt``, a NeuralBench-exported state dict, or a bare model state dict.

NeuralBench trains VEMG2Pose on 5.895-second windows. The first 1,790 samples
provide the model's temporal context and its output covers the final 5 seconds.
Codabench supplies those scored 5-second windows, so ``predict`` recreates the
left context before inference. Predictions remain in radians as required by
the submission contract; only the final leaderboard error is shown in degrees.
"""

import torch
from torch.nn import functional as F
from braindecode.models import VEMG2Pose

from benchmark_utils.base_solver import CompetSolver

TRAIN_N_TIMES = 11_790
LEFT_CONTEXT = 1_790


def _load_neuralbench_weights(module, path, device):
    """Load only the brain model from a NeuralBench or plain checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    expected = module.state_dict()

    if any(key.startswith("model.") for key in state):
        state = {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        }
    state = {key: value for key, value in state.items() if key in expected}

    missing = sorted(set(expected) - set(state))
    if missing:
        raise RuntimeError(
            "The checkpoint does not match the NeuralBench VEMG2Pose "
            f"architecture. Missing keys: {missing}"
        )
    module.load_state_dict(state, strict=True)


class NeuralBenchVEMG2Pose:
    def __init__(self, n_chans, n_joints, sfreq, device):
        self.device = device
        self.net = VEMG2Pose(
            n_chans=n_chans,
            n_times=TRAIN_N_TIMES,
            n_outputs=n_joints,
            sfreq=sfreq,
        ).to(device)

    @torch.inference_mode()
    def predict(self, X):
        self.net.eval()
        X = X.to(self.device, dtype=torch.float32)
        mode = "reflect" if X.shape[-1] > LEFT_CONTEXT else "replicate"
        X = F.pad(X, (LEFT_CONTEXT, 0), mode=mode)
        prediction = self.net(X)  # (B, T, joints), radians
        return prediction.transpose(1, 2)  # (B, joints, T), radians


class Solver(CompetSolver):

    name = "NeuralBench-VEMG2Pose"
    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        model = NeuralBenchVEMG2Pose(
            n_chans=meta["n_chans"],
            n_joints=meta["n_joints"],
            sfreq=meta["sfreq"],
            device=meta["device"],
        )
        _load_neuralbench_weights(
            model.net,
            meta["submission_dir"] / "weights.pt",
            meta["device"],
        )
        model.net.eval()
        return model
