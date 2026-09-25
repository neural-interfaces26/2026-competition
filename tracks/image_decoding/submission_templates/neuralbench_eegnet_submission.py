"""Package an EEGNet checkpoint trained by the NeuralBench image start kit.

Copy this file to ``submission.py``, copy the retained NeuralBench checkpoint
to ``weights.pt``, then ZIP those two files. The loader accepts a Lightning
``best.ckpt``, a NeuralBench-exported state dict, or a bare EEGNet state dict.
"""

import torch
from braindecode.models import EEGNet

from benchmark_utils.base_solver import CompetSolver


def _load_neuralbench_weights(module, path, device):
    """Load only the brain model from a NeuralBench or plain checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    expected = module.state_dict()

    # NeuralBench's Lightning module stores the network below ``model.`` and
    # may also store learned loss parameters such as ``loss.temperature``.
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
            "The checkpoint does not match the NeuralBench EEGNet image "
            f"architecture. Missing keys: {missing}"
        )
    module.load_state_dict(state, strict=True)


class NeuralBenchEEGNetImage:
    def __init__(self, n_chans, n_times, n_outputs, device):
        self.device = device
        self.net = EEGNet(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
        ).to(device)

    @torch.inference_mode()
    def predict(self, X):
        self.net.eval()
        return self.net(X.to(self.device, dtype=torch.float32))


class Solver(CompetSolver):

    name = "NeuralBench-EEGNet-Image"
    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        model = NeuralBenchEEGNetImage(
            n_chans=meta["n_chans"],
            n_times=meta["n_times"],
            n_outputs=meta["n_outputs"],
            device=meta["device"],
        )
        _load_neuralbench_weights(
            model.net,
            meta["submission_dir"] / "weights.pt",
            meta["device"],
        )
        model.net.eval()
        return model
