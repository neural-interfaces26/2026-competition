"""Baseline — frozen REVE encoder + ridge linear probe.

A complete submission example: the frozen encoder is defined *here* (as
``REVEEncoder``) so you can swap it, the head, or the pooling freely. REVE
(braindecode, pretrained) maps each window to an embedding; a ridge head on
the mean-pooled embedding regresses the onset latency (seconds). Only the head
is trained — its weights travel as a joblib dump; REVE's weights are pulled
from the Hugging Face Hub at load time (not shipped).

``LinearProbe`` / ``Encoder`` are shared infra in
``benchmark_utils/linear_probe.py`` — reused here to keep the example short;
inline them if you want to change the probe itself.
"""

import joblib
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.linear_probe import Encoder, LinearProbe


class REVEEncoder(Encoder):
    """Frozen pretrained braindecode REVE (200 Hz) -> ``(B, T', D)``.

    Needs a window of at least ~1 s (200 samples at 200 Hz); shorter windows
    raise in ``encode``.
    """

    REVE_SFREQ = 200

    def __init__(self, meta):
        from braindecode.models import REVE
        self.sfreq = float(meta["sfreq"])
        self.device = meta.get("device", "cpu")
        self._n_times = max(
            round(meta["n_times"] * self.REVE_SFREQ / self.sfreq), 1
        )
        self.model = REVE.from_pretrained(
            "brain-bzh/reve-base",
            n_outputs=1,
            n_chans=meta["n_chans"],
            n_times=self._n_times,
            sfreq=self.REVE_SFREQ,
            chs_info=meta["chs_info"],
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def encode(self, X):
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        if X.shape[-1] != self._n_times:
            X = F.interpolate(
                X, size=self._n_times, mode="linear", align_corners=False
            )
        with torch.inference_mode():
            out = self.model(X, return_features=True)
        feats = out["features"] if isinstance(out, dict) else out
        if feats.ndim == 4:      # (B, C, T', D) -> mean over channels
            feats = feats.mean(dim=1)
        elif feats.ndim == 2:    # (B, D) -> add a singleton time axis
            feats = feats[:, None, :]
        return feats             # (B, T', D)


class Solver(CompetSolver):

    name = "REVE"

    requirements = ["pip::braindecode", "pip::safetensors"]

    def load_model(self, meta):
        probe = LinearProbe(REVEEncoder(meta), Ridge())
        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists():
            print(f"[loading] {weights} into {self.name}")
            probe.head = joblib.load(weights)
        return probe

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.head, path / "weights.joblib")
