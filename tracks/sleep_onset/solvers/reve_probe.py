"""Baseline — frozen REVE encoder + ridge linear probe.

A complete submission example: the frozen encoder is defined *here* (as
``REVEEncoder``) so you can swap it, the head, or the pooling freely. REVE
(braindecode, pretrained) maps each window to an embedding; a ridge head on
the mean-pooled embedding regresses the onset latency (seconds). Only the head
is trained. Its weights travel as a joblib dump; REVE's pretrained weights
come from the worker's staged Hugging Face cache and are downloaded once on
first use in a local environment.

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
    PATCH_SIZE = 200

    def __init__(self, meta):
        from braindecode.models import REVE
        self.sfreq = float(meta["sfreq"])
        self.device = meta.get("device", "cpu")
        self._n_times = max(
            round(meta["n_times"] * self.REVE_SFREQ / self.sfreq), 1
        )
        if self._n_times < self.PATCH_SIZE:
            raise ValueError(
                "REVE needs at least one second of EEG after resampling "
                f"to 200 Hz; this window has {self._n_times} samples"
            )
        self.model = REVE.from_pretrained(
            "brain-bzh/reve-base",
            n_outputs=1,
            n_chans=meta["n_chans"],
            n_times=self._n_times,
            sfreq=self.REVE_SFREQ,
            chs_info=None,
        ).to(self.device)
        self._set_channel_positions(meta.get("ch_names"))
        self.model.eval()
        self.model.requires_grad_(False)

    def _set_channel_positions(self, ch_names):
        if not ch_names or len(ch_names) != self.model.n_chans:
            raise ValueError(
                "REVE needs one standard EEG channel name per input channel"
            )
        # Sleep-EDF exposes bipolar names such as Fpz-Cz. REVE expects one
        # standard electrode position per signal, so use the active electrode
        # (Fpz and Pz) consistently with NeuralBench's montage fallback.
        names = [str(name).removeprefix("EEG ").split("-", 1)[0]
                 for name in ch_names]
        known = self.model._position_bank.mapping
        unknown = [name for name in names if name not in known]
        if unknown:
            raise ValueError(
                "REVE cannot locate these EEG channels: "
                + ", ".join(unknown)
            )
        self.model.default_pos = self.model.get_positions(names)

    def encode(self, X):
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        if X.shape[-1] != self._n_times:
            X = F.interpolate(
                X, size=self._n_times, mode="linear", align_corners=False
            )
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        X = ((X - mean) / std).clamp(-15.0, 15.0)
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
        elif self.train_loader is None:
            raise FileNotFoundError(
                "REVE inference needs weights.joblib. Train this solver with "
                "training=True and upload the exported submission folder."
            )
        return probe

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.head, path / "weights.joblib")
