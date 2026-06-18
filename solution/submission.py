"""Sample submission for the EEG competition — foundation-model track.

A submission is a **valid benchopt solver** subclassing ``CompetEEGSolver``.
You only implement how your *frozen* foundation model embeds windows:

- ``load_model(self, meta)``      -> the frozen model (loaded once). ``meta``
                                     carries ``sfreq, ch_names, chs_info,
                                     n_chans, n_times, n_classes, task``.
- ``time_embed(self, model, X)``  -> ``(B, T', D)`` temporal embedding for a
                                     batch of windows ``X: (B, C, T)`` (torch).
- ``embed(self, model, X)``       -> ``(B, D)`` window embedding. Optional;
                                     defaults to mean-pooling ``time_embed``.

The competition fits the scikit-learn linear head for you (on-device, via the
array API) and scores both the epoched and dense tasks — you ship only the
encoder. This sample wraps braindecode's **REVE**, with a dependency-light
fallback so it runs in CI without the pretrained weights.

Test it locally (from the bundle root):

    cp solution/submission.py benchmark/solvers/_submission.py
    benchopt run benchmark/ -d Simulated -s REVE
    rm benchmark/solvers/_submission.py
"""

import torch

from benchmark_utils.base_solver import CompetEEGSolver

REVE_SFREQ = 200  # REVE is pretrained at 200 Hz


class Solver(CompetEEGSolver):

    name = "REVE"

    # ``pip::torch`` etc. are inherited from CompetEEGSolver; add the model's.
    requirements = CompetEEGSolver.requirements + ["pip::braindecode"]

    def load_model(self, meta):
        self._sfreq = float(meta.get("sfreq", REVE_SFREQ))
        self._fallback = False
        try:
            from braindecode.models import REVE
            model = REVE.from_pretrained("brain-bzh/reve-base")
            model.eval()
            model.requires_grad_(False)
            return model
        except Exception as e:  # ImportError / missing weights / network
            print(f"[submission] REVE unavailable ({e!r}); using fallback.")
            from benchmark_utils.linear_probe import RandomProjectionEncoder
            self._fallback = True
            return RandomProjectionEncoder(n_chans=meta["n_chans"])

    def _resample(self, X):
        if int(self._sfreq) == REVE_SFREQ:
            return X
        n_out = int(round(X.shape[-1] * REVE_SFREQ / self._sfreq))
        return torch.nn.functional.interpolate(
            X, size=n_out, mode="linear", align_corners=False
        )

    def time_embed(self, model, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        if self._fallback:
            return model.encode(X)  # (B, T', D)
        X = self._resample(X)
        with torch.inference_mode():
            out = model(X, return_features=True)
        feats = out["features"] if isinstance(out, dict) else out
        if feats.ndim == 2:  # (B, D) -> add a singleton time axis
            feats = feats[:, None, :]
        return feats  # (B, T', D)
