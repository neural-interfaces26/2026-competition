"""Sample submission for the EEG models competition.

A submission is a *self-contained* ``submission.py`` exposing one or both
entry points, depending on the track you enter:

- **linear_probe track** (foundation models): ``get_encoder(meta)`` returns a
  *frozen* encoder with ``encode(X: (B, C, T)) -> (B, T', D)`` (a temporal
  embedding). The competition fits the linear head for you — you only ship the
  encoder. This sample wraps braindecode's **REVE** foundation model.

- **general track** (task-specific models): ``get_model(task, meta)`` returns a
  scikit-learn-like model with ``fit(train_loader)`` and
  ``predict((B, C, T)) -> (B,)`` (epoched) or ``(B, T)`` (dense).

``meta`` carries everything an encoder/model needs to configure itself:
``sfreq, ch_names, chs_info, n_chans, n_times, n_classes, task``.

Both entry points degrade to a dependency-light fallback when the heavy stack
(braindecode/REVE + pretrained weights) is unavailable, so the sample runs in
CI out of the box.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Foundation-model track: a frozen temporal-embedding encoder
# ---------------------------------------------------------------------------

REVE_SFREQ = 200  # REVE is pretrained at 200 Hz


class REVEEncoder:
    """Frozen braindecode REVE encoder -> per-patch temporal embedding.

    ``encode`` resamples each window to 200 Hz, runs REVE with
    ``return_features=True`` and returns the per-patch tokens ``(B, T', D)``.
    """

    def __init__(self, meta, weights="brain-bzh/reve-base"):
        import torch
        from braindecode.models import REVE

        self.torch = torch
        self.sfreq = float(meta.get("sfreq", REVE_SFREQ))
        self.model = REVE.from_pretrained(weights)
        self.model.eval()
        self.model.requires_grad_(False)

    def _resample(self, X):
        # X: (B, C, T) torch tensor at self.sfreq -> 200 Hz
        if int(self.sfreq) == REVE_SFREQ:
            return X
        torch = self.torch
        n_out = int(round(X.shape[-1] * REVE_SFREQ / self.sfreq))
        return torch.nn.functional.interpolate(
            X, size=n_out, mode="linear", align_corners=False
        )

    def encode(self, X):
        torch = self.torch
        X = torch.as_tensor(X, dtype=torch.float32)
        X = self._resample(X)
        with torch.inference_mode():
            out = self.model(X, return_features=True)
        feats = out["features"] if isinstance(out, dict) else out
        if feats.ndim == 2:  # (B, D) -> add a singleton time axis
            feats = feats[:, None, :]
        return feats  # (B, T', D) torch tensor


class _FallbackEncoder:
    """Numpy-only temporal embedding used when REVE is unavailable."""

    def __init__(self, meta, patch_len=20, d=64, seed=0):
        self.patch_len = patch_len
        self.d = d
        rng = np.random.default_rng(seed)
        self._proj = None
        self._rng = rng

    def encode(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[None]
        B, C, T = X.shape
        n_patches = max(T // self.patch_len, 1)
        usable = n_patches * self.patch_len
        patches = X[:, :, :usable].reshape(B, C, n_patches, self.patch_len)
        patches = patches.transpose(0, 2, 1, 3).reshape(B, n_patches, -1)
        if self._proj is None or self._proj.shape[0] != patches.shape[-1]:
            in_dim = patches.shape[-1]
            self._proj = self._rng.standard_normal((in_dim, self.d))
            self._proj /= np.sqrt(in_dim)
        return patches @ self._proj


def get_encoder(meta):
    """Return a frozen encoder for the linear-probe track."""
    try:
        return REVEEncoder(meta)
    except Exception as e:  # ImportError, missing weights, network, ...
        print(f"[submission] REVE unavailable ({e!r}); using fallback.")
        return _FallbackEncoder(meta)


# ---------------------------------------------------------------------------
# Specialist track: a task-specific model
# ---------------------------------------------------------------------------

class FlattenLogReg:
    """Simple specialist baseline (epoched + dense), scikit-learn based."""

    def __init__(self, task_kind):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.task_kind = task_kind
        self.clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        )

    @staticmethod
    def _to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def fit(self, train_loader):
        feats, targets = [], []
        for X, y, _info in train_loader:
            X, y = self._to_numpy(X), self._to_numpy(y)
            if self.task_kind == "epoched":
                feats.append(X.mean(axis=-1))      # (B, C)
                targets.append(y)
            else:
                B, C, T = X.shape
                feats.append(X.transpose(0, 2, 1).reshape(B * T, C))
                targets.append(y.reshape(-1))
        self.clf.fit(np.concatenate(feats), np.concatenate(targets))
        return self

    def predict(self, X):
        X = self._to_numpy(X)
        if self.task_kind == "epoched":
            return self.clf.predict(X.mean(axis=-1))
        B, C, T = X.shape
        flat = self.clf.predict(X.transpose(0, 2, 1).reshape(B * T, C))
        return flat.reshape(B, T)


def get_model(task, meta):
    """Return a task-specific model for the general track."""
    return FlattenLogReg(task_kind=meta["task_kind"])
