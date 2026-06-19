"""Temporal-embedding encoder protocol + linear probe.

Heart of the **linear-probe (foundation-model) track**. A submission provides
only a frozen *encoder* mapping a batch of windows to a **temporal
embedding**::

    encode(X: (B, C, T)) -> (B, T', D)

i.e. one ``D``-dim vector per output time position ``T'`` (no pooling). The
encoder receives **torch tensors** (so braindecode foundation models run
directly), and thanks to scikit-learn's **array API** support the torch
features are passed straight to the linear head — no numpy round-trip, data
stays on-device. One encoder style serves both regimes:

- ``epoched`` : mean-pool over ``T'`` → one label per window.
- ``dense``   : keep every position; the head maps ``(B, T', D) -> (B, T',
                K+1)``; predictions are upsampled back to the raw ``T``.

The encoder is frozen and never sees the labels.
"""

import os
# scikit-learn array-API dispatch requires scipy's array-API support, which
# is gated behind this env var and must be set before scipy is imported.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

from abc import ABC, abstractmethod  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn import config_context  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from benchmark_utils.data import to_numpy  # noqa: E402


class Encoder(ABC):
    """Frozen feature extractor returning a temporal embedding.

    Subclasses implement :meth:`encode`, receiving a torch tensor
    ``(B, C, T)`` and returning ``(B, T', D)``. The time axis ``T'`` is kept
    so the same encoder serves window-classification and dense labelling.
    """

    @abstractmethod
    def encode(self, X):
        """Map ``(B, C, T)`` to a temporal embedding ``(B, T', D)``."""


class RandomProjectionEncoder(Encoder):
    """Dependency-light default/fallback encoder (torch only).

    Splits each window into ``T // patch_len`` non-overlapping time patches
    and random-projects each flattened ``(C * patch_len)`` patch to ``D``
    dims, mimicking a real EEG foundation model's per-patch tokenisation so
    the linear-probe path runs without pretrained weights.
    """

    def __init__(self, n_chans, patch_len=20, d=64, seed=0):
        self.n_chans = n_chans
        self.patch_len = patch_len
        self.d = d
        self._gen = torch.Generator().manual_seed(seed)
        self._proj = None  # built lazily once the patch size is known

    def _projection(self, in_dim):
        if self._proj is None or self._proj.shape[0] != in_dim:
            self._proj = torch.randn(
                in_dim, self.d, generator=self._gen
            ) / np.sqrt(in_dim)
        return self._proj

    def encode(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        if X.ndim == 2:  # (C, T) -> (1, C, T)
            X = X[None]
        B, C, T = X.shape
        n_patches = max(T // self.patch_len, 1)
        usable = n_patches * self.patch_len
        # (B, C, n_patches, patch_len) -> (B, n_patches, C * patch_len)
        patches = X[:, :, :usable].reshape(B, C, n_patches, self.patch_len)
        patches = patches.permute(0, 2, 1, 3).reshape(B, n_patches, -1)
        proj = self._projection(patches.shape[-1]).to(patches.device)
        return patches @ proj  # (B, T', D)


def _resample(seq, new_len):
    """Nearest-neighbour resample a torch label sequence on the last axis."""
    length = seq.shape[-1]
    if length == new_len:
        return seq
    idx = torch.floor(
        torch.arange(new_len, device=seq.device) * (length / new_len)
    ).long().clamp(0, length - 1)
    return seq[..., idx]


class LinearProbe:
    """Frozen encoder + per-position scikit-learn linear head (array API).

    Parameters
    ----------
    encoder : Encoder
        Frozen feature extractor (``encode(X) -> (B, T', D)``).
    mode : {"epoched", "dense"}
        ``epoched`` pools over ``T'`` and predicts one label per window;
        ``dense`` fits the head over every position and predicts a per-step
        sequence (upsampled to the input length on ``predict``).
    """

    def __init__(self, encoder, mode="epoched"):
        self.encoder = encoder
        self.mode = mode
        self.head = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )

    def fit(self, train_loader):
        feats, targets = [], []
        for X, y, _info in train_loader:
            emb = torch.as_tensor(self.encoder.encode(X))  # (B, T', D)
            if self.mode == "epoched":
                feats.append(emb.mean(dim=1))            # (B, D)
                targets.append(y)                        # (B,)
            else:  # dense
                y_ds = _resample(y, emb.shape[1])        # (B, T')
                feats.append(emb.reshape(-1, emb.shape[-1]))
                targets.append(y_ds.reshape(-1))
        X_feat = torch.cat(feats)
        y_all = torch.cat(targets)
        # Prefer scikit-learn array-API dispatch (keeps the torch tensors
        # on-device, no numpy copy); fall back to numpy if the env doesn't
        # have scipy's array-API support enabled.
        try:
            with config_context(array_api_dispatch=True):
                self.head.fit(X_feat, y_all)
            self._array_api = True
        except (RuntimeError, ValueError):
            self.head.fit(to_numpy(X_feat), to_numpy(y_all))
            self._array_api = False
        return self

    def _head_predict(self, X_feat):
        if self._array_api:
            with config_context(array_api_dispatch=True):
                return torch.as_tensor(self.head.predict(X_feat))
        return torch.as_tensor(to_numpy(self.head.predict(to_numpy(X_feat))))

    def predict(self, X):
        emb = torch.as_tensor(self.encoder.encode(X))    # (B, T', D)
        if self.mode == "epoched":
            return self._head_predict(emb.mean(dim=1))   # (B,)
        B, t_prime, D = emb.shape
        pred = self._head_predict(emb.reshape(-1, D)).reshape(B, t_prime)
        return _resample(pred, X.shape[-1])              # (B, T_in)
