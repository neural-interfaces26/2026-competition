"""Frozen-encoder linear probe with a scikit-learn head.

Shared infrastructure for frozen-encoder baselines (the per-track REVE
probes). A frozen *encoder* maps a batch of windows to a temporal embedding::

    encode(X: (B, C, T)) -> (B, T', D)

i.e. one ``D``-dim vector per output time position ``T'`` (no pooling). The
encoder returns **torch tensors** (so braindecode foundation models run
directly). Mean-pooled features cross once to CPU/NumPy at the scikit-learn
boundary. This keeps the exported joblib head portable across CPU and GPU
machines and works with ordinary scikit-learn estimators.

One probe style serves both regimes:

- ``epoched`` : mean-pool over ``T'`` → one target per window.
- ``dense``   : keep every position; the head maps ``(B, T', D) -> (B, T',
                *)``; predictions are upsampled back to the raw ``T``.

The encoder is frozen and never sees the labels. Solvers define their own
encoder (e.g. a pretrained REVE) and reuse :class:`Encoder` /
:class:`LinearProbe` from here.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.data import resample_labels, to_numpy


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
    """Dependency-light temporal encoder for probe contract tests.

    This is an explicit test utility, not a silent fallback for a missing
    foundation model. It splits a window into patches and projects each patch
    into a deterministic embedding.
    """

    def __init__(self, n_chans, patch_len=20, d=64, seed=0):
        self.n_chans = n_chans
        self.patch_len = patch_len
        self.d = d
        self._gen = torch.Generator().manual_seed(seed)
        self._proj = None

    def _projection(self, in_dim):
        if self._proj is None or self._proj.shape[0] != in_dim:
            self._proj = torch.randn(
                in_dim, self.d, generator=self._gen
            ) / np.sqrt(in_dim)
        return self._proj

    def encode(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        if X.ndim == 2:
            X = X[None]
        batch, n_chans, n_times = X.shape
        n_patches = max(n_times // self.patch_len, 1)
        usable = n_patches * self.patch_len
        patches = X[:, :, :usable].reshape(
            batch, n_chans, n_patches, self.patch_len
        )
        patches = patches.permute(0, 2, 1, 3).reshape(
            batch, n_patches, -1
        )
        return patches @ self._projection(patches.shape[-1]).to(X.device)


class LinearProbe:
    """Frozen encoder + scikit-learn linear head.

    Parameters
    ----------
    encoder : Encoder
        Frozen feature extractor (``encode(X) -> (B, T', D)``).
    estimator : scikit-learn estimator, optional
        The linear head, wrapped in a ``StandardScaler`` pipeline. Defaults
        to ``RidgeClassifier``; pass ``Ridge()`` for scalar or multi-output
        regression.
    mode : {"epoched", "dense"}
        ``epoched`` pools over ``T'`` and predicts one target per window;
        ``dense`` fits the head over every position and predicts a per-step
        sequence (upsampled to the input length on ``predict``).
    """

    def __init__(self, encoder, estimator=None, mode="epoched"):
        if mode not in ("epoched", "dense"):
            raise ValueError(f"mode must be 'epoched' or 'dense', got {mode!r}")
        self.encoder = encoder
        self.mode = mode
        self.head = make_pipeline(
            StandardScaler(),
            estimator if estimator is not None else RidgeClassifier(),
        )

    def fit(self, train_loader):
        feats, targets = [], []
        for X, y, _info in train_loader:
            emb = torch.as_tensor(self.encoder.encode(X))  # (B, T', D)
            if emb.ndim != 3:
                raise ValueError(
                    "encoder must return (B, T', D), got "
                    f"shape {tuple(emb.shape)}"
                )
            if self.mode == "epoched":
                feats.append(to_numpy(emb.mean(dim=1)))  # (B, D)
                targets.append(to_numpy(y))              # (B,) or (B, K)
            else:  # dense
                y_ds = resample_labels(y, emb.shape[1])  # (B, T')
                feats.append(to_numpy(emb.reshape(-1, emb.shape[-1])))
                targets.append(to_numpy(y_ds.reshape(-1)))
        self.head.fit(np.concatenate(feats), np.concatenate(targets))
        return self

    def predict(self, X):
        emb = torch.as_tensor(self.encoder.encode(X))    # (B, T', D)
        if self.mode == "epoched":
            return self.head.predict(to_numpy(emb.mean(dim=1)))
        batch, t_prime, dim = emb.shape
        pred = self.head.predict(
            to_numpy(emb.reshape(-1, dim))
        ).reshape(batch, t_prime)
        return resample_labels(pred, X.shape[-1])        # (B, T_in)
