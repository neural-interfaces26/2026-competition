"""Frozen-encoder linear probe with an on-device scikit-learn head.

Shared infrastructure for frozen-encoder baselines (the per-track REVE
probes). A frozen *encoder* maps a batch of windows to a temporal embedding::

    encode(X: (B, C, T)) -> (B, T', D)

i.e. one ``D``-dim vector per output time position ``T'`` (no pooling). The
encoder returns **torch tensors** (so braindecode foundation models run
directly), and the linear head consumes them **in place** via scikit-learn's
array-API dispatch — the features never leave the encoder's device.

Array-API dispatch requires ``SCIPY_ARRAY_API=1`` in the environment *before
scipy is imported*; the competition worker image sets it. Otherwise
scikit-learn raises telling you to set it (we deliberately do not silently
copy to numpy — that would hide that the on-device path is off). Only
array-API-aware heads work here: ``Ridge`` / ``RidgeClassifier`` /
``LinearDiscriminantAnalysis`` (``LogisticRegression`` does not dispatch).

One probe style serves both regimes:

- ``epoched`` : mean-pool over ``T'`` → one target per window.
- ``dense``   : keep every position; the head maps ``(B, T', D) -> (B, T',
                *)``; predictions are upsampled back to the raw ``T``.

The encoder is frozen and never sees the labels. Solvers define their own
encoder (e.g. a pretrained REVE) and reuse :class:`Encoder` /
:class:`LinearProbe` from here.
"""

from abc import ABC, abstractmethod

import torch
from sklearn import config_context
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Encoder(ABC):
    """Frozen feature extractor returning a temporal embedding.

    Subclasses implement :meth:`encode`, receiving a torch tensor
    ``(B, C, T)`` and returning ``(B, T', D)``. The time axis ``T'`` is kept
    so the same encoder serves window-classification and dense labelling.
    """

    @abstractmethod
    def encode(self, X):
        """Map ``(B, C, T)`` to a temporal embedding ``(B, T', D)``."""


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
    """Frozen encoder + per-position scikit-learn linear head (on-device).

    Parameters
    ----------
    encoder : Encoder
        Frozen feature extractor (``encode(X) -> (B, T', D)``).
    estimator : scikit-learn estimator, optional
        The linear head, wrapped in a ``StandardScaler`` pipeline. Must be
        array-API-aware (see the module docstring). Defaults to
        ``RidgeClassifier``; pass ``Ridge()`` for regression / multi-output
        embedding targets.
    mode : {"epoched", "dense"}
        ``epoched`` pools over ``T'`` and predicts one target per window;
        ``dense`` fits the head over every position and predicts a per-step
        sequence (upsampled to the input length on ``predict``).
    """

    def __init__(self, encoder, estimator=None, mode="epoched"):
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
            if self.mode == "epoched":
                feats.append(emb.mean(dim=1))            # (B, D)
                targets.append(y)                        # (B,) or (B, K)
            else:  # dense
                y_ds = _resample(y, emb.shape[1])        # (B, T')
                feats.append(emb.reshape(-1, emb.shape[-1]))
                targets.append(y_ds.reshape(-1))
        # Fit in the features' torch namespace: the tensors stay on-device.
        # Needs SCIPY_ARRAY_API=1 (set in the image); scikit-learn raises with
        # instructions if it is not.
        with config_context(array_api_dispatch=True):
            self.head.fit(torch.cat(feats), torch.cat(targets))
        return self

    def predict(self, X):
        emb = torch.as_tensor(self.encoder.encode(X))    # (B, T', D)
        with config_context(array_api_dispatch=True):
            if self.mode == "epoched":
                return torch.as_tensor(self.head.predict(emb.mean(dim=1)))
            B, t_prime, D = emb.shape
            pred = torch.as_tensor(
                self.head.predict(emb.reshape(-1, D))
            ).reshape(B, t_prime)
        return _resample(pred, X.shape[-1])              # (B, T_in)
