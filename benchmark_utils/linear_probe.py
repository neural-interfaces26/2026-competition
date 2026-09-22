"""Frozen-encoder linear probe with an on-device scikit-learn head.

Shared infrastructure for frozen-encoder baselines (the per-track REVE
probes). A frozen *encoder* maps a batch of windows to a temporal embedding::

    encode(X: (B, C, T)) -> (B, T', D)

i.e. one ``D``-dim vector per output time position ``T'`` (no pooling). The
encoder receives **torch tensors** (so braindecode foundation models run
directly), and thanks to scikit-learn's **array API** support the torch
features can be passed straight to the linear head with no numpy round-trip
(when scipy's array-API support is enabled — see the env var below —
otherwise the head transparently falls back to numpy). One probe style serves
both regimes:

- ``epoched`` : mean-pool over ``T'`` → one target per window.
- ``dense``   : keep every position; the head maps ``(B, T', D) -> (B, T',
                *)``; predictions are upsampled back to the raw ``T``.

The encoder is frozen and never sees the labels. Solvers define their own
encoder (e.g. a pretrained REVE) and reuse :class:`Encoder` /
:class:`LinearProbe` from here.
"""

import os
# scikit-learn array-API dispatch requires scipy's array-API support, which
# is gated behind this env var and must be set before scipy is imported.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

from abc import ABC, abstractmethod  # noqa: E402

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
    estimator : scikit-learn estimator, optional
        The linear head, wrapped in a ``StandardScaler`` pipeline. Defaults to
        ``LogisticRegression`` (classification); pass ``Ridge()`` for
        regression / multi-output embedding targets.
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
            estimator if estimator is not None
            else LogisticRegression(max_iter=1000),
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
        X_feat = torch.cat(feats)
        y_all = torch.cat(targets)
        # Prefer array-API dispatch (torch tensors stay on-device, no numpy
        # copy); fall back to numpy if scipy's array-API support is off.
        try:
            with config_context(array_api_dispatch=True):
                self.head.fit(X_feat, y_all)
        except (RuntimeError, ValueError, TypeError):
            self.head.fit(to_numpy(X_feat), to_numpy(y_all))
        return self

    def _fit_used_torch(self):
        # StandardScaler.mean_ is a torch tensor iff the head was fit under
        # array-API dispatch; mirror that at predict so params and input share
        # a namespace (also correct for a joblib-reloaded head).
        return isinstance(getattr(self.head[0], "mean_", None), torch.Tensor)

    def _head_predict(self, X_feat):
        if self._fit_used_torch():
            with config_context(array_api_dispatch=True):
                return torch.as_tensor(self.head.predict(X_feat))
        return torch.as_tensor(to_numpy(self.head.predict(to_numpy(X_feat))))

    def predict(self, X):
        emb = torch.as_tensor(self.encoder.encode(X))    # (B, T', D)
        if self.mode == "epoched":
            return self._head_predict(emb.mean(dim=1))   # (B,) or (B, K)
        B, t_prime, D = emb.shape
        pred = self._head_predict(emb.reshape(-1, D)).reshape(B, t_prime)
        return _resample(pred, X.shape[-1])              # (B, T_in)
