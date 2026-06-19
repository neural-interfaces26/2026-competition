"""Built-in baseline models used as fallbacks and for testing.

These keep the benchmark runnable on its own (``benchopt run benchmark/``)
without any participant submission and without the heavy EEG stack:

- :func:`default_encoder` — a frozen :class:`RandomProjectionEncoder` for the
  **linear-probe track** when no ``submission.get_encoder`` is importable.
- :class:`DefaultSpecificModel` — a simple scikit-learn baseline for the
  **specific track** (``fit(loader)`` / ``predict((B, C, T))``), handling both
  the epoched and dense regimes.
- :class:`ConstantModel` — a trivial model for ``Objective.get_one_result``.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.data import to_numpy
from benchmark_utils.linear_probe import RandomProjectionEncoder


def default_encoder(meta):
    """Fallback frozen encoder for the linear-probe track."""
    return RandomProjectionEncoder(n_chans=meta.get("n_chans", 1))


class DefaultSpecificModel:
    """Dependency-light specialist baseline for the specific track.

    - ``epoched`` : mean over time per window → LogReg → one label/window.
    - ``dense``   : per-time-step features (channel values) → LogReg →
                    one label per time-step.
    """

    def __init__(self, task_kind="epoched"):
        self.task_kind = task_kind
        self.clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        )

    def _features_epoched(self, X):
        # (B, C, T) -> (B, C) mean over time
        return to_numpy(X).mean(axis=-1)

    def _features_dense(self, X):
        # (B, C, T) -> (B*T, C) one sample per time-step
        X = to_numpy(X)
        B, C, T = X.shape
        return X.transpose(0, 2, 1).reshape(B * T, C), (B, T)

    def fit(self, train_loader):
        feats, targets = [], []
        for X, y, _info in train_loader:
            y = to_numpy(y)
            if self.task_kind == "epoched":
                feats.append(self._features_epoched(X))
                targets.append(y)
            else:
                f, _ = self._features_dense(X)
                feats.append(f)
                targets.append(y.reshape(-1))
        self.clf.fit(np.concatenate(feats), np.concatenate(targets))
        return self

    def predict(self, X):
        if self.task_kind == "epoched":
            return self.clf.predict(self._features_epoched(X))
        f, (B, T) = self._features_dense(X)
        return self.clf.predict(f).reshape(B, T)


class ConstantModel:
    """Always predict class 0, with a shape matching the task regime."""

    def __init__(self, task_kind="epoched"):
        self.task_kind = task_kind

    def fit(self, train_loader):
        return self

    def predict(self, X):
        X = to_numpy(X)
        B, _C, T = X.shape
        if self.task_kind == "epoched":
            return np.zeros(B, dtype=np.int64)
        return np.zeros((B, T), dtype=np.int64)
