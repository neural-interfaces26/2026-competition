"""Built-in baseline models used as reference solvers and for testing.

These keep every track runnable on its own (``benchopt run tracks/<t>``)
without any participant submission and without the heavy EEG stack:

- :class:`MeanLogReg` — a simple scikit-learn classification baseline
  (``fit(loader)`` / ``predict((B, C, T)) -> (B,)``).
- :class:`ConstantClassifier` — a trivial classifier, used by the
  classification objectives' ``get_one_result``.
"""

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.data import to_numpy


class MeanLogReg:
    """Dependency-light classification baseline.

    Mean over time per window ``(B, C, T) -> (B, C)`` → logistic regression
    → one label per window.
    """

    def __init__(self):
        self.clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        )

    def _features(self, X):
        return to_numpy(X).mean(axis=-1)

    def fit(self, train_loader):
        feats, targets = [], []
        for X, y, _info in train_loader:
            feats.append(self._features(X))
            targets.append(to_numpy(y))
        self.clf.fit(np.concatenate(feats), np.concatenate(targets))
        return self

    def predict(self, X):
        feats = self._features(X)
        try:
            labels = self.clf.predict(feats)
        except NotFittedError:
            # Inference-only run: no fit, so fall back to the floor class
            # like the constant baselines rather than raising.
            labels = np.zeros(len(feats), dtype=np.int64)
        return labels


class ConstantClassifier:
    """Always predict class 0 — one label per window."""

    def fit(self, train_loader):
        return self

    def predict(self, X):
        return np.zeros(len(to_numpy(X)), dtype=np.int64)


class ConstantEmbedder:
    """Always predict the same embedding — used as retrieval floor/test."""

    def __init__(self, n_outputs):
        self.n_outputs = n_outputs
        self.value = np.ones(n_outputs, dtype=np.float32)

    def fit(self, train_loader):
        # Mean training embedding (a slightly-better-than-arbitrary constant).
        total, count = 0.0, 0
        for _X, y, _info in train_loader:
            y = to_numpy(y)
            total = total + y.sum(axis=0)
            count += len(y)
        if count:
            self.value = (total / count).astype(np.float32)
        return self

    def predict(self, X):
        return np.tile(self.value, (len(to_numpy(X)), 1))


class ConstantPose:
    """Always predict the train-mean pose, constant over time.

    ``predict((B, C, T)) -> (B, n_joints, T)`` — every window gets the same
    per-joint angle vector, broadcast along time.
    """

    def __init__(self, n_joints):
        self.value = np.zeros(n_joints, dtype=np.float32)

    def fit(self, train_loader):
        total, count = 0.0, 0
        for _X, y, _info in train_loader:
            y = to_numpy(y)                       # (B, J, T)
            total = total + y.mean(axis=-1).sum(axis=0)
            count += len(y)
        if count:
            self.value = (total / count).astype(np.float32)
        return self

    def predict(self, X):
        B, _C, T = to_numpy(X).shape
        return np.broadcast_to(
            self.value[None, :, None], (B, len(self.value), T)
        ).copy()


class MedianRegressor:
    """Always predict the median of the train targets — one value/window."""

    def __init__(self, value=0.0):
        self.value = value

    def fit(self, train_loader):
        targets = [to_numpy(y).ravel() for _X, y, _info in train_loader]
        self.value = float(np.median(np.concatenate(targets)))
        return self

    def predict(self, X):
        return np.full(len(to_numpy(X)), self.value, dtype=np.float64)
