"""Built-in baseline models used as reference solvers and for testing.

These keep every track runnable on its own (``benchopt run tracks/<t>``)
without any participant submission and without the heavy EEG stack:

- :class:`MeanLogReg` — a simple scikit-learn classification baseline
  (``fit(loader)`` / ``predict((B, C, T)) -> (B,)``).
- :class:`ConstantClassifier` — a trivial classifier, used by the
  classification objectives' ``get_one_result``.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compet_core.data import to_numpy


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
        return self.clf.predict(self._features(X))


class ConstantClassifier:
    """Always predict class 0 — one label per window."""

    def fit(self, train_loader):
        return self

    def predict(self, X):
        return np.zeros(len(to_numpy(X)), dtype=np.int64)
