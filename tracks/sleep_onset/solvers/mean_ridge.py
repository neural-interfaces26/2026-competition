"""Reference baseline: ridge regression on the per-channel time means.

The scikit-learn counterpart of ``torch_linear``: each window collapses to
one value per channel, ``(B, C, T) -> (B, C)``, and a linear model maps those
to a latency. Weights travel as a joblib dump, so this is also the example
for a submission whose model is not a torch module.
"""

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy

CAP_S = 600.0


class MeanRidge:
    """Per-channel time means -> ridge regression -> one latency."""

    def __init__(self, estimator=None):
        # A loaded estimator arrives fitted; a fresh pipeline does not.
        self.fitted = estimator is not None
        self.clf = estimator or make_pipeline(StandardScaler(), Ridge())

    def _features(self, X):
        return to_numpy(X).mean(axis=-1)

    def fit(self, train_loader):
        feats, targets = [], []
        for X, y, _info in train_loader:
            feats.append(self._features(X))
            targets.append(to_numpy(y).ravel())
        self.clf.fit(np.concatenate(feats), np.concatenate(targets))
        self.fitted = True
        return self

    def predict(self, X):
        feats = self._features(X)
        if not self.fitted:
            # Inference-only run with no shipped weights: predict the floor
            # rather than raise.
            return np.full(len(feats), CAP_S / 2, dtype=np.float64)
        return np.clip(self.clf.predict(feats), 0.0, CAP_S)


class Solver(CompetSolver):

    name = "Mean-Ridge"

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists():
            return MeanRidge(estimator=joblib.load(weights))
        return MeanRidge()

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.clf, path / "weights.joblib")
