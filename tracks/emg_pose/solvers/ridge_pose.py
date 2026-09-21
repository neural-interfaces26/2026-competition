"""Reference baseline: per-time-step ridge regression to joint angles.

The scikit-learn slot for this track. The Simulated EMG is an instantaneous
linear mixture of the joint angles, so a linear readout applied independently
at each time step recovers the pose: the channel vector at time ``t`` maps to
the joint-angle vector at ``t``. Windows flatten to ``(B*T, C)`` for a
multi-output ridge and the prediction reshapes back to ``(B, n_joints, T)``.
Weights travel as a joblib dump, so this is also the example for a submission
whose model is not a torch module.
"""

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


class TimeRidge:
    """Per-time-step ridge: ``(B, C, T) -> (B, n_joints, T)``."""

    def __init__(self, n_joints, estimator=None):
        self.n_joints = n_joints
        # A loaded estimator arrives fitted; a fresh pipeline does not.
        self.fitted = estimator is not None
        self.clf = estimator or make_pipeline(StandardScaler(), Ridge())

    def _features(self, X):
        # (B, C, T) -> (B*T, C): one sample per time step.
        X = to_numpy(X)
        return np.moveaxis(X, 1, -1).reshape(-1, X.shape[1])

    def fit(self, train_loader):
        feats, targets = [], []
        for X, y, _info in train_loader:
            feats.append(self._features(X))
            y = to_numpy(y)                                 # (B, J, T)
            targets.append(np.moveaxis(y, 1, -1).reshape(-1, y.shape[1]))
        self.clf.fit(np.concatenate(feats), np.concatenate(targets))
        self.fitted = True
        return self

    def predict(self, X):
        X = to_numpy(X)
        B, _C, T = X.shape
        if not self.fitted:
            # Untrained (no shipped weights): a zero pose, rather than raise.
            return np.zeros((B, self.n_joints, T), dtype=np.float32)
        pred = self.clf.predict(np.moveaxis(X, 1, -1).reshape(-1, X.shape[1]))
        return np.moveaxis(pred.reshape(B, T, self.n_joints), -1, 1)


class Solver(CompetSolver):

    name = "Ridge"

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists():
            print(f"[loading] {weights} into {self.name}")
            return TimeRidge(meta["n_joints"], estimator=joblib.load(weights))
        return TimeRidge(meta["n_joints"])

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.clf, path / "weights.joblib")
