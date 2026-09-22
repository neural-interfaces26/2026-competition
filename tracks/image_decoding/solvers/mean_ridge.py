"""Reference baseline: ridge regression to the target embedding space.

The scikit-learn slot for this track and the retrieval counterpart of the
sleep track's ``Mean-Ridge``: each window collapses to one value per channel,
``(B, C, T) -> (B, C)``, and a multi-output ridge maps those to a ``D``-dim
embedding the objective ranks by cosine similarity. Weights travel as a
joblib dump, so this is also the example for a submission whose model is not
a torch module.
"""

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


class MeanRidge:
    """Per-channel time means -> ridge regression -> an embedding (B, D)."""

    def __init__(self, n_outputs, estimator=None):
        self.n_outputs = n_outputs
        # A loaded estimator arrives fitted; a fresh pipeline does not.
        self.fitted = estimator is not None
        self.clf = estimator or make_pipeline(StandardScaler(), Ridge())

    def _features(self, X):
        return to_numpy(X).mean(axis=-1)

    def fit(self, train_loader):
        feats, targets = [], []
        for X, y, _info in train_loader:
            feats.append(self._features(X))
            targets.append(to_numpy(y))
        self.clf.fit(np.concatenate(feats), np.concatenate(targets))
        self.fitted = True
        return self

    def predict(self, X):
        feats = self._features(X)
        if not self.fitted:
            # Inference-only run with no shipped weights: a constant
            # embedding (uninformative ranking), like the other untrained
            # baselines, rather than raise.
            return np.ones((len(feats), self.n_outputs), dtype=np.float32)
        return self.clf.predict(feats)


class Solver(CompetSolver):

    name = "Mean-Ridge"

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists():
            print(f"[loading] {weights} into {self.name}")
            return MeanRidge(meta["n_outputs"],
                             estimator=joblib.load(weights))
        return MeanRidge(meta["n_outputs"])

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.clf, path / "weights.joblib")
