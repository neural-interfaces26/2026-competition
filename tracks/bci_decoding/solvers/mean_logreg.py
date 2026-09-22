"""Reference baseline: mean-over-time features + logistic regression.

The scikit-learn slot for this track: each window collapses to one value per
channel, ``(B, C, T) -> (B, C)``, and logistic regression maps those to a
class. Weights travel as a joblib dump, so this is also the example for a
submission whose model is not a torch module.
"""

import joblib

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.baselines import MeanLogReg


class Solver(CompetSolver):

    name = "MeanLogReg"

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists():
            print(f"[loading] {weights} into {self.name}")
            return MeanLogReg(estimator=joblib.load(weights))
        return MeanLogReg()

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.clf, path / "weights.joblib")
