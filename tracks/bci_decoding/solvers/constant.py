"""Reference floor: always predict a single class.

Dependency-light floor for the leaderboard, and the minimal example of the
submission contract on a classification track — no weights, no training.
Balanced accuracy lands at chance (``1 / n_classes``).
"""

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.baselines import ConstantClassifier


class Solver(CompetSolver):

    name = "Constant"

    def load_model(self, meta):
        return ConstantClassifier()

    def fit(self, model, train_loader):
        model.fit(train_loader)
