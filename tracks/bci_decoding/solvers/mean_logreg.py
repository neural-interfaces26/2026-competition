"""Reference baseline: mean-over-time features + logistic regression.

Dependency-light floor for the leaderboard, and the minimal example of the
submission contract: subclass
:class:`~benchmark_utils.base_solver.CompetSolver`,
build the model in ``load_model``, train it in ``fit``.
"""

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.baselines import MeanLogReg


class Solver(CompetSolver):

    name = "MeanLogReg"

    def load_model(self, meta):
        return MeanLogReg()

    def fit(self, model, train_loader):
        model.fit(train_loader)
