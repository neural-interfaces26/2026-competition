"""Reference baseline: always predict the train-split median latency.

Dependency-light floor for the leaderboard, and the minimal example of the
submission contract on a regression track.
"""

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.baselines import MedianRegressor


class Solver(CompetSolver):

    name = "Median"

    def load_model(self, meta):
        return MedianRegressor()

    def fit(self, model, train_loader):
        model.fit(train_loader)
