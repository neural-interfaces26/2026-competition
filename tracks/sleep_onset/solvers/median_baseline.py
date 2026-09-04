"""Reference baseline: always predict the train-split median latency.

Dependency-light floor for the leaderboard, and the minimal example of the
submission contract on a regression track.
"""

from compet_core.base_solver import CompetSolver
from compet_core.baselines import MedianRegressor


class Solver(CompetSolver):

    name = "Median"

    def load_model(self, meta):
        return MedianRegressor()

    def fit(self, model, train_loader):
        model.fit(train_loader)
