"""Reference floor: always predict the train-mean pose.

Dependency-light floor for the leaderboard, and the minimal example of the
submission contract on this track.
"""

from compet_core.base_solver import CompetSolver
from compet_core.baselines import ConstantPose


class Solver(CompetSolver):

    name = "MeanPose"

    def load_model(self, meta):
        return ConstantPose(meta["n_joints"])

    def fit(self, model, train_loader):
        model.fit(train_loader)
