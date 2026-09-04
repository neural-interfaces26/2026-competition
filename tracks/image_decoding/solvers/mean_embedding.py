"""Reference floor: always predict the mean training embedding.

The retrieval equivalent of a chance model — every window gets the same
prediction, so ranking is uninformative. Also the minimal example of the
submission contract on this track.
"""

import benchmark_utils  # noqa: F401 — locates compet_core
from compet_core.base_solver import CompetSolver
from compet_core.baselines import ConstantEmbedder


class Solver(CompetSolver):

    name = "MeanEmbedding"

    def load_model(self, meta):
        return ConstantEmbedder(meta["n_outputs"])

    def fit(self, model, train_loader):
        model.fit(train_loader)
