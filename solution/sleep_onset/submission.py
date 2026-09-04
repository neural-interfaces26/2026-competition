"""Sample submission for the sleep-onset track.

Demonstrates the contract: ship a **fully trained** model — the platform runs
inference-only (``fit`` never runs there). This trivial example predicts a
constant latency loaded from ``weights.npz`` shipped alongside (standing in
for your real training artefacts).
"""

import numpy as np

from compet_core.base_solver import CompetSolver
from compet_core.baselines import MedianRegressor


class Solver(CompetSolver):

    name = "Sample-Sleep"

    def load_model(self, meta):
        weights = meta["weights_dir"] / "weights.npz"
        if not weights.exists():                      # stand-in artefact
            np.savez(weights, latency=np.float64(300.0))
        return MedianRegressor(value=float(np.load(weights)["latency"]))
