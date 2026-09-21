"""Sample submission for the sleep-onset track.

Demonstrates the full contract with a trivial constant-latency regressor:

- submitted as-is, it **loads** its latency from ``weights.npz`` shipped
  alongside (the platform runs inference-only — a stand-in file is generated
  when missing, so the sample always runs);
- run with ``-o "Sleep-onset[training=True]"``, ``fit`` recomputes it on the
  train split and ``save_model`` writes it — the run then drops a
  ready-to-upload ``outputs/Sample-Sleep/`` folder.
"""

import numpy as np

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.baselines import MedianRegressor


class Solver(CompetSolver):

    name = "Sample-Sleep"

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.npz"
        if not weights.exists():                      # stand-in artefact
            np.savez(weights, latency=np.float64(300.0))
        return MedianRegressor(value=float(np.load(weights)["latency"]))

    def fit(self, model, train_loader):
        model.fit(train_loader)                       # median train latency

    def save_model(self, model, path):
        # Same name ``load_model`` reads from ``meta["submission_dir"]``.
        np.savez(path / "weights.npz", latency=np.float64(model.value))
