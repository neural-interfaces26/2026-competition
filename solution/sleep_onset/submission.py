"""Sample submission for the sleep-onset track.

Demonstrates the full contract with a trivial constant-latency regressor:

- submitted as-is, it **loads** its latency from ``weights.npz`` when one is
  shipped alongside, and otherwise uses an in-memory stand-in value so the
  sample also works from Codabench's read-only submission directory;
- run with ``-o "Sleep-onset[training=True]"``, ``fit`` recomputes it on the
  train split and ``save_model`` writes it — the run then drops a
  ready-to-upload ``outputs/submission_Sample-Sleep.zip``.
"""

import numpy as np

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.baselines import MedianRegressor


class Solver(CompetSolver):

    name = "Sample-Sleep"

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.npz"
        value = (float(np.load(weights)["latency"])
                 if weights.exists() else 300.0)
        return MedianRegressor(value=value)

    def fit(self, model, train_loader):
        model.fit(train_loader)                       # median train latency

    def save_model(self, model, path):
        # Same name ``load_model`` reads from ``meta["submission_dir"]``.
        np.savez(path / "weights.npz", latency=np.float64(model.value))
