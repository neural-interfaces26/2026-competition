"""Smoke-test submission for the sleep-onset track.

This deliberately simple constant predictor validates ingestion, real-data
loading, inference, scoring, and leaderboard publication. It loads a locally
trained latency from ``weights.npz`` when provided and otherwise uses an
in-memory fallback. The optional ``fit`` and ``save_model`` hooks support
local training and export. A full PyTorch code-and-weights example lives in
``examples/sleep_onset/minimal_cnn``.
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
