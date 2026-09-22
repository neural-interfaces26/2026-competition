"""Smoke-test submission for the EMG-to-pose track.

This deliberately simple constant-pose predictor validates ingestion, data
loading, inference, scoring, and leaderboard publication. It loads locally
trained joint angles from ``weights.npz`` when provided and otherwise uses
the baseline's in-memory zero pose. The optional ``fit`` and ``save_model``
hooks support local training and export.

See "Submission Guide > Practice 2: Train and package with Benchopt" to train
this model on actual data to produce weights.npz with benchopt.
"""

import numpy as np

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.baselines import ConstantPose


class Solver(CompetSolver):

    name = "Sample-EMG"

    def load_model(self, meta):
        model = ConstantPose(meta["n_joints"])
        weights = meta["submission_dir"] / "weights.npz"
        if weights.exists():                          # written by save_model
            model.value = np.load(weights)["pose"]
        return model

    def fit(self, model, train_loader):
        model.fit(train_loader)                       # mean train pose

    def save_model(self, model, path):
        # Same name ``load_model`` reads from ``meta["submission_dir"]``.
        np.savez(path / "weights.npz", pose=model.value)
