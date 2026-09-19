"""Smoke-test submission for the EEG-to-image track.

This deliberately simple constant-embedding predictor validates ingestion,
data loading, inference, scoring, and leaderboard publication. It loads a
locally trained embedding from ``weights.npz`` when provided and otherwise
uses the baseline's in-memory constant. The optional ``fit`` and
``save_model`` hooks support local training and export.
"""

import numpy as np

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.baselines import ConstantEmbedder


class Solver(CompetSolver):

    name = "Sample-Image"

    def load_model(self, meta):
        model = ConstantEmbedder(meta["n_outputs"])
        weights = meta["submission_dir"] / "weights.npz"
        if weights.exists():                          # written by save_model
            model.value = np.load(weights)["embedding"]
        return model

    def fit(self, model, train_loader):
        model.fit(train_loader)                       # mean train embedding

    def save_model(self, model, path):
        # Same name ``load_model`` reads from ``meta["submission_dir"]``.
        np.savez(path / "weights.npz", embedding=model.value)
