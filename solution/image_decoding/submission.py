"""Sample submission for the image-decoding track.

Demonstrates the full contract with the shared ``ConstantEmbedder`` baseline:

- submitted as-is, it **loads** its embedding from ``weights.npz`` shipped
  alongside (the platform runs inference-only);
- run with ``-o "Image-decoding[training=True]"``, ``fit`` recomputes it on
  the train split and ``save_model`` writes it — the run then drops a
  ready-to-upload ``outputs/Sample-Image/`` folder.
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
