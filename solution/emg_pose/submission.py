"""Sample submission for the EMG-to-pose track.

Demonstrates the full contract with the shared ``ConstantPose`` baseline:

- submitted as-is, it **loads** its per-joint angles from ``weights.npz``
  shipped alongside (the platform runs inference-only);
- run with ``-o "EMG-pose[training=True]"``, ``fit`` recomputes them on the
  train split and ``save_model`` writes them — the run then drops a
  ready-to-upload ``outputs/Sample-EMG/`` folder.
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
