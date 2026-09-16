"""Sample submission for the EMG-to-pose track.

Demonstrates the full contract with a trivial linear read-out:

- submitted as-is, it **loads** its read-out from ``weights.npz`` shipped
  alongside (the platform runs inference-only — a stand-in file is generated
  when missing, so the sample always runs);
- run with ``-o "EMG-pose[training=True]"``, ``fit`` fits it on the train
  split and ``save_model`` writes it — the run then drops a ready-to-upload
  ``outputs/submission_Sample-EMG.zip``.
"""

import numpy as np

import benchmark_utils  # noqa: F401 — locates compet_core
from compet_core.base_solver import CompetSolver
from compet_core.data import resample_labels, to_numpy


class LinearPose:

    def __init__(self, readout):
        self.readout = readout                        # (J, C)

    def predict(self, X):
        X = to_numpy(X)                               # (B, C, T)
        return np.einsum("jc,bct->bjt", self.readout, X)


class Solver(CompetSolver):

    name = "Sample-EMG"

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.npz"
        if not weights.exists():                      # stand-in artefact
            rng = np.random.default_rng(0)
            np.savez(weights, readout=rng.standard_normal(
                (meta["n_joints"], meta["n_chans"])) * 0.1)
        return LinearPose(np.load(weights)["readout"])

    def fit(self, model, train_loader):
        # Least squares per time-step, on angles resampled to the EMG rate.
        emg, angles = [], []
        for X, y, _info in train_loader:
            X, y = to_numpy(X), to_numpy(y)
            y = resample_labels(y, X.shape[-1])
            emg.append(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
            angles.append(y.transpose(0, 2, 1).reshape(-1, y.shape[1]))
        readout = np.linalg.lstsq(
            np.concatenate(emg), np.concatenate(angles), rcond=None)[0]
        model.readout = readout.T                     # (J, C)

    def save_model(self, model, path):
        # Same name ``load_model`` reads from ``meta["submission_dir"]``.
        np.savez(path / "weights.npz", readout=model.readout)
