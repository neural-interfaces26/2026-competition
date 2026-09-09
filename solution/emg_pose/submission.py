"""Sample submission for the EMG-to-pose track.

Demonstrates the contract: ship a **fully trained** model — the platform runs
inference-only (``fit`` never runs there). This trivial example maps each EMG
time-step to joint angles with a fixed linear read-out loaded from
``weights.npz`` shipped alongside (standing in for your real artefacts).
"""

import numpy as np

import benchmark_utils  # noqa: F401 — locates compet_core
from compet_core.base_solver import CompetSolver
from compet_core.data import to_numpy


class LinearPose:

    def __init__(self, readout):
        self.readout = readout                        # (J, C)

    def predict(self, X):
        X = to_numpy(X)                               # (B, C, T)
        return np.einsum("jc,bct->bjt", self.readout, X)


class Solver(CompetSolver):

    name = "Sample-EMG"

    def load_model(self, meta):
        weights = meta["weights_dir"] / "weights.npz"
        if not weights.exists():                      # stand-in artefact
            rng = np.random.default_rng(0)
            np.savez(weights, readout=rng.standard_normal(
                (meta["n_joints"], meta["n_chans"])) * 0.1)
        return LinearPose(np.load(weights)["readout"])
