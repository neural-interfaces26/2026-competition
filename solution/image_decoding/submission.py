"""Sample submission for the image-decoding track.

Demonstrates the contract: ship a **fully trained** model — the platform runs
inference-only (``fit`` never runs there). This trivial example projects the
window channel means to the embedding space with a fixed random map loaded
from ``weights.npz`` (standing in for your real training artefacts).

``solution/bci_decoding/submission.py`` shows the optional
``fit``/``save_model`` pair: local training plus a ready-to-upload
artifact.
"""

import numpy as np

import benchmark_utils  # noqa: F401 — locates compet_core
from compet_core.base_solver import CompetSolver
from compet_core.data import to_numpy


class LinearEmbedder:

    def __init__(self, proj):
        self.proj = proj                              # (C, D)

    def predict(self, X):
        feats = to_numpy(X).mean(axis=-1)             # (B, C)
        return feats @ self.proj                      # (B, D)


class Solver(CompetSolver):

    name = "Sample-Image"

    def load_model(self, meta):
        weights = meta["weights_dir"] / "weights.npz"
        if not weights.exists():                      # stand-in artefact
            rng = np.random.default_rng(0)
            np.savez(weights, proj=rng.standard_normal(
                (meta["n_chans"], meta["n_outputs"])))
        return LinearEmbedder(np.load(weights)["proj"])
