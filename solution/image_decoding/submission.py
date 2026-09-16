"""Sample submission for the image-decoding track.

Demonstrates the full contract with a trivial linear read-out:

- submitted as-is, it **loads** its projection from ``weights.npz`` shipped
  alongside (the platform runs inference-only — a stand-in file is generated
  when missing, so the sample always runs);
- run with ``-o "Image-decoding[training=True]"``, ``fit`` fits it on the
  train split and ``save_model`` writes it — the run then drops a
  ready-to-upload ``outputs/submission_Sample-Image.zip``.
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
        weights = meta["submission_dir"] / "weights.npz"
        if not weights.exists():                      # stand-in artefact
            rng = np.random.default_rng(0)
            np.savez(weights, proj=rng.standard_normal(
                (meta["n_chans"], meta["n_outputs"])))
        return LinearEmbedder(np.load(weights)["proj"])

    def fit(self, model, train_loader):
        # Least squares from the window channel means to the embeddings.
        feats, targets = [], []
        for X, y, _info in train_loader:
            feats.append(to_numpy(X).mean(axis=-1))
            targets.append(to_numpy(y))
        model.proj = np.linalg.lstsq(
            np.concatenate(feats), np.concatenate(targets), rcond=None)[0]

    def save_model(self, model, path):
        # Same name ``load_model`` reads from ``meta["submission_dir"]``.
        np.savez(path / "weights.npz", proj=model.proj)
