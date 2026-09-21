"""Smoke-test submission for the BCI-decoding track.

This deliberately simple template classifier validates ingestion, data
loading, inference, scoring, and leaderboard publication. It loads locally
trained templates from ``weights.npz`` when provided and otherwise creates a
deterministic stand-in in memory. The optional ``fit`` and ``save_model``
hooks support local training and export.

See "Participation > Optional: train through Benchopt" to see how to train
this model on actual data to produce weights.npz with benchopt.
"""

import numpy as np

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


class TemplateClassifier:
    """Nearest class-template on the window channel means."""

    def __init__(self, templates):
        self.templates = templates                    # (K, C)

    def predict(self, X):
        feats = to_numpy(X).mean(axis=-1)             # (B, C)
        d = ((feats[:, None] - self.templates[None]) ** 2).sum(-1)
        return d.argmin(axis=1)                       # (B,)


class Solver(CompetSolver):

    name = "Sample-BCI"

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.npz"
        if weights.exists():                          # written by save_model
            templates = np.load(weights)["templates"]
        else:                                         # in-memory smoke fallback
            rng = np.random.default_rng(0)
            templates = rng.standard_normal(
                (meta["n_classes"], meta["n_chans"])
            )
        return TemplateClassifier(templates)

    def fit(self, model, train_loader):
        # Per-class mean of the window channel means.
        feats, labels = [], []
        for X, y, _info in train_loader:
            feats.append(to_numpy(X).mean(axis=-1))
            labels.append(to_numpy(y))
        feats, labels = np.concatenate(feats), np.concatenate(labels)
        model.templates = np.stack(
            [feats[labels == k].mean(axis=0) for k in np.unique(labels)])

    def save_model(self, model, path):
        # ``path`` is the submission folder, so write the
        # file name ``load_model`` reads back.
        np.savez(path / "weights.npz", templates=model.templates)
