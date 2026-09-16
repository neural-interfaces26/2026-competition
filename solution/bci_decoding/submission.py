"""Sample submission for the BCI-decoding track.

Demonstrates the full contract with a trivial per-class template matcher:

- submitted as-is, it **loads** its trained templates from ``weights.npz``
  shipped alongside (the platform runs inference-only — a stand-in file is
  generated when missing, so the sample always runs);
- run with ``-o "BCI-decoding[training=True]"``, ``fit`` recomputes the
  templates on the train split and ``save_model`` writes them — the run then
  drops a ready-to-upload ``outputs/submission_Sample-BCI.zip``.
"""

import numpy as np

import benchmark_utils  # noqa: F401 — locates compet_core
from compet_core.base_solver import CompetSolver
from compet_core.data import to_numpy


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
        weights = meta["weights_dir"] / "weights.npz"
        if not weights.exists():                      # stand-in artefact
            rng = np.random.default_rng(0)
            np.savez(weights, templates=rng.standard_normal(
                (meta["n_classes"], meta["n_chans"])))
        return TemplateClassifier(np.load(weights)["templates"])

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
        # ``path`` is the root of the zip built at the end of a training run,
        # next to this file copied in as ``submission.py``. Uploading that zip
        # makes ``path`` the submission folder — i.e. ``meta["weights_dir"]``
        # — so write the file name ``load_model`` reads back.
        np.savez(path / "weights.npz", templates=model.templates)
