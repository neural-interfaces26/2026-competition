"""Sample submission for the BCI-decoding track.

Demonstrates the contract: ship a **fully trained** model — the platform runs
inference-only (``fit`` never runs there). This trivial example "loads" a
frozen per-class template matcher from ``weights.npz`` shipped alongside
(created on the fly if absent, standing in for your real training artefacts).
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
