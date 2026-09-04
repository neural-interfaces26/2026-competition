"""Objective for the image-decoding track (track 1).

Decode which image a subject viewed from a single EEG epoch, evaluated by
**retrieval**: a submission's model maps torch batches ``(B, C, T)`` to
predicted image *embeddings* (``predict(X) -> (B, D)``, matching the target
embedding space, e.g. DINOv2). Each test window's prediction is ranked
against the candidate pool (the unique target embeddings of the test split)
by cosine similarity.

Ranking metric: **top-5 accuracy** (top-1 reported alongside).
Data flows as lazy dataloaders — see ``compet_core/data.py``; targets ``y``
are the float embeddings ``(B, D)`` of the viewed images.
"""

import numpy as np
from benchopt import BaseObjective

from compet_core.data import to_numpy
from compet_core.metrics import topk_accuracy


def _normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class Objective(BaseObjective):

    name = "Image-decoding"
    url = "https://github.com/tomMoral/2026-neurips_compet-eeg"

    requirements = [
        "scikit-learn", "pip::torch",
        # Shared competition components (repo root package).
        "pip::git+https://github.com/tomMoral/2026-neurips_compet-eeg"
        "@4-track-restructure",
    ]

    min_benchopt_version = "1.9.2"

    # Each solver runs once to completion (no convergence curve).
    sampling_strategy = "run_once"

    def set_data(self, train_loader, test_loader, n_outputs, **meta):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_outputs = n_outputs  # embedding dimension D
        self.meta = meta  # sfreq, ch_names, chs_info, n_chans, n_times, ...

    def get_objective(self):
        # Train loader goes to the solver; the test loader stays here so the
        # objective owns evaluation.
        return dict(
            train_loader=self.train_loader,
            n_outputs=self.n_outputs,
            **self.meta,
        )

    def evaluate_result(self, model):
        y_true, y_pred = [], []
        for X, y, _info in self.test_loader:
            y_pred.append(to_numpy(model.predict(X)))
            y_true.append(to_numpy(y))
        y_true = np.concatenate(y_true)          # (N, D)
        y_pred = np.concatenate(y_pred)          # (N, D)

        # Candidate pool: the unique target embeddings of the test split
        # (repeated presentations of one image share its embedding).
        candidates, target_idx = np.unique(
            y_true, axis=0, return_inverse=True
        )
        scores = _normalize(y_pred) @ _normalize(candidates).T  # (N, M)

        return dict(
            top5_acc=topk_accuracy(scores, target_idx, k=5),
            top1_acc=topk_accuracy(scores, target_idx, k=1),
            n_candidates=int(len(candidates)),
        )

    def get_one_result(self):
        # A trivial constant model, used by ``benchopt test`` to validate the
        # metric computation.
        from compet_core.baselines import ConstantEmbedder
        return dict(model=ConstantEmbedder(self.n_outputs))
