"""Objective for the BCI-decoding track (track 2).

Cued mental-command classification from short EEG windows — *epoched*: one
label per window. A submission's model receives torch batches ``(B, C, T)``
and must return one predicted class per window (``predict(X) -> (B,)``).

Ranking metric: **balanced accuracy** (plain accuracy reported alongside).
Data flows as lazy dataloaders — see ``compet_core/data.py``.
"""

import numpy as np
from benchopt import BaseObjective
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from compet_core.data import to_numpy


class Objective(BaseObjective):

    name = "BCI-decoding"
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

    def set_data(self, train_loader, test_loader, n_classes, **meta):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_classes = n_classes
        self.meta = meta  # sfreq, ch_names, chs_info, n_chans, n_times, ...

    def get_objective(self):
        # Train loader goes to the solver; the test loader stays here so the
        # objective owns evaluation.
        return dict(
            train_loader=self.train_loader,
            n_classes=self.n_classes,
            **self.meta,
        )

    def evaluate_result(self, model):
        y_true, y_pred = [], []
        for X, y, _info in self.test_loader:
            y_pred.append(to_numpy(model.predict(X)))
            y_true.append(to_numpy(y))
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        return dict(
            balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
            accuracy=accuracy_score(y_true, y_pred),
            n_classes=self.n_classes,
        )

    def get_one_result(self):
        # A trivial constant model, used by ``benchopt test`` to validate the
        # metric computation.
        from compet_core.baselines import ConstantClassifier
        return dict(model=ConstantClassifier())
