"""Objective for the EEG models competition.

Two evaluation **tracks** are selected through the ``track`` parameter (the
competition's core switch):

- ``linear_probe`` : foundation-model track. The submission provides a frozen
  encoder; the infra auto-fits a scikit-learn linear probe on top.
- ``general``      : specialist track. The submission provides a model that
  trains on the task data and predicts directly.

Two task **regimes** are dispatched through ``task_kind`` (carried by each
dataset), so one set of solvers serves both:

- ``epoched`` : one label per window (e.g. motor imagery).
- ``dense``   : a per-time-step label sequence (e.g. sleep staging / onset).

Data flows as lazy dataloaders — see ``benchmark_utils/data.py``.
"""

import numpy as np
from benchopt import BaseObjective
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from benchmark_utils.data import resample_labels, to_numpy


class Objective(BaseObjective):

    name = "EEG"
    url = "https://github.com/benchopt/benchmark_compet_eeg"

    requirements = ["scikit-learn", "pip::torch"]

    min_benchopt_version = "1.8"

    # Each solver runs once to completion (no convergence curve).
    sampling_strategy = "run_once"

    # The competition's track switch — parametrizes the *evaluation*, so each
    # solver declares (via ``skip``) which track it belongs to.
    parameters = {
        "track": ["general", "linear_probe"],
    }

    def set_data(self, train_loader, test_loader, task, task_kind, metrics,
                 n_classes, **meta):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.task = task
        self.task_kind = task_kind
        self.metrics = metrics
        self.n_classes = n_classes
        self.meta = meta  # sfreq, ch_names, chs_info, n_chans, n_times, ...

    def get_objective(self):
        # Train loader goes to the solver; the test loader stays here so the
        # objective owns evaluation. ``track`` lets each solver skip the other.
        return dict(
            train_loader=self.train_loader,
            task=self.task,
            task_kind=self.task_kind,
            track=self.track,
            n_classes=self.n_classes,
            **self.meta,
        )

    def _collect_predictions(self, model):
        """Iterate the test loader, gather (y_true, y_pred) per time-step.

        The model receives torch tensors directly; predictions/targets are
        converted to numpy here, at the scikit-learn metric boundary.
        """
        y_true, y_pred = [], []
        for X, y, _info in self.test_loader:
            pred = to_numpy(model.predict(X))
            y = to_numpy(y)
            if self.task_kind == "epoched":
                y_true.append(y)
                y_pred.append(pred)
            else:
                # dense: align prediction length to the target per window
                for yi, pi in zip(y, pred):
                    pi = resample_labels(pi, yi.shape[-1])
                    y_true.append(yi)
                    y_pred.append(pi)
        return np.concatenate(y_true), np.concatenate(y_pred)

    def evaluate_result(self, model):
        y_true, y_pred = self._collect_predictions(model)

        result = {}
        for metric in self.metrics:
            if metric == "accuracy":
                result[metric] = accuracy_score(y_true, y_pred)
            elif metric in ("balanced_accuracy", "staging_balanced_accuracy"):
                result[metric] = balanced_accuracy_score(y_true, y_pred)
            elif metric == "onset_f1":
                # Placeholder onset/event metric: detect "non-background"
                # time-steps (background = last class ``n_classes - 1``).
                bg = self.n_classes - 1
                result[metric] = f1_score(
                    y_true != bg, y_pred != bg, zero_division=0
                )
            else:
                raise ValueError(f"Unknown metric: {metric!r}")

        # Context columns (not metrics) — handy on the leaderboard / for debug.
        result.update(
            task=self.task,
            n_classes=self.n_classes,
            track=self.track,
            task_kind=self.task_kind,
        )
        return result

    def get_one_result(self):
        # A trivial constant model, used by ``benchopt test`` to validate the
        # metric computation in both regimes.
        from benchmark_utils.baselines import ConstantModel
        return dict(model=ConstantModel(task_kind=self.task_kind))
