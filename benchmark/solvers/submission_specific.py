"""Sample submission for the EEG competition — specific (specialist) track.

A specialist submission is a **valid benchopt solver** subclassing
``CompetEEGSpecificSolver``. Unlike the foundation-model track, a specialist is
**task-specific**: you declare the task you target via the ``task`` class
attribute (one submission per task) and train a model directly on that task's
data — labels *do* reach the model here. You implement:

- ``load_model(self, meta)`` -> a model exposing ``fit(train_loader)`` and
  ``predict(X)``. ``meta`` carries ``sfreq, ch_names, chs_info, n_chans,
  n_times, n_classes, task, task_kind``.

This sample targets motor imagery (``task="mi"``, epoched) with a classic
per-channel **log-variance + logistic-regression** pipeline — a genuinely
task-specific feature for MI. Swap in a real specialist (e.g. a braindecode
EEGNet trained on the dataloader) the same way.

Test it locally (from the bundle root) — ``Simulated`` mimics the ``mi`` task
with no downloads, so you can validate the full pipeline in seconds:

    cp solution/submission_specific.py benchmark/solvers/_submission_spec.py
    benchopt run benchmark/ -d "Simulated[task=mi]" -s MI-LogVar
    rm benchmark/solvers/_submission_spec.py

    # then on the real task (downloads MOABB once): -d MOABB-MI
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetEEGSpecificSolver
from benchmark_utils.data import to_numpy


class LogVarMI:
    """Per-channel log-variance + logistic regression (epoched motor imagery).

    Log-variance per channel is a classic discriminative feature for motor
    imagery — it captures the band power modulation the task relies on.
    """

    def __init__(self):
        self.clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        )

    @staticmethod
    def _features(X):
        # (B, C, T) -> (B, C) log-variance over time.
        X = to_numpy(X)
        return np.log(X.var(axis=-1) + 1e-8)

    def fit(self, train_loader):
        feats, targets = [], []
        for X, y, _info in train_loader:
            feats.append(self._features(X))
            targets.append(to_numpy(y))
        self.clf.fit(np.concatenate(feats), np.concatenate(targets))
        return self

    def predict(self, X):
        return self.clf.predict(self._features(X))


class Solver(CompetEEGSpecificSolver):

    name = "MI-LogVar"
    task = "mi"

    def load_model(self, meta):
        return LogVarMI()
