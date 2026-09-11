"""Objective for the sleep-onset track (track 3).

Regress the latency (in seconds) to the first stable N2 sleep epoch from a
short EEG window, capped at 600 s. A submission's model receives torch
batches ``(B, C, T)`` and must return one predicted latency per window
(``predict(X) -> (B,)`` floats, in seconds).

Ranking metric: **binned MAE** (bMAE) — the MAE is computed inside
time-to-onset bins ``[0, 40, 90, 300, 600]`` s and averaged with equal
weight across bins, so early-onset windows (rare, clinically interesting)
count as much as the common late ones. Plain MAE is reported alongside.
Data flows as lazy dataloaders — see ``compet_core/data.py``.
"""

import numpy as np
from benchopt import BaseObjective

import benchmark_utils  # noqa: F401 — locates compet_core
from compet_core.data import to_numpy
from compet_core.metrics import binned_mae

BIN_EDGES = (0.0, 40.0, 90.0, 300.0, 600.0)


class Objective(BaseObjective):

    name = "Sleep-onset"
    url = "https://github.com/tomMoral/2026-neurips_compet-eeg"

    # CPU/GPU variants resolved by ``benchopt install [--gpu]``; the conda
    # metapackages pin the matching torch build (CI installs the cpu one).
    requirements = {
        "cpu": ["scikit-learn", "pytorch-cpu"],
        "gpu": ["scikit-learn", "pytorch-gpu"],
    }

    min_benchopt_version = "1.9.2"

    # Each solver runs once to completion (no convergence curve).
    sampling_strategy = "run_once"

    # training=True lets the solvers' optional ``fit`` run before evaluation
    # (how baselines and participants train); default runs are
    # inference-only, like the platform. Select it with
    #     benchopt run ... -o "<objective>[training=True]"
    parameters = {"training": [False]}

    # ``benchopt test`` exercises the full contract, training included.
    test_config = {"training": True}

    def set_data(self, train_loader, test_loader, **meta):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.meta = meta  # sfreq, ch_names, chs_info, n_chans, n_times, ...

    def get_objective(self):
        # Train loader goes to the solver; the test loader stays here so the
        # objective owns evaluation.
        return dict(
            train_loader=self.train_loader,
            training=self.training,
            n_outputs=1,
            **self.meta,
        )

    def evaluate_result(self, model):
        y_true, y_pred = [], []
        for X, y, _info in self.test_loader:
            y_pred.append(to_numpy(model.predict(X)).ravel())
            y_true.append(to_numpy(y).ravel())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        return dict(
            bmae=binned_mae(y_true, y_pred, BIN_EDGES),
            mae=float(np.abs(y_pred - y_true).mean()),
        )

    def get_one_result(self):
        # A trivial constant model, used by ``benchopt test`` to validate the
        # metric computation.
        from compet_core.baselines import MedianRegressor
        return dict(model=MedianRegressor(value=300.0))
