"""Objective for the EMG-to-pose track (track 4).

Regress hand-joint angle trajectories from wrist surface EMG: a submission's
model receives torch batches ``(B, C, T)`` (16-channel EMG windows) and must
return the joint-angle sequences ``predict(X) -> (B, n_joints, T)`` in
**degrees** (predictions on a coarser time axis are nearest-resampled to the
target's ``T``).

Ranking metric: **mean angular MAE** (degrees), averaged over joints and
time. Data flows as lazy dataloaders — see ``compet_core/data.py``; targets
``y`` are float ``(B, n_joints, T)`` angle sequences.

.. note:: The real dataset (Salter2024 emg2pose) has no neuralfetch study /
   neuralbench task config yet — only ``Simulated`` ships for now; the real
   loader is tracked as follow-up work.
"""

import numpy as np
from benchopt import BaseObjective

import benchmark_utils  # noqa: F401 — locates compet_core
from compet_core.data import resample_labels, to_numpy


class Objective(BaseObjective):

    name = "EMG-pose"
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

    def set_data(self, train_loader, test_loader, n_joints, **meta):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_joints = n_joints
        self.meta = meta  # sfreq, ch_names, n_chans, n_times, ...

    def skip(self, train_loader=None, **data):
        if self.training and train_loader is None:
            return True, "training=True needs a dataset with a train split"
        return False, None

    def get_objective(self):
        # Train loader goes to the solver; the test loader stays here so the
        # objective owns evaluation.
        return dict(
            # Inference-only unless training is selected: no loader, no fit.
            train_loader=self.train_loader if self.training else None,
            n_joints=self.n_joints,
            **self.meta,
        )

    def evaluate_result(self, model):
        abs_err, count = 0.0, 0
        for X, y, _info in self.test_loader:
            pred = to_numpy(model.predict(X))     # (B, J, T')
            y = to_numpy(y)                       # (B, J, T)
            pred = resample_labels(pred, y.shape[-1])
            abs_err += np.abs(pred - y).sum()
            count += y.size

        return dict(angular_mae=float(abs_err / count))

    def get_one_result(self):
        # A trivial constant model, used by ``benchopt test`` to validate the
        # metric computation.
        from compet_core.baselines import ConstantPose
        return dict(model=ConstantPose(self.n_joints))
