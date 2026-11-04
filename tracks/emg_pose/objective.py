"""Objective for the EMG-to-pose track (track 4).

Regress hand-joint angle trajectories from wrist surface EMG: a submission's
model receives torch batches ``(B, C, T)`` (16-channel EMG windows) and must
return the joint-angle sequences ``predict(X) -> (B, n_joints, T)`` in
**radians** (predictions on a coarser time axis are nearest-resampled to the
target's ``T``).

Sealed ranking: **Mean Angular MAE per subject (deg)**
(``angular_mae_subject_mean``): average errors within each subject over valid
examples, joints and time, then average subjects equally. Pooled angular MAE
(``angular_mae``) remains the historical warm-up ranking and a sealed
diagnostic. Also reported: the sample standard deviation of per-subject MAEs,
RMSE (in degrees), and R2 averaged over joints. Data flows as lazy dataloaders — see
``benchmark_utils/data.py``; targets ``y`` are float ``(B, n_joints, T)``
angle sequences.

Warm-up and sealed evaluation both use ordinary batches of independent
windows. Batch size, composition, and order are evaluator details rather than
model-facing signals; submissions must not rely on temporal continuity or
persistent state across ``predict`` calls.

.. note:: Two datasets ship: ``Salter2024Emg2pose`` — the official corpus via
   the neuralbench ``emg/pose`` task — and the zero-dependency ``Simulated``
   smoke test.
"""

import numpy as np
from benchopt import BaseObjective

from benchmark_utils.data import resample_labels, subject_ids, to_numpy
from benchmark_utils.metrics import SequenceRegressionScores


class Objective(BaseObjective):

    name = "EMG-pose"
    url = "https://github.com/neural-interfaces26/2026-competition"

    # CPU/GPU variants resolved by ``benchopt install [--gpu]``; the conda
    # metapackages pin the matching torch build (CI installs the cpu one).
    # Install the full trio from one channel: mixing a conda torch with pip
    # torchvision/torchaudio breaks operators such as torchvision::nms.
    requirements = {
        "cpu": ["scikit-learn", "pytorch-cpu", "torchvision", "torchaudio"],
        "gpu": ["scikit-learn", "pytorch-gpu", "torchvision", "torchaudio"],
    }

    min_benchopt_version = "1.10.0"

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
        scores = SequenceRegressionScores()
        for X, y, info in self.test_loader:
            pred = to_numpy(model.predict(X))     # (B, J, T')
            y = to_numpy(y)                       # (B, J, T)
            pred = resample_labels(pred, y.shape[-1])
            scores.update(pred, y, subject_ids(info, len(y)))
        scores = scores.compute()

        # NeuralBench and EMG2Pose train and infer in radians. The competition
        # reports the same angular errors in degrees.
        deg = 180.0 / np.pi
        return dict(
            angular_mae=scores["mae"] * deg,
            angular_mae_subject_mean=scores["mae_group_mean"] * deg,
            angular_mae_subject_std=scores["mae_group_std"] * deg,
            angular_rmse=scores["rmse"] * deg,
            r2_score=scores["r2"],
            n_subjects=scores["n_groups"],
        )

    def get_one_result(self):
        # A trivial constant model, used by ``benchopt test`` to validate the
        # metric computation.
        from benchmark_utils.baselines import ConstantPose
        return dict(model=ConstantPose(self.n_joints))
