"""Objective for the BCI-decoding track (track 2).

Cued mental-command classification from short EEG windows — *epoched*: one
label per window. A submission's model receives torch batches ``(B, C, T)``
and must return one predicted class per window (``predict(X) -> (B,)``).

Evaluation is causal and streamed (``benchmark_utils/streaming.py``): the
windows of each session of each participant reach the model one at a time
(``B = 1``), its runs in recording order, and every session starts from a
fresh copy of the model whose optional ``reset_state()`` is called first.

Ranking metric: **balanced accuracy averaged over cells**. It is computed
within each subject x session x context cell, then averaged over cells, so
every cell counts equally regardless of its number of windows. A dataset
whose windows carry no ``context_id`` in their ``info`` has one context per
session, which makes it NeuralBench's ``bal_acc_stream_mean``. Balanced
accuracy pooled over all windows and plain accuracy are reported alongside.
Data flows as lazy dataloaders — see ``benchmark_utils/data.py``.
"""

import numpy as np
from benchopt import BaseObjective

from benchmark_utils.metrics import balanced_accuracy, group_scores
from benchmark_utils.streaming import predict_streams


class Objective(BaseObjective):

    name = "BCI-decoding"
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

    def set_data(self, train_loader, test_loader, n_classes, **meta):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_classes = n_classes
        self.meta = meta  # sfreq, ch_names, chs_info, n_chans, n_times, ...

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
            n_classes=self.n_classes,
            **self.meta,
        )

    def evaluate_result(self, model):
        y_true, y_pred, info = predict_streams(model, self.test_loader)
        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
        if len(y_pred) != len(info["stream_id"]):
            raise ValueError("predict(X) must return one class per window.")

        # A cell is a stream (one session of one participant) and a context.
        context = info.get("context_id", np.zeros(len(y_true), np.int64))
        cells = np.stack([info["stream_id"], context], axis=1)
        cell_scores = group_scores(balanced_accuracy, y_true, y_pred, cells)

        return dict(
            balanced_accuracy_cell_mean=float(cell_scores.mean()),
            balanced_accuracy=balanced_accuracy(y_true, y_pred),
            accuracy=float(np.mean(y_true == y_pred)),
            n_classes=self.n_classes,
            n_cells=len(cell_scores),
        )

    def get_one_result(self):
        # A trivial constant model, used by ``benchopt test`` to validate the
        # metric computation.
        from benchmark_utils.baselines import ConstantClassifier
        return dict(model=ConstantClassifier())
