"""Objective for the sleep-onset track (track 3).

Regress the latency (in seconds) to the first N2 sleep epoch from a
short EEG window, capped at 600 s. A submission's model receives torch
batches ``(B, C, T)`` and must return one predicted latency per window
(``predict(X) -> (B,)`` floats, in seconds).

Evaluation is causal and streamed (``benchmark_utils/streaming.py``): each
recording is predicted one window at a time (``B = 1``), forward in time,
and every recording starts from a fresh copy of the model whose optional
``reset_state()`` is called first.

Ranking metric: **W-bMAE per recording, averaged over recordings** (the
Muse warm-up score, NeuralBench's ``wbmae_stream_mean``). Within a
recording, the MAE is computed inside each true time-to-onset range
``[0, 40)``, ``[40, 90)``, ``[90, 300)`` and ``[300, 600]`` s, and the
non-empty ranges are averaged with severity weights 10, 5, 3 and 1. The
unweighted binned MAE (bMAE) and the plain MAE, both over all windows, are
reported alongside. The sealed phase's seen/unseen macro-average ships with
the sealed evaluation data. Data flows as lazy dataloaders — see
``benchmark_utils/data.py``.
"""

import functools

import numpy as np
from benchopt import BaseObjective

from benchmark_utils.metrics import binned_mae, group_scores
from benchmark_utils.streaming import predict_streams

BIN_EDGES = (0.0, 40.0, 90.0, 300.0, 600.0)
# Severity weights of the four ranges, closest to sleep onset first.
BIN_WEIGHTS = (10.0, 5.0, 3.0, 1.0)


class Objective(BaseObjective):

    name = "Sleep-onset"
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

    def set_data(self, train_loader, test_loader, **meta):
        self.train_loader = train_loader
        self.test_loader = test_loader
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
            n_outputs=1,
            **self.meta,
        )

    def evaluate_result(self, model):
        y_true, y_pred, info = predict_streams(model, self.test_loader)
        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
        if len(y_pred) != len(info["stream_id"]):
            raise ValueError("predict(X) must return one latency per window.")

        # Each stream is one recording.
        wbmae = functools.partial(binned_mae, bin_edges=BIN_EDGES,
                                  bin_weights=BIN_WEIGHTS)
        per_recording = group_scores(wbmae, y_true, y_pred,
                                     info["stream_id"])

        return dict(
            wbmae_recording_mean=float(per_recording.mean()),
            bmae=binned_mae(y_true, y_pred, BIN_EDGES),
            mae=float(np.abs(y_pred - y_true).mean()),
            n_recordings=len(per_recording),
        )

    def get_one_result(self):
        # A trivial constant model, used by ``benchopt test`` to validate the
        # metric computation.
        from benchmark_utils.baselines import MedianRegressor
        return dict(model=MedianRegressor(value=300.0))
