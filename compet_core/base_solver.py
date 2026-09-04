"""Base solver — the submission contract shared by all competition tracks.

A submission is a **valid benchopt solver**: a folder with a ``submission.py``
defining ``class Solver(CompetSolver)``, plus any weight files it needs. The
participant only implements plain-PyTorch code — no benchopt, neuralset or
neuralbench knowledge is required:

- ``load_model(self, meta)`` -> model (required). Build the model (loading
  any shipped weights from ``meta["weights_dir"]``) and place it on
  ``meta["device"]``. The model must expose ``predict(X)`` taking a torch
  batch ``(B, C, T)``; the output shape is track-specific (see each track's
  objective docstring).
- ``fit(self, model, train_loader)`` (optional, default no-op). Training /
  light fine-tuning on the track's train split. It runs both locally (the
  starting kit trains baselines this way) and **on the competition server**,
  within the compute budget — heavy pre-training should happen offline and be
  shipped as weights loaded in ``load_model``.

``meta`` is a plain dict: ``sfreq, ch_names, chs_info, n_chans, n_times,
n_classes`` (classification) or ``n_outputs`` (regression), ``device,
weights_dir``. Batches are torch tensors ``(X, y, info)`` already moved onto
``meta["device"]`` by the track's dataloaders.
"""

import inspect
import os
from pathlib import Path

from benchopt import BaseSolver


class CompetSolver(BaseSolver):
    """Base class for all track submissions (and built-in baselines)."""

    requirements = ["scikit-learn", "pip::torch"]

    sampling_strategy = "run_once"

    def set_objective(self, train_loader, **meta):
        self.train_loader = train_loader
        # Device the batches already live on (set by the dataset's loaders);
        # place the model there in ``load_model``.
        self.device = meta.get("device", "cpu")
        # Weight files ship alongside ``submission.py``: the ingestion program
        # points COMPET_SUBMISSION_DIR at the submission folder; locally this
        # defaults to the directory holding the solver file itself.
        weights_dir = os.environ.get("COMPET_SUBMISSION_DIR")
        if weights_dir is None:
            weights_dir = Path(inspect.getfile(type(self))).parent
        self.meta = {**meta, "device": self.device,
                     "weights_dir": Path(weights_dir)}
        self.model = self.load_model(self.meta)

    def run(self, _):
        self.fit(self.model, self.train_loader)

    def get_result(self):
        return dict(model=self.model)

    # --- to be implemented by the submission ---------------------------

    def load_model(self, meta):
        """Build and return the model (must expose ``predict(X)``)."""
        raise NotImplementedError

    def fit(self, model, train_loader):
        """Optional training / light fine-tuning on the train split."""
