"""Base solver — the submission contract shared by all competition tracks.

A submission is a **valid benchopt solver**: a folder with a ``submission.py``
defining ``class Solver(CompetSolver)``, plus any weight files it needs. The
participant only implements plain-PyTorch code — no benchopt, neuralset or
neuralbench knowledge is required:

- ``load_model(self, meta)`` -> model (required). Build the model (loading
  any shipped weights from ``meta["submission_dir"]``) and place it on
  ``meta["device"]``. The model must expose ``predict(X)`` taking a torch
  batch ``(B, C, T)``; the output shape is track-specific (see each track's
  objective docstring).
- ``fit(self, model, train_loader)`` (optional, default no-op). **Opt-in
  training**: it only runs when the objective's ``training`` parameter is
  selected (``benchopt run ... -o "<objective>[training=True]"``) — this is
  how the baselines are trained and how you can train your own model with
  the exact competition data and evaluation. A plain ``benchopt run`` is
  inference-only, mirroring the competition server (whose phase configs
  never select ``training``): the submitted model must be fully trained
  offline and shipped as weights loaded in ``load_model``.
- ``save_model(self, model, path)`` (optional). Write the trained model's
  weight files into the directory ``path`` — the mirror of ``load_model``.
  When implemented, a training run writes your solver (as ``submission.py``)
  and those files into ``<track>/outputs/<name>/`` — a ready-to-upload
  submission folder.

``meta`` is a plain dict: ``sfreq, ch_names, chs_info, n_chans, n_times,
n_classes`` (classification) or ``n_outputs`` (regression), ``device,
submission_dir``. Batches are torch tensors ``(X, y, info)`` already moved onto
``meta["device"]`` by the track's dataloaders.
"""

import inspect
import os
import shutil
from pathlib import Path

from benchopt import BaseSolver
from benchopt.benchmark import get_running_benchmark


class CompetSolver(BaseSolver):
    """Base class for all track submissions (and built-in baselines).

    torch and scikit-learn are provided by the benchmark environment (the
    Objective's cpu/gpu requirements); submissions only declare their own
    extras, e.g. ``requirements = ["pip::my-model-pkg"]``.
    """

    requirements = []

    sampling_strategy = "run_once"

    def set_objective(self, train_loader, **meta):
        self.train_loader = train_loader
        # Device the batches already live on (set by the dataset's loaders);
        # place the model there in ``load_model``.
        self.device = meta.get("device", "cpu")
        # A submission is one folder: ``submission.py`` plus its weight files,
        # read back through ``meta["submission_dir"]``. The platform points
        # COMPET_SUBMISSION_DIR at the uploaded folder; locally each solver
        # gets its own ``outputs/<name>/`` so trained submissions persist,
        # coexist, and reload on the next (inference-only) run.
        submission_dir = os.environ.get("COMPET_SUBMISSION_DIR")
        if submission_dir is None:
            out = get_running_benchmark().get_output_folder()
            submission_dir = out / self.name
        submission_dir = Path(submission_dir)
        submission_dir.mkdir(parents=True, exist_ok=True)
        self.meta = {**meta, "device": self.device,
                     "submission_dir": submission_dir}
        self.model = self.load_model(self.meta)

    def run(self, _):
        # No train loader means inference-only, exactly like the competition
        # platform: the objective only hands one over when its ``training``
        # parameter is selected (`-o "<objective>[training=True]"`).
        if self.train_loader is not None:
            self.fit(self.model, self.train_loader)
            self._export_submission()

    def _export_submission(self):
        """Write the solver + trained weights into ``submission_dir`` (its
        ``outputs/<name>/`` folder) — a ready-to-upload submission that the
        next inference-only run reloads."""
        if type(self).save_model is CompetSolver.save_model:
            return
        sub_dir = self.meta["submission_dir"]
        shutil.copyfile(Path(inspect.getfile(type(self))),
                        sub_dir / "submission.py")
        self.save_model(self.model, sub_dir)
        print(f"[compet] submission ready in {sub_dir} — zip its contents to "
              "upload on the competition's 'My Submissions' tab.")

    def get_result(self):
        return dict(model=self.model)

    # --- to be implemented by the submission ---------------------------

    def load_model(self, meta):
        """Build and return the model (must expose ``predict(X)``)."""
        raise NotImplementedError

    def fit(self, model, train_loader):
        """Optional training / light fine-tuning on the train split."""

    def save_model(self, model, path):
        """Optional: write the trained weights into ``path`` (see above)."""
