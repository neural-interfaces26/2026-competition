"""Base solver for the linear-probe (foundation-model) track.

A submission is a **valid benchopt solver** subclassing
:class:`CompetEEGSolver`. The participant only implements how their frozen
foundation model turns windows into embeddings:

- ``load_model(self, meta)``           -> the frozen model (loaded once).
- ``time_embed(self, model, X)``       -> ``(B, T', D)`` temporal embedding.
- ``embed(self, model, X)``            -> ``(B, D)`` window embedding
                                          (default: mean-pool ``time_embed``
                                          over time; override for custom
                                          pooling).

The base class handles everything else: track gating, fitting the
scikit-learn linear head (via :class:`~benchmark_utils.linear_probe.
LinearProbe`, array-API on-device), and returning the fitted model. The
embedding used depends on the task regime — ``embed`` for epoched tasks,
``time_embed`` for dense tasks — so one submission serves both.
"""

import torch
from benchopt import BaseSolver

from benchmark_utils.linear_probe import Encoder, LinearProbe


class _SubmissionEncoder(Encoder):
    """Adapt a solver's ``embed``/``time_embed`` to the probe ``Encoder``."""

    def __init__(self, solver):
        self.solver = solver

    def encode(self, X):
        s = self.solver
        if s.task_kind == "epoched":
            emb = torch.as_tensor(s.embed(s.model, X))  # (B, D)
            return emb[:, None, :]                      # (B, 1, D)
        return torch.as_tensor(s.time_embed(s.model, X))  # (B, T', D)


class CompetEEGSolver(BaseSolver):
    """Base class for foundation-model (linear-probe track) submissions."""

    requirements = ["scikit-learn", "pip::torch"]

    sampling_strategy = "run_once"

    # ``benchopt test`` instantiates the objective on the linear_probe track.
    test_config = {"objective": {"track": "linear_probe"}}

    def skip(self, track, **objective_dict):
        if track != "linear_probe":
            return True, "Foundation-model submissions run on linear_probe"
        return False, None

    def set_objective(self, train_loader, task, task_kind, track, n_classes,
                      **meta):
        self.train_loader = train_loader
        self.task = task
        self.task_kind = task_kind
        self.n_classes = n_classes
        # Device the batches already live on (set by the dataset's loaders);
        # place the model here in ``load_model`` so it matches ``X``.
        self.device = meta.get("device", "cpu")
        self.meta = {
            **meta, "n_classes": n_classes, "task": task,
            "task_kind": task_kind,
        }
        self.model = self.load_model(self.meta)
        self.probe = LinearProbe(_SubmissionEncoder(self), mode=task_kind)

    def run(self, _):
        self.probe.fit(self.train_loader)

    def get_result(self):
        return dict(model=self.probe)

    # --- to be implemented by the submission ---------------------------

    def load_model(self, meta):
        """Load and return the frozen foundation model (called once)."""
        raise NotImplementedError

    def time_embed(self, model, X):
        """Map windows ``(B, C, T)`` to a temporal embedding ``(B, T', D)``."""
        raise NotImplementedError

    def embed(self, model, X):
        """Map windows ``(B, C, T)`` to a window embedding ``(B, D)``.

        Default: mean-pool the temporal embedding over time. Override for a
        custom pooling (e.g. a CLS token).
        """
        return torch.as_tensor(self.time_embed(model, X)).mean(dim=1)


class CompetEEGSpecificSolver(BaseSolver):
    """Base class for specialist (specific track) submissions.

    Unlike the foundation-model track, a specialist is **task-specific**: it
    declares the task it targets via the ``task`` class attribute and trains a
    model directly on that task's data (labels *do* reach the model). One
    submission per task — set ``task`` and implement ``load_model``:

    - ``load_model(self, meta)`` -> a model exposing ``fit(train_loader)`` and
      ``predict(X)`` (epoched: ``(B,)`` labels; dense: ``(B, T)`` per-step
      labels). ``meta`` carries ``sfreq, ch_names, chs_info, n_chans, n_times,
      n_classes, task, task_kind, device``. Batches already arrive on
      ``meta["device"]``; build the model there too (see ``solvers/
      specific_eegnet.py`` for a GPU example).

    The base class gates on the specific track *and* the targeted task (via
    ``skip``), runs training, and returns the fitted model.
    """

    requirements = ["scikit-learn", "pip::torch"]

    sampling_strategy = "run_once"

    # The task this specialist targets — subclasses must set it.
    task = None

    # ``benchopt test`` instantiates the objective on the specific track.
    test_config = {"objective": {"track": "specific"}}

    def skip(self, track, task, **objective_dict):
        if track != "specific":
            return True, "Specialist submissions run on the specific track"
        if self.task is not None and task != self.task:
            return True, (
                f"{self.name} targets task {self.task!r}, not {task!r}"
            )
        return False, None

    def set_objective(self, train_loader, task, task_kind, track, n_classes,
                      **meta):
        # Note: ``self.task`` is the *declared* target (class attribute); the
        # objective's task arrives here and is exposed through ``meta``.
        self.train_loader = train_loader
        self.task_kind = task_kind
        # Device the batches already live on (set by the dataset's loaders);
        # place the model here in ``load_model`` so it matches ``X``.
        self.device = meta.get("device", "cpu")
        self.meta = {
            **meta, "n_classes": n_classes, "task": task,
            "task_kind": task_kind,
        }
        self.model = self.load_model(self.meta)

    def run(self, _):
        self.model.fit(self.train_loader)

    def get_result(self):
        return dict(model=self.model)

    # --- to be implemented by the submission ---------------------------

    def load_model(self, meta):
        """Build the specialist model (``fit(loader)`` / ``predict(X)``)."""
        raise NotImplementedError
