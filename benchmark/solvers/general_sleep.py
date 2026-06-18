"""Reference specialist for the ``sleep`` (sleep staging) task (general track).

Built-in per-task baseline using the dependency-light
:class:`~benchmark_utils.baselines.DefaultGeneralModel` (dense regime). A
specialist submission targeting sleep staging subclasses
:class:`~benchmark_utils.base_solver.CompetEEGGeneralSolver` and swaps
``load_model`` for its own task-specific model.
"""

from benchmark_utils.base_solver import CompetEEGGeneralSolver
from benchmark_utils.baselines import DefaultGeneralModel


class Solver(CompetEEGGeneralSolver):

    name = "General-Sleep"
    task = "sleep"

    def load_model(self, meta):
        return DefaultGeneralModel(task_kind=meta["task_kind"])
