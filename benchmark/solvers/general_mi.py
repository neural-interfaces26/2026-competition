"""Reference specialist for the ``mi`` (motor imagery) task (general track).

Built-in per-task baseline using the dependency-light
:class:`~benchmark_utils.baselines.DefaultGeneralModel`. A specialist
submission targeting motor imagery subclasses
:class:`~benchmark_utils.base_solver.CompetEEGGeneralSolver` and swaps
``load_model`` for its own task-specific model.
"""

from benchmark_utils.base_solver import CompetEEGGeneralSolver
from benchmark_utils.baselines import DefaultGeneralModel


class Solver(CompetEEGGeneralSolver):

    name = "General-MI"
    task = "mi"

    def load_model(self, meta):
        return DefaultGeneralModel(task_kind=meta["task_kind"])
