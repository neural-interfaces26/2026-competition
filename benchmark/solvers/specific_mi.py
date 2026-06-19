"""Reference specialist for the ``mi`` (motor imagery) task (specific track).

Built-in per-task baseline using the dependency-light
:class:`~benchmark_utils.baselines.DefaultSpecificModel`. A specialist
submission targeting motor imagery subclasses
:class:`~benchmark_utils.base_solver.CompetEEGSpecificSolver` and swaps
``load_model`` for its own task-specific model.
"""

from benchmark_utils.base_solver import CompetEEGSpecificSolver
from benchmark_utils.baselines import DefaultSpecificModel


class Solver(CompetEEGSpecificSolver):

    name = "Specific-MI"
    task = "mi"

    def load_model(self, meta):
        return DefaultSpecificModel(task_kind=meta["task_kind"])
