"""Reference specialist for the ``sleep`` (staging) task — specific track.

Built-in per-task baseline using the dependency-light
:class:`~benchmark_utils.baselines.DefaultSpecificModel` (dense regime). A
specialist submission targeting sleep staging subclasses
:class:`~benchmark_utils.base_solver.CompetEEGSpecificSolver` and swaps
``load_model`` for its own task-specific model.
"""

from benchmark_utils.base_solver import CompetEEGSpecificSolver
from benchmark_utils.baselines import DefaultSpecificModel


class Solver(CompetEEGSpecificSolver):

    name = "Specific-Sleep"
    task = "sleep"

    def load_model(self, meta):
        return DefaultSpecificModel(task_kind=meta["task_kind"])
