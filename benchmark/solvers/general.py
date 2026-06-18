"""Reference specialist solver (general track).

A built-in task-specific baseline (no foundation model): a simple scikit-learn
model trained directly on the windows, handling both regimes. It runs
standalone and serves as the reference for the general track. A specialist
submission can replace it with its own ``Solver`` along the same lines.
"""

from benchopt import BaseSolver

from benchmark_utils.baselines import DefaultGeneralModel


class Solver(BaseSolver):

    name = "General"

    requirements = ["scikit-learn", "pip::torch"]

    sampling_strategy = "run_once"

    test_config = {"objective": {"track": "general"}}

    def skip(self, track, **objective_dict):
        if track != "general":
            return True, "General only runs on the general track"
        return False, None

    def set_objective(self, train_loader, task, task_kind, track, n_classes,
                      **meta):
        self.train_loader = train_loader
        self.model = DefaultGeneralModel(task_kind=task_kind)

    def run(self, _):
        self.model.fit(self.train_loader)

    def get_result(self):
        return dict(model=self.model)
