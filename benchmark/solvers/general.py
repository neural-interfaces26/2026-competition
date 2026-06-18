"""General solver — the specialist (task-specific) track.

The participant's submission provides a model via ``get_model(task, meta)``
with a scikit-learn-like contract: ``fit(train_loader)`` then ``predict(X)``
returning ``(B,)`` for epoched tasks or ``(B, T)`` for dense tasks.

When no ``submission`` module is importable, it falls back to a built-in
:class:`~benchmark_utils.baselines.DefaultGeneralModel` so the track stays
runnable standalone.
"""

from benchopt import BaseSolver


class Solver(BaseSolver):

    name = "General"

    requirements = ["scikit-learn", "pip::torch"]

    sampling_strategy = "run_once"

    # ``benchopt test`` instantiates the objective with this track so the
    # solver runs (the default ``general`` already matches, but be explicit).
    test_config = {"objective": {"track": "general"}}

    def skip(self, track, **objective_dict):
        if track != "general":
            return True, "General only runs on the general track"
        return False, None

    def set_objective(self, train_loader, task, task_kind, track, n_classes,
                      **meta):
        self.train_loader = train_loader
        meta = {**meta, "n_classes": n_classes, "task_kind": task_kind}

        try:
            from submission import get_model
        except ImportError:
            def get_model(task, meta):
                from benchmark_utils.baselines import DefaultGeneralModel
                return DefaultGeneralModel(task_kind=meta["task_kind"])

        self.model = get_model(task, meta)

    def run(self, _):
        self.model.fit(self.train_loader)

    def get_result(self):
        return dict(model=self.model)
