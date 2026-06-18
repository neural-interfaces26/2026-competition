"""Linear-probe solver — the foundation-model track.

The participant's submission provides only a frozen encoder via
``get_encoder(meta)`` (``encode(X) -> (B, T', D)``). This solver wraps it in a
:class:`~benchmark_utils.linear_probe.LinearProbe` and fits the scikit-learn
head by streaming the train loader. The encoder is frozen, never sees labels.

When no ``submission`` module is importable (e.g. standalone
``benchopt run benchmark/``), it falls back to a built-in random-projection
encoder so the track stays runnable without a submission or pretrained weights.
"""

from benchopt import BaseSolver

from benchmark_utils.linear_probe import LinearProbe


class Solver(BaseSolver):

    name = "LinearProbe"

    requirements = ["scikit-learn", "pip::torch"]

    sampling_strategy = "run_once"

    # ``benchopt test`` instantiates the objective with this track so the
    # solver runs (instead of skipping the default ``general`` variant).
    test_config = {"objective": {"track": "linear_probe"}}

    def skip(self, track, **objective_dict):
        if track != "linear_probe":
            return True, "LinearProbe only runs on the linear_probe track"
        return False, None

    def set_objective(self, train_loader, task, task_kind, track, n_classes,
                      **meta):
        self.train_loader = train_loader
        meta = {**meta, "n_classes": n_classes, "task": task}

        # Import the participant's encoder if present, else use the fallback.
        # NB: not a module-level import — a missing ``submission`` must not
        # make benchopt think the solver itself is not installed.
        try:
            from submission import get_encoder
        except ImportError:
            from benchmark_utils.baselines import (
                default_encoder as get_encoder,
            )

        encoder = get_encoder(meta)
        self.probe = LinearProbe(encoder, mode=task_kind)

    def run(self, _):
        self.probe.fit(self.train_loader)

    def get_result(self):
        return dict(model=self.probe)
