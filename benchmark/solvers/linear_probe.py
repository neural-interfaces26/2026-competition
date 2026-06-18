"""Reference foundation-model solver (linear-probe track).

A built-in :class:`~benchmark_utils.base_solver.CompetEEGSolver` using a
dependency-light random-projection encoder, so the benchmark runs standalone
(``benchopt run benchmark/``) and serves as a template for participant
submissions. A real submission subclasses ``CompetEEGSolver`` the same way and
swaps ``load_model`` / ``time_embed`` for an actual foundation model — see
``solution/submission.py``.
"""

from benchmark_utils.base_solver import CompetEEGSolver
from benchmark_utils.linear_probe import RandomProjectionEncoder


class Solver(CompetEEGSolver):

    name = "LinearProbe"

    def load_model(self, meta):
        return RandomProjectionEncoder(n_chans=meta["n_chans"])

    def time_embed(self, model, X):
        # RandomProjectionEncoder already returns a temporal embedding.
        return model.encode(X)  # (B, T', D)
