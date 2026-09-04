# How to participate

Your submission is a single `submission.py` that defines a **benchopt solver**.
Two tracks are available:

## Foundation-model track (linear probing)

Subclass `CompetEEGSolver` and ship only your *frozen* encoder — the
competition fits the scikit-learn linear head for you and scores every task.
You implement:

- `load_model(self, meta)` — load and return your frozen model (once). `meta`
  carries `sfreq, ch_names, chs_info, n_chans, n_times, n_classes, task`.
- `time_embed(self, model, X)` — map a batch of windows `X: (B, C, T)` (torch
  tensor) to a temporal embedding `(B, T', D)`.
- `embed(self, model, X)` — *optional* window embedding `(B, D)`; defaults to
  mean-pooling `time_embed` over time.

```python
import torch
from benchmark_utils.base_solver import CompetEEGSolver


class Solver(CompetEEGSolver):
    name = "MyModel"
    requirements = CompetEEGSolver.requirements + ["pip::my-model-pkg"]

    def load_model(self, meta):
        model = load_my_pretrained_model()
        model.eval()
        return model

    def time_embed(self, model, X):
        with torch.inference_mode():
            return model.encode(X)   # (B, T', D)
```

The same encoder serves both regimes: **epoched** tasks (one label per window,
e.g. motor imagery) use `embed`; **dense** tasks (a per-time-step label
sequence, e.g. sleep staging / onset detection) use `time_embed`.

## Specialist track (task-specific model)

A specialist is **task-specific**: subclass `CompetEEGSpecificSolver`, set the
`task` class attribute to the task you target (one submission per task), and
implement `load_model(self, meta)` returning a model with `fit(train_loader)`
and `predict(X)`. Labels *do* reach your model here. The base class gates on
the specific track *and* your targeted task — see
`solution/submission_specific.py` and the per-task references in
`benchmark/solvers/specific_*.py`.

```python
from benchmark_utils.base_solver import CompetEEGSpecificSolver


class Solver(CompetEEGSpecificSolver):
    name = "MyMISpecialist"
    task = "mi"   # the task this submission targets

    def load_model(self, meta):
        return MyTaskSpecificModel(meta)   # fit(loader) / predict(X)
```

## Test your submission locally

You run the **exact same code** the competition runs — only the data path and
settings change, via config. From a checkout of the benchmark:

```bash
# 1. (once) install the benchmark dependencies
benchopt install benchmark/

# 2. quick check on the zero-download Simulated data
cp submission.py benchmark/solvers/_submission.py
benchopt run benchmark/ -d Simulated -s MyModel
rm benchmark/solvers/_submission.py

# 3. on the real tasks (downloads the data once)
benchopt prepare benchmark/        # or: python tools/setup_data.py
cp submission.py benchmark/solvers/_submission.py
benchopt run benchmark/ -s MyModel
rm benchmark/solvers/_submission.py
```

`Simulated` provides both an epoched and a dense variant with no downloads, so
you can validate the full pipeline (encoder → linear probe → metrics) in
seconds before pulling the real datasets.

See the "Seed" page for a complete starter `submission.py`, and the "Timeline"
page for the competition phases.
