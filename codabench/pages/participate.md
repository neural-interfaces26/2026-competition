# How to participate

Your submission is a folder containing a `submission.py` that defines a
**benchopt solver**, plus any weight files your model needs. The competition
server evaluates it **inference-only**: your model must arrive fully trained.

## The contract

Subclass `CompetSolver` (from the bundled `compet_core` package) and
implement:

- `load_model(self, meta)` — build your model and load your shipped weights
  from `meta["weights_dir"]`, placing it on `meta["device"]`. Return an
  object exposing `predict(X)`.
- `predict(X)` receives torch batches `X: (B, C, T)` already on
  `meta["device"]`; the expected output shape is track-specific (see the
  competition description — e.g. class labels `(B,)`, latencies `(B,)`,
  embeddings `(B, D)`, or joint-angle sequences `(B, J, T)`).

`meta` also carries `sfreq, ch_names, chs_info, n_chans, n_times` and the
track's output size (`n_classes` / `n_outputs` / `n_joints`).

Optionally, implement `fit(self, model, train_loader)` — it only runs
**locally** (never on the server) and lets you train your model with the
exact competition data through the starting kit.

```python
import torch

import benchmark_utils  # noqa: F401 — locates compet_core
from compet_core.base_solver import CompetSolver


class Solver(CompetSolver):
    name = "MyModel"
    # torch/scikit-learn come with the benchmark env; declare only extras.
    requirements = ["pip::my-model-pkg"]

    def load_model(self, meta):
        model = build_my_model(
            n_chans=meta["n_chans"], n_times=meta["n_times"],
        )
        state = torch.load(meta["weights_dir"] / "weights.pt",
                           map_location=meta["device"])
        model.load_state_dict(state)
        return model.to(meta["device"]).eval()
```

Your `predict` can be a method of the returned model (plain PyTorch — no
benchopt, neuralset or neuralbench knowledge is needed inside your model).

## Test locally

The starting kit is the benchmark itself. From the competition repo (no
install of the repo needed — `benchmark_utils` locates the shared
`compet_core` package):

```bash
benchopt install tracks/<track>            # CPU env (add --gpu for CUDA)
benchopt run tracks/<track> -d Simulated   # zero-download smoke test
```

To iterate on a submission before uploading, drop your files (code +
weights) into the track's `solvers/` folder and run it like any benchopt
solver — the platform evaluation is the same `benchopt run`,
inference-only:

```bash
cp my_submission/* tracks/<track>/solvers/
benchopt run tracks/<track> -d Simulated -s my-solver
```

Training on the real data locally: `benchopt prepare tracks/<track>`
downloads it once, then your solver's `fit` runs with
`benchopt run tracks/<track>`.

## Submit

Zip the submission folder (`submission.py` + weights) and upload it on the
*My Submissions* tab. Baselines and reference submissions live in the
`solution/` folder of the bundle.
