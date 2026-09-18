# How to participate

Your submission is a folder holding a `submission.py` plus any weight files
your model needs, zipped and uploaded on the *My Submissions* tab. The
server evaluates it **inference-only**: your model must arrive fully
trained. Nothing is installed at submission time — the evaluation image
already carries torch, scikit-learn, benchopt and the data stack.

## The contract

`submission.py` defines `class Solver(CompetSolver)`, a
[benchopt](https://benchopt.github.io) solver. You implement plain PyTorch:
no benchopt, neuralset or neuralbench knowledge is needed inside your model.

- **`load_model(self, meta)` → model** (required). Build your model, load the
  weights you shipped from `meta["submission_dir"]`, place it on
  `meta["device"]`, and return an object exposing `predict(X)`.
- **`predict(X)`** receives a torch batch `X: (B, C, T)` already on
  `meta["device"]` and returns the track's output (table below).
- **`fit(self, model, train_loader)`** (optional) trains your model. It never
  runs on the server — the platform's phase configs are inference-only — but
  it is how you train locally against the exact competition data and
  evaluation.
- **`save_model(self, model, path)`** (optional) writes the trained weights
  into the directory `path`. When you implement it, a local training run ends
  by zipping your solver (as `submission.py`) together with those files into
  `outputs/submission_<name>.zip`, ready to upload.

`meta` is a plain dict: `sfreq`, `ch_names`, `chs_info`, `n_chans`,
`n_times`, `device`, `submission_dir`, and the track's output size.

| Track | Objective | `predict(X)` returns | Output size | Ranking metric |
|---|---|---|---|---|
| 1 — EEG-to-Image | `Image-decoding` | image embeddings `(B, D)` | `meta["n_outputs"]` = D | top-5 retrieval accuracy |
| 2 — BCI Decoding | `BCI-decoding` | class index per window `(B,)` | `meta["n_classes"]` | balanced accuracy |
| 3 — Sleep Onset | `Sleep-onset` | seconds to onset `(B,)`, float | `meta["n_outputs"]` = 1 | binned MAE |
| 4 — EMG-to-Pose | `EMG-pose` | joint angles `(B, n_joints, T)`, degrees | `meta["n_joints"]` | mean angular MAE |

```python
import torch

from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):
    name = "MyModel"
    # torch and scikit-learn come with the evaluation environment;
    # declare only your own extras.
    requirements = ["pip::my-model-pkg"]

    def load_model(self, meta):
        model = build_my_model(
            n_chans=meta["n_chans"], n_times=meta["n_times"],
        )
        state = torch.load(meta["submission_dir"] / "weights.pt",
                           map_location=meta["device"])
        model.load_state_dict(state)
        return model.to(meta["device"]).eval()
```

`predict` can simply be a method of the model you return.

## What you can see, what stays hidden

Only the **data** is hidden. Everything that loads it, evaluates it and talks
to your model is public, and is the same code in both phases:

|                           | Warm-up phase              | Sealed final phase              |
|---------------------------|----------------------------|---------------------------------|
| Data                      | public proxy, downloadable | held-out cohort, never released |
| Dataset name / parameters | shown on the leaderboard   | disclosed after the phase       |
| Docker image              | same                       | same                            |
| Ingestion + scoring       | same                       | same                            |
| Objective + metric        | same                       | same                            |
| `meta` keys, batch shapes | same contract              | same contract                   |
| Your submission           | unchanged                  | unchanged                       |

So one rule keeps a submission valid on data you never see: **use only
`meta` and the batches you are given**. A solver that reads a dataset name, a
file path, a subject id, or a hard-coded channel count may break on the
held-out cohort.

## Test locally

The starting kit is the benchmark itself, one per track. From a checkout:

```bash
benchopt install tracks/<track>            # CPU env (add --gpu for CUDA)
benchopt run tracks/<track> -d Simulated   # zero-download smoke test
```

`Simulated` needs no download and no data stack, so it is the fastest way to
check that your solver loads and predicts the right shape. To try a
submission, drop your files into the track's `solvers/` folder and run it
like any benchopt solver — the platform evaluation is the same
`benchopt run`, inference-only:

```bash
cp my_submission/* tracks/<track>/solvers/
benchopt run tracks/<track> -d Simulated -s my-solver
```

`benchopt test tracks/<track> --skip-install` is the rehearsal for the
sealed phase: it runs your solver against a differently shaped dataset,
the closest local stand-in for data it has never seen.

To train on the real data, `benchopt prepare tracks/<track>` downloads it
once (large for some tracks), then

```bash
benchopt run tracks/<track> -s my-solver -o "<objective>[training=True]"
```

trains your solver through `fit` and evaluates it exactly as the platform
does — `<objective>` being the track's objective name from the table above.
This is also how the baselines shipped in `solvers/` are trained.

## Develop with benchopt

The starting kit is a set of plain benchopt benchmarks, so while iterating you
get hyperparameter grids in one flag (`-s "my-solver[lr=[1e-4,1e-3]]"`),
cached reruns, interactive HTML reports (`benchopt plot`), reproducible
experiment yamls (`--config`) and parallel/SLURM execution (`-j`,
`--parallel-config`). If you code with an AI assistant, `benchopt
sync-skills --global` teaches it the solver conventions. The starting-kit
README has the full tour.
