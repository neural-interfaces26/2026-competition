# Track 04 — EMG-to-Pose

Regress hand-joint angle trajectories from 16-channel wrist surface EMG — the
core of wearable control beyond a calibrated lab setup.

- **Input:** torch batch `X` `(B, C, T)` (16-channel EMG windows), on `meta["device"]`.
- **Output:** `predict(X) -> (B, n_joints, T)` — joint-angle sequences in
  **radians** (a coarser time axis is nearest-resampled to the target `T`).
- **Ranking metric:** mean angular MAE, averaged over joints and time, reported
  in **degrees**. Return radians — do not convert.
- **Objective:** `EMG-pose` · output size `meta["n_joints"]`.

## Data

Pick what you train on with `-d` (one-time `benchopt prepare` download):

| `-d` selector | What it is |
|---|---|
| `Salter2024Emg2pose` | public EMG2Pose corpus (via the neuralbench `emg/pose` task) |
| `Simulated` | tiny synthetic set — contract check only, no download |

## Two starting kits

Both paths finish with the same upload: a `submission.py` + weights. Pick by
your architecture — see the [participant guide](https://neural-interfaces26.github.io/participant-guide.html).

### Benchopt — run experiments & package (this repo)

The track is a [benchopt](https://benchopt.github.io) benchmark — the same code
Codabench runs. Prepare the data and train the linear baseline into a
ready-to-upload submission with the starter config:

```bash
benchopt prepare tracks/emg_pose --config tracks/emg_pose/starter.yml
benchopt run     tracks/emg_pose --config tracks/emg_pose/starter.yml
```

Baselines (each a solver **and** a valid submission) live in
[`solvers/`](solvers/) — full details in [`solvers/README.md`](solvers/README.md):

| Solver | What it shows |
|---|---|
| `MeanPose` | the contract with no weights and no training (train-mean-pose floor) |
| `Ridge` | a scikit-learn model; `save_model` writes a joblib dump |
| `Torch-Linear` | the same model in PyTorch, with its own Adam loop in `fit` |
| `EEGNet` | braindecode EEGNet with a dense readout, trained end-to-end |

The two linear baselines are *per-time-step* readouts: the Simulated EMG is an
instantaneous linear mixture of the joint angles.

### NeuralBench — tasks, datasets & reference baselines

[NeuralBench](https://facebookresearch.github.io/neuroai/neuralbench/) defines
the task, public splits, and reference baselines. Reproduce the start kit:

```bash
pip install neuralbench 'eegdash>=0.8.2'
neuralbench emg pose -m vemg2pose --download
neuralbench emg pose -m vemg2pose --prepare
neuralbench emg pose -m vemg2pose --debug
```

Reference result on public EMG2Pose (`Salter2024Emg2pose`) — development data,
**not** a Codabench warm-up score (angular MAE, lower is better):

| Baseline | Angular MAE (°) |
|---|---|
| NeuroPose | 17.5 ± 1.5 |

Full guide: [Track 04 on NeuralBench ↗](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html).

## Develop your own model

A submission is one `submission.py` with `class Solver(CompetSolver)`:

- **`load_model(meta) -> model`** (required) — build your architecture, load the
  shipped weights from `meta["submission_dir"]`, return a model exposing
  `predict(X)` (in eval mode, on `meta["device"]`).
- `fit(model, train_loader)` (optional) — train locally; Codabench never calls it.
- `save_model(model, path)` (optional) — write weights; a training run then
  packages `outputs/<Solver.name>/` (`submission.py` + weights), ready to zip.

`meta` provides `n_chans`, `n_times`, `sfreq`, `ch_names`, `chs_info`,
`n_joints`, and `device`. `predict(X)` must return `(B, n_joints, T)` joint
angles in radians.

Fastest loop — copy a baseline from `solvers/`, rename it `MyModel`, edit
`load_model` / `fit`, then reuse the starter config (it pins the dataset and
`training=True`; `-s` overrides its baseline solver, so that flag is all you
change — the data is already prepared from the step above):

```bash
benchopt run tracks/emg_pose --config tracks/emg_pose/starter.yml -s MyModel
```

Full submission contract, `meta` keys, and packaging:
[`codabench/pages/participate.md`](../../codabench/pages/participate.md) and the
repo [README](../../README.md#develop--train-your-model-with-benchopt).
