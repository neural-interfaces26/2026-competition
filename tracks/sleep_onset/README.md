# Track 03 — Sleep onset

Regress the latency (in seconds) to the first N2 sleep epoch from a short EEG
window, capped at 600 s — lightweight, home-headband-friendly sleep monitoring.

- **Input:** torch batch `X` `(B, C, T)`, already on `meta["device"]`.
- **Output:** `predict(X) -> (B,)` — one predicted latency per window, float
  seconds.
- **Ranking metric:** warm-up reports **binned MAE** (bMAE) over time-to-onset
  bins `[0, 40, 90, 300, 600]` s, plus plain MAE. The sealed Muse evaluation
  ranks **weighted binned MAE** (W-bMAE), macro-averaged across seen- and
  unseen-subject groups.
- **Objective:** `Sleep-onset` · output size `meta["n_outputs"]` = `1`.

## Data

Pick what you train on with `-d` (one-time `benchopt prepare` download):

| `-d` selector | What it is |
|---|---|
| `Sleep-EDF` | Sleep-EDF (`Kemp2000Analysis`) — the public proxy |
| `Simulated` | tiny synthetic set — contract check only, no download |

## Two starting kits

Both paths finish with the same upload: a `submission.py` + weights. Pick by
your architecture — see the [participant guide](https://neural-interfaces26.github.io/participant-guide.html).

### Benchopt — run experiments & package (this repo)

The track is a [benchopt](https://benchopt.github.io) benchmark — the same code
Codabench runs. First, a zero-download check that everything works — the dummy
baseline on the Simulated data (no `prepare` needed):

```bash
benchopt run tracks/sleep_onset --config tracks/sleep_onset/starter.yml
```

Then prepare the real data and train the linear baseline into a ready-to-upload
submission with the training config:

```bash
benchopt prepare tracks/sleep_onset --config tracks/sleep_onset/training.yml
benchopt run     tracks/sleep_onset --config tracks/sleep_onset/training.yml
```

Baselines (each a solver **and** a valid submission) live in
[`solvers/`](solvers/) — full details in [`solvers/README.md`](solvers/README.md):

| Solver | What it shows |
|---|---|
| `Median` | the contract with no weights and no training (leaderboard floor) |
| `Mean-Ridge` | a scikit-learn model; `save_model` writes a joblib dump |
| `Torch-Linear` | the same model in PyTorch, with its own Adam loop in `fit` |
| `EEGNet` | braindecode EEGNet regressor, trained end-to-end |

### NeuralBench — tasks, datasets & reference baselines

[NeuralBench](https://facebookresearch.github.io/neuroai/neuralbench/) defines
the task, public splits, and reference baselines. Reproduce the start kit:

```bash
pip install neuralbench
neuralbench eeg sleep_onset --download
neuralbench eeg sleep_onset --prepare
neuralbench eeg sleep_onset -m eegnet --debug
```

Reference results on Sleep-EDF (`Kemp2000Analysis`) — development data,
**not** Codabench warm-up scores (bMAE, lower is better):

| Baseline | bMAE (s) |
|---|---|
| Chance | 205.42 ± 0.01 |
| EEGNet | 143.30 ± 0.40 |
| REVE (frozen probe) | 134.89 ± 2.02 |

Full guide: [Track 03 on NeuralBench ↗](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html).

## Develop your own model

A submission is one `submission.py` with `class Solver(CompetSolver)`:

- **`load_model(meta) -> model`** (required) — build your architecture, load the
  shipped weights from `meta["submission_dir"]`, return a model exposing
  `predict(X)` (in eval mode, on `meta["device"]`).
- `fit(model, train_loader)` (optional) — train locally; Codabench never calls it.
- `save_model(model, path)` (optional) — write weights; a training run then
  packages `outputs/<Solver.name>/` (`submission.py` + weights), ready to zip.

`meta` provides `n_chans`, `n_times`, `sfreq`, `ch_names`, `chs_info`,
`n_outputs` (= `1`), and `device`. `predict(X)` must return `(B,)` float
latencies in seconds.

Fastest loop — copy a baseline from `solvers/`, rename it `MyModel`, edit
`load_model` / `fit`, then reuse the training config (it pins the dataset and
`training=True`; `-s` overrides its baseline solver, so that flag is all you
change — the data is already prepared from the step above):

```bash
benchopt run tracks/sleep_onset --config tracks/sleep_onset/training.yml -s MyModel
```

**Train on different data.** The config pins the dataset; override it with `-d`
to train elsewhere — `Sleep-EDF` with different parameters, or your own data
loaded straight from a file: `-d path/to/my_dataset.py`.

Full submission contract, `meta` keys, and packaging:
[`codabench/pages/participate.md`](../../codabench/pages/participate.md) and the
repo [README](../../README.md#develop--train-your-model-with-benchopt).
