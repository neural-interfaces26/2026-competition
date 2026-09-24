# Track 02 — BCI decoding

Cued mental-command classification from short EEG windows — one label per
window, decoded reliably across sessions and days.

- **Input:** torch batch `X` `(B, C, T)`, already on `meta["device"]`.
- **Output:** `predict(X) -> (B,)` — one predicted class index per window.
- **Ranking metric:** balanced accuracy (plain accuracy reported alongside).
- **Objective:** `BCI-decoding` · output size `meta["n_classes"]`.

## Data

Pick what you train on with `-d` (one-time `benchopt prepare` download):

| `-d` selector | What it is |
|---|---|
| `BCI[study=dreyer2023]` | Dreyer2023Large — the **warm-up evaluation** study, **default** |
| `BCI[study=stieger2021]` | Stieger2021Continuous — additional 4-class public proxy |
| `BCI[study=tangermann2012]` | BNCI2014_001 — small, for real-data smoke tests |
| `Simulated` | tiny synthetic set — contract check only, no download |

## Two starting kits

Both paths finish with the same upload: a `submission.py` + weights. Pick by
your architecture — see the [participant guide](https://neural-interfaces26.github.io/participant-guide.html).

### Benchopt — run experiments & package (this repo)

The track is a [benchopt](https://benchopt.github.io) benchmark — the same code
Codabench runs. Prepare the data and train the linear baseline into a
ready-to-upload submission with the starter config:

```bash
benchopt prepare tracks/bci_decoding --config tracks/bci_decoding/starter.yml
benchopt run     tracks/bci_decoding --config tracks/bci_decoding/starter.yml
```

Baselines (each a solver **and** a valid submission) live in
[`solvers/`](solvers/) — full details in [`solvers/README.md`](solvers/README.md):

| Solver | What it shows |
|---|---|
| `Constant` | the contract with no weights and no training (chance floor) |
| `MeanLogReg` | scikit-learn: mean-over-time features + logistic regression |
| `Torch-Linear` | the same model in PyTorch, with its own Adam loop in `fit` |
| `EEGNet` | braindecode EEGNet, trained end-to-end |

### NeuralBench — tasks, datasets & reference baselines

[NeuralBench](https://facebookresearch.github.io/neuroai/neuralbench/) defines
the task, public splits, and reference baselines. Reproduce the start kit:

```bash
pip install neuralbench 'moabb>=1.7.1'
neuralbench eeg motor_imagery --dataset dreyer2023 --download
neuralbench eeg motor_imagery --dataset dreyer2023 --prepare
neuralbench eeg motor_imagery --dataset dreyer2023 -m eegnet --debug
```

Reference results on Stieger 2021 (`Stieger2021Continuous`) — development data,
**not** Codabench warm-up scores; warm-up currently evaluates on Dreyer 2023:

| Baseline | Balanced accuracy |
|---|---|
| Chance | 24.81 ± 1.03 |
| EEGNet | 58.58 ± 0.34 |
| REVE (frozen probe) | 68.04 ± 0.73 |

Full guide: [Track 02 on NeuralBench ↗](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html).

## Develop your own model

A submission is one `submission.py` with `class Solver(CompetSolver)`:

- **`load_model(meta) -> model`** (required) — build your architecture, load the
  shipped weights from `meta["submission_dir"]`, return a model exposing
  `predict(X)` (in eval mode, on `meta["device"]`).
- `fit(model, train_loader)` (optional) — train locally; Codabench never calls it.
- `save_model(model, path)` (optional) — write weights; a training run then
  packages `outputs/<Solver.name>/` (`submission.py` + weights), ready to zip.

`meta` provides `n_chans`, `n_times`, `sfreq`, `ch_names`, `chs_info`,
`n_classes`, and `device`. `predict(X)` must return `(B,)` integer class labels.

Fastest loop — copy a baseline from `solvers/`, rename it `MyModel`, edit
`load_model` / `fit`, then reuse the starter config (it pins the dataset and
`training=True`; `-s` overrides its baseline solver, so that flag is all you
change — the data is already prepared from the step above):

```bash
benchopt run tracks/bci_decoding --config tracks/bci_decoding/starter.yml -s MyModel
```

**Train on different data.** The config pins the default study; override it with
`-d` to train elsewhere — another study, e.g. `-d "BCI[study=stieger2021]"`
(see the Data table above), or your own data loaded straight from a file:
`-d path/to/my_dataset.py`.

Full submission contract, `meta` keys, and packaging:
[`codabench/pages/participate.md`](../../codabench/pages/participate.md) and the
repo [README](../../README.md#develop--train-your-model-with-benchopt).
