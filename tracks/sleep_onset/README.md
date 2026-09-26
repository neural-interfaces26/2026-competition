# Track 03: Sleep Onset

Estimate the remaining time, in seconds, until the first N2 sleep epoch from a
short EEG window.

## Model contract

- **Input:** torch tensor `(B, C, T)` on `meta["device"]`.
- **Output:** `predict(X) -> (B,)` float latencies in seconds, capped at 600,
  with `meta["n_outputs"] = 1`.
- **Metric implemented here:** Sleep-EDF warm-up binned MAE over target bins
  `[0, 40, 90, 300, 600]` seconds, with plain MAE reported alongside.
- **Benchopt objective:** `Sleep-onset`.

The final Muse evaluation uses severity-weighted binned MAE and a seen versus
unseen sleeper macro-average. See the Codabench **Track description** for the
active phase specification.

## Public data choices

| Benchopt `-d` selector | Data |
|---|---|
| `Sleep-EDF` | Sleep-EDF (`Kemp2000Analysis`), the current public proxy |
| `Simulated` | tiny synthetic contract check, with no download |

## Worked examples

### NeuralBench start kit

Use NeuralBench to explore the neurophysiology task, preprocessing, public
split, and reference model pipeline:

```bash
pip install neuralbench
neuralbench eeg sleep_onset --download
neuralbench eeg sleep_onset --prepare
neuralbench eeg sleep_onset -m eegnet --debug
```

Reference development results on Sleep-EDF, not Codabench warm-up scores:

| Baseline | bMAE (s) |
|---|---|
| Chance | 205.42 ± 0.01 |
| EEGNet | 143.30 ± 0.40 |
| REVE frozen probe | 134.89 ± 2.02 |

[Open the Track 03 NeuralBench guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html).
After training its EEGNet, use the shared
[NeuralBench-to-Codabench bridge](../README.md#package-a-neuralbench-checkpoint).

### Benchopt competition kit

Use the shared [Benchopt workflow](../README.md#develop-and-package-with-benchopt)
with `<track> = sleep_onset`. `training.yml` selects Sleep-EDF and exports the
`Torch-Linear` submission.

Editable solvers live in [`solvers/`](solvers/):

| Solver | Purpose |
|---|---|
| `Median` | uploadable constant floor |
| `Mean-Ridge` | scikit-learn linear baseline with joblib weights |
| `Torch-Linear` | PyTorch linear baseline with `fit` and `save_model` |
| `EEGNet` | end-to-end Braindecode EEGNet regressor |

## Adapt your own model

The track metadata adds `n_outputs = 1` to the shared submission metadata.
Your model must return one float latency in seconds per window.

To use a custom Benchopt dataset instead of the configured Sleep-EDF proxy:

```bash
benchopt run tracks/sleep_onset \
  --config tracks/sleep_onset/training.yml \
  -d path/to/my_dataset.py -s MyModel
```

The [Submission Guide](../../codabench/pages/participate.md) defines the full
contract and ZIP layout.
