# Track 03: Sleep Onset

Estimate the remaining time, in seconds, until the first N2 sleep epoch from a
short EEG window.

## Model contract

- **Input:** torch tensor `(B, C, T)` on `meta["device"]`.
- **Output:** `predict(X) -> (B,)` float latencies in seconds, capped at 600,
  with `meta["n_outputs"] = 1`.
- **Evaluation:** causal and streamed. Each recording reaches `predict` one
  window at a time (`B = 1`), forward in time, and starts from a
  `copy.deepcopy` of your model whose optional `reset_state()` is called
  first: a model can adapt within a recording but carries nothing over to the
  next one.
- **Metric implemented here:** the Muse warm-up W-bMAE: within each
  recording, the MAE inside the target ranges `[0, 40, 90, 300, 600]`
  seconds, averaged over the non-empty ranges with severity weights 10, 5, 3
  and 1, then averaged over recordings. The unweighted binned MAE and plain
  MAE over all windows are reported alongside.
- **Benchopt objective:** `Sleep-onset`.

The final Muse evaluation uses severity-weighted binned MAE and a seen versus
unseen sleeper macro-average. See the Codabench **Track description** for the
active phase specification.

## Public data choices

| Benchopt `-d` selector | Data |
|---|---|
| `Interaxon2026Muse` | public Muse data (NEMAR nm000287), the warm-up set |
| `Sleep-EDF` | Sleep-EDF (`Kemp2000Analysis`), a public proxy |
| `Simulated` | tiny synthetic contract check, with no download |

Both real selectors run NeuralBench's streamed Track 03 task,
`eeg _sleep_onset_stream`, on its default (Muse) and `kemp2000analysis`
variants. Each window is 5 s long, at the recording's own rate,
unfiltered, in microvolts with no scaling or clamping. On Muse a window
holds the four channels TP9, AF7, AF8 and TP10 at 128 Hz, and the windows
tile each recording from its start to its first N2 epoch.

## Worked examples

### NeuralBench start kit

Use NeuralBench to explore the neurophysiology task, preprocessing, public
split, and reference model pipeline:

```bash
pip install neuralbench 'nemar-py>=0.3.1'
neuralbench eeg _sleep_onset_stream --download
neuralbench eeg _sleep_onset_stream --prepare
neuralbench eeg _sleep_onset_stream -m eegnet --debug
```

Until the next NeuralBench release on PyPI includes the streamed task,
install NeuralBench from the revision pinned in
[`requirements.txt`](../../requirements.txt).

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
with `<track> = sleep_onset`. `training.yml` selects the Muse warm-up data
and exports the `Torch-Linear` submission.

Editable solvers live in [`solvers/`](solvers/):

| Solver | Purpose |
|---|---|
| `Median` | uploadable constant floor |
| `Mean-Ridge` | scikit-learn linear baseline with joblib weights |
| `Torch-Linear` | PyTorch linear baseline with `fit` and `save_model` |
| `EEGNet` | end-to-end Braindecode EEGNet regressor |
| `REVE` | frozen pretrained REVE encoder with a fitted ridge head |

## Adapt your own model

The track metadata adds `n_outputs = 1` to the shared submission metadata.
Your model must return one float latency in seconds per window.

To use a custom Benchopt dataset instead of the configured Muse data:

```bash
benchopt run tracks/sleep_onset \
  --config tracks/sleep_onset/training.yml \
  -d path/to/my_dataset.py -s MyModel
```

The [Submission Guide](../../codabench/pages/participate.md) defines the full
contract and ZIP layout.
