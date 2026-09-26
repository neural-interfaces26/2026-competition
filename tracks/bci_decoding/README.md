# Track 02: BCI Decoding

Classify cued mental commands from short EEG windows across recording sessions
and days.

## Model contract

- **Input:** torch tensor `(B, C, T)` on `meta["device"]`.
- **Output:** `predict(X) -> (B,)`, one integer class index per window, with
  `meta["n_classes"]` possible classes.
- **Metric implemented here:** balanced accuracy, with plain accuracy reported
  alongside.
- **Benchopt objective:** `BCI-decoding`.

See the Codabench **Track description** for the active phase data and official
ranking specification.

## Public data choices

| Benchopt `-d` selector | Data |
|---|---|
| `BCI[study=dreyer2023]` | Dreyer2023Large, the default warm-up study |
| `BCI[study=stieger2021]` | Stieger2021Continuous, an additional public proxy |
| `BCI[study=tangermann2012]` | BNCI2014_001, a smaller real-data check |
| `Simulated` | tiny synthetic contract check, with no download |

## Worked examples

### NeuralBench start kit

Use NeuralBench to explore the neurophysiology task, preprocessing, public
split, and reference model pipeline:

```bash
pip install neuralbench 'moabb>=1.7.1'
neuralbench eeg motor_imagery --dataset dreyer2023 --download
neuralbench eeg motor_imagery --dataset dreyer2023 --prepare
neuralbench eeg motor_imagery --dataset dreyer2023 -m eegnet --debug
```

Reference development results on Stieger 2021, not the current Dreyer 2023
Codabench warm-up evaluation:

| Baseline | Balanced accuracy |
|---|---|
| Chance | 24.81 ± 1.03 |
| EEGNet | 58.58 ± 0.34 |
| REVE frozen probe | 68.04 ± 0.73 |

[Open the Track 02 NeuralBench guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html).
After training its EEGNet, use the shared
[NeuralBench-to-Codabench bridge](../README.md#package-a-neuralbench-checkpoint).

### Benchopt competition kit

Use the shared [Benchopt workflow](../README.md#develop-and-package-with-benchopt)
with `<track> = bci_decoding`. `training.yml` selects Dreyer 2023 and exports
the `Torch-Linear` submission.

Editable solvers live in [`solvers/`](solvers/):

| Solver | Purpose |
|---|---|
| `Constant` | uploadable constant floor |
| `MeanLogReg` | scikit-learn linear baseline with joblib weights |
| `Torch-Linear` | PyTorch linear baseline with `fit` and `save_model` |
| `EEGNet` | end-to-end Braindecode EEGNet |

## Adapt your own model

The track metadata adds `n_classes` to the shared submission metadata. Your
model must return one integer class index per window.

Override the training config's dataset with another selector from the table
above or a custom Benchopt dataset file:

```bash
benchopt run tracks/bci_decoding \
  --config tracks/bci_decoding/training.yml \
  -d "BCI[study=stieger2021]" -s MyModel
```

The [Submission Guide](../../codabench/pages/participate.md) defines the full
contract and ZIP layout.
