# Track 02: BCI Decoding

Classify cued mental commands from short EEG windows across recording sessions
and days.

## Model contract

- **Input:** torch tensor `(B, C, T)` on `meta["device"]`.
- **Output:** `predict(X) -> (B,)`, one integer class index per window, with
  `meta["n_classes"]` possible classes.
- **Evaluation:** model-facing inference is causal and streamed. Windows reach
  `predict` one at a time (`B = 1`), one session of each participant at a time,
  and every session starts from a `copy.deepcopy` of your model whose optional
  `reset_state()` is called first: a model can adapt within a session but
  carries nothing over to the next one.
- **Metric implemented here:** balanced accuracy within each subject, session
  and context cell, averaged over cells. On PROTEUS the context of a run is
  its interface, Graz or BrainHero. Balanced accuracy averaged over sessions
  alone (NeuralBench's `bal_acc_stream_mean`), balanced accuracy pooled over
  all windows and plain accuracy are reported alongside.
- **Benchopt objective:** `BCI-decoding`.

See the Codabench **Track description** for the active phase data and official
ranking specification.

## Public data choices

| Benchopt `-d` selector | Data |
|---|---|
| `BCI[study=dreyer2026proteus]` | PROTEUS (NEMAR nm000290), the default warm-up study |
| `BCI[study=tangermann2012]` | BNCI2014_001, a smaller real-data check |
| `Simulated` | tiny synthetic contract check, with no download |

The real selectors run NeuralBench's streamed Track 02 task,
`eeg _motor_imagery_stream`, on its default (PROTEUS) and `tangermann2012`
variants. Each window is 4 s long; the signal is notch-filtered at 50 and
60 Hz and band-pass filtered at 0.1-75 Hz at its recorded rate, resampled to
120 Hz, and fed in microvolts with no scaling or clamping. On PROTEUS a
window holds 41 EEG channels and starts 1 s after its cue.

## Worked examples

### NeuralBench start kit

Use NeuralBench to explore the neurophysiology task, preprocessing, public
split, and reference model pipeline:

```bash
pip install neuralbench 'nemar-py>=0.3.1'
neuralbench eeg _motor_imagery_stream --download
neuralbench eeg _motor_imagery_stream --prepare
neuralbench eeg _motor_imagery_stream -m eegnet --debug
```

Reference development results on the public PROTEUS cross-subject split with
NeuralBench 0.4.0. The metric is `test/bal_acc_stream_mean`, averaged over
the streamed test sessions; Codabench additionally separates PROTEUS
contexts when computing its official warm-up cell mean.

| Baseline | Balanced accuracy |
|---|---|
| Constant prediction | 33.3% |
| EEGNet | 32.6% |
| CTNet | 35.0% |
| REVE fine-tuned | 36.0% |

Values are means over seeds 33, 34 and 35. This public split is purely
cross-subject and is harder than the mixed cross-session and cross-subject
sealed protocol.

[Open the Track 02 NeuralBench guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html).
After training its EEGNet, use the shared
[NeuralBench-to-Codabench bridge](../README.md#package-a-neuralbench-checkpoint).

### Benchopt competition kit

Use the shared [Benchopt workflow](../README.md#develop-and-package-with-benchopt)
with `<track> = bci_decoding`. `training.yml` selects PROTEUS and exports
the `Torch-Linear` submission.

Editable solvers live in [`solvers/`](solvers/):

| Solver | Purpose |
|---|---|
| `Constant` | uploadable constant floor |
| `MeanLogReg` | scikit-learn linear baseline with joblib weights |
| `Torch-Linear` | PyTorch linear baseline with `fit` and `save_model` |
| `EEGNet` | end-to-end Braindecode EEGNet |
| `REVE` | frozen pretrained REVE encoder with a ridge classifier |

## Adapt your own model

The track metadata adds `n_classes` to the shared submission metadata. Your
model must return one integer class index per window.

Override the training config's dataset with another selector from the table
above or a custom Benchopt dataset file:

```bash
benchopt run tracks/bci_decoding \
  --config tracks/bci_decoding/training.yml \
  -d "BCI[study=tangermann2012]" -s MyModel
```

The [Submission Guide](../../codabench/pages/participate.md) defines the full
contract and ZIP layout.
