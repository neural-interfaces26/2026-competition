# Trained EEGNet start-kit submission

This folder is a complete, inference-only Track 03 warm-up submission derived
from the official NeuralBench sleep-onset start kit. It shows how to turn a
model trained outside Codabench into an uploadable competition submission.

## How the model was trained

NeuralBench uses the public `Kemp2000Analysis` Sleep-EDF study. It creates
non-overlapping 5-second windows at 120 Hz from the two bipolar EEG channels
`Fpz-Cz` and `Pz-Oz`. Each target is the number of seconds until the first
stable N2 epoch, capped at 600 seconds.

Sleep-EDF is split **by participant**, with fixed split random states of 33:

| Split | Participants | Windows | Role |
|---|---:|---:|---|
| Training | 46 | 21,840 | Update the EEGNet parameters |
| Validation | 16 | 7,680 | Select the best checkpoint and trigger early stopping |
| Test | 16 | 7,200 | Evaluate the selected model only |

All nights from one participant remain in one split. The training sampler
balances the four latency ranges `[0, 40)`, `[40, 90)`, `[90, 300)`, and
`[300, 600]` seconds so that common long-latency windows do not dominate
training.

NeuralBench trains for at most 40 epochs and monitors **validation bMAE**, with
lower values considered better and an early-stopping patience of seven epochs.
The test partition is not used to update the model or select its checkpoint.

This artifact retains seed 33 only. Its exported checkpoint is from Lightning
epoch index 21, the 22nd training epoch, after 7,524 optimizer steps. It was the
best checkpoint observed, with a validation bMAE of 144.359 seconds. The other
planned seeds were not run.

## What Codabench evaluates during warm-up

The current warm-up configuration also uses public Sleep-EDF and the same
NeuralBench participant split. Codabench prepares the full study so that it can
reconstruct the training, validation, and test partitions, but it does **not**
score the full dataset:

The Codabench run seed is 42, but it does not change the task configuration's
fixed participant-split random states of 33.

1. Codabench runs the submitted solver with `training=False`.
2. `fit` is never called and the shipped weights remain unchanged.
3. Only the **7,200 windows from the 16-participant test partition** are passed
   to `predict(X)` and used for leaderboard scoring.
4. bMAE is computed on those test predictions. MAE is reported separately.

This example selected its checkpoint using validation data only, not the test
partition. However, Sleep-EDF and its labels are public, and the split is
reproducible. Participants can therefore inspect or intentionally overfit the
warm-up test data. Warm-up scores are useful for validating the submission
workflow and comparing iterations, but they are not leakage-free estimates of
generalization and do not determine the final ranking.

The sealed phase uses the private 2026 Muse cohort with hidden participants,
recordings, and labels. Only that sealed evaluation determines the final
ranking. This two-channel Sleep-EDF checkpoint is therefore a **warm-up
example**, not a ready-made sealed-phase model for the four-channel Muse input.

## How this submission was produced

1. Downloaded Sleep-EDF with `neuralbench eeg sleep_onset --download`.
2. Prepared and cached the official windows, targets, and participant split
   with `neuralbench eeg sleep_onset --prepare`.
3. Validated the complete training pipeline with
   `neuralbench eeg sleep_onset -m eegnet --debug`.
4. Trained the seed-33 EEGNet run through its 22nd epoch and retained the
   checkpoint with the lowest validation bMAE observed.
5. Extracted the underlying Braindecode EEGNet state dictionary from the
   NeuralBench checkpoint and saved it as `weights.pt`.
6. Wrapped inference in `eegnet_reg.py`. The solver reconstructs EEGNet from
   Codabench's `meta`, loads `weights.pt`, and returns one latency in seconds
   per input window, clipped to `[0, 600]`.
7. Packaged `eegnet_reg.py` as `submission.py` with `weights.pt` at the ZIP
   root, then verified strict weight loading and the `(B,)` prediction contract.

The ready-to-upload archive is `eegnet-sleep-onset-startkit.zip` and contains:

```text
submission.py
weights.pt
```

To rebuild it after editing the source, run:

```bash
python make_zip.py
```
