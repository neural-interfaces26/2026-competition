# Track 04: EMG-to-Pose

Reconstruct continuous hand-joint trajectories from 16-channel wrist surface
EMG.

## Model contract

- **Input:** torch tensor `(B, C, T)` on `meta["device"]`.
- **Output:** `predict(X) -> (B, n_joints, T)` joint angles in **radians**,
  where `n_joints = meta["n_joints"]`.
- **Metric implemented here:** mean angular MAE over joints and time. Codabench
  converts only the final aggregate score to degrees for the leaderboard.
- **Benchopt objective:** `EMG-pose`.

Return radians, not degrees. See the Codabench **Track description** for the
active phase data and official ranking specification.

## Public data choices

| Benchopt `-d` selector | Data |
|---|---|
| `Salter2024Emg2pose` | public EMG2Pose corpus through the NeuralBench task |
| `Simulated` | tiny synthetic contract check, with no download |

## Worked examples

### NeuralBench start kit

Use NeuralBench to explore the neurophysiology task, preprocessing, public
split, and reference model pipeline:

```bash
pip install neuralbench 'eegdash>=0.8.2'
neuralbench emg pose -m vemg2pose --download
neuralbench emg pose -m vemg2pose --prepare
neuralbench emg pose -m vemg2pose --debug
```

Reference development result on public EMG2Pose, not a Codabench warm-up
score:

| Baseline | Angular MAE (°) |
|---|---|
| NeuroPose | 17.5 ± 1.5 |

[Open the Track 04 NeuralBench guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html).
After training its VEMG2Pose model, use the shared
[NeuralBench-to-Codabench bridge](../README.md#package-a-neuralbench-checkpoint).

### Benchopt competition kit

Use the shared [Benchopt workflow](../README.md#develop-and-package-with-benchopt)
with `<track> = emg_pose`. `training.yml` selects public EMG2Pose and exports
the `Torch-Linear` submission.

Editable solvers live in [`solvers/`](solvers/):

| Solver | Purpose |
|---|---|
| `MeanPose` | uploadable train-mean-pose floor |
| `Ridge` | scikit-learn per-time-step baseline with joblib weights |
| `Torch-Linear` | PyTorch per-time-step baseline with `fit` and `save_model` |
| `EEGNet` | end-to-end dense Braindecode model |

The two linear examples are per-time-step readouts. The simulated EMG is an
instantaneous linear mixture of the joint trajectories so this relationship
remains learnable in the zero-download check.

## Adapt your own model

The track metadata adds `n_joints` to the shared submission metadata. Your
model must return radian trajectories. If its output has a coarser time axis,
the objective nearest-resamples it to the target length.

To use a custom Benchopt dataset instead of the configured EMG2Pose corpus:

```bash
benchopt run tracks/emg_pose \
  --config tracks/emg_pose/training.yml \
  -d path/to/my_dataset.py -s MyModel
```

The [Submission Guide](../../codabench/pages/participate.md) defines the full
contract and ZIP layout.
