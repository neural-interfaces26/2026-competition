# Scientific FAQ · Track 04

Answers to recurring scientific questions, mostly from Discord, about Track 04. The **Track description** tab and its public **[Markdown source](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/competition_emg.md)** define the phase specifications. The public **[Track 04 repository](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/emg_pose)** contains the executable task, metric, data adapter, and worked models.

## Warm-up and sealed evaluation

### What changes between the warm-up and sealed phases?

|                           | Warm-up                                                                                                                                                                                                                                                                                                                                                 | Sealed final                                                                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Evaluation data**       | Public [EMG2Pose `user_stage` test split](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html#split-and-model-selection), selected by the [warm-up phase config](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/phases/warmup/emg_pose/config.yaml) | Private [2026 Meta Reality Labs evaluation cohort](https://neural-interfaces26.github.io/tracks.html#dataset-track-4)                                              |
| **Generalization tested** | Unseen combinations of users and movement stages that are each represented elsewhere in the public data                                                                                                                                                                                                                                                 | New users, new movement stages, and unseen user-stage combinations                                                                                                 |
| **Ranking metric**        | Mean absolute angular error, computed from radian predictions and reported in degrees                                                                                                                                                                                                                                                                   | The same metric and unit conversion                                                                                                                                |
| **Role**                  | Public development proxy                                                                                                                                                                                                                                                                                                                                | Final competition ranking, as defined in the [competition timeline](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/timeline.md) |

Warm-up scores are indicative only. Although both phases use the same prediction and metric contract, their datasets and generalization shifts differ, so their scores are not directly comparable.

### Will participants receive any data from the sealed cohort?

No. The raw recordings, pose trajectories, participant metadata, and labels are not released or directly accessible. During evaluation, the submitted model receives only task-formatted EMG tensor batches.

## Data and preprocessing

### What is the sealed evaluation dataset?

It is a private **2026 Meta Reality Labs cohort** used only for final evaluation, not a hidden split of public EMG2Pose. The recordings, labels, and participant metadata are not released. Participants only need the published model-facing contract: batches of 16-channel wrist sEMG as input and trajectories for 20 hand-joint angles in radians as output. Exact cohort size, recording count, movement-stage composition, and evaluation order are intentionally not disclosed. See the [website dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-4) for the public description and the [NeuralBench Track 04 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html) for the public EMG2Pose task.

### Is preprocessing the same in warm-up and sealed evaluation?

Yes. The competition pipeline applies **no signal preprocessing** in either phase: no filtering, notch filtering, baseline correction, scaling, or clipping. It segments the 16-channel, 2 kHz sEMG into contiguous, non-overlapping **5-second windows**. Incomplete windows and windows containing invalid inverse-kinematics target frames are excluded. The exact public settings are defined in the **[NeuralBench task config](https://github.com/facebookresearch/neuroai/blob/main/neuralbench-repo/neuralbench/tasks/emg/pose/config.yaml)** and invoked through the competition's **[dataset adapter](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/emg_pose/datasets/emg2pose.py)**.

### May we use NinaPro or other public EMG datasets?

Yes. Pre-training or training on publicly available, redistributable datasets is allowed, subject to their original licenses and the competition **[Terms](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/terms.md)**. The sealed test set and closed clinical datasets are not allowed. External corpora must be declared in the final method description for the reproducibility audit.

### May a model assume fixed input dimensions or batch size?

Use `meta["n_chans"]`, `meta["n_times"]`, `meta["sfreq"]`, and `meta["n_joints"]` when constructing the model. The evaluation signal is fixed to the published 16-channel, 2 kHz protocol, but batch size and ordering are not part of the contract.

## Predictions and scoring

### What must the model predict?

For an input batch `X` of shape `(B, 16, T)`, where `B` is the batch size and `T` is the number of input samples, `predict(X)` must return continuous joint-angle trajectories with shape `(B, 20, T_out)` in **radians**. `T_out` is the number of predicted time points. It may equal `T` for one prediction per input sample, or be smaller for a coarser output timeline. Before scoring, Codabench nearest-neighbor resamples the prediction to the target length `T`.

### Should submissions return radians or degrees?

**Radians.** The reference targets and submitted trajectories remain in radians throughout inference and error computation. Codabench converts only the final aggregate MAE to degrees for display on the leaderboard. Do not convert model predictions to degrees.

### How is the leaderboard score computed?

The public scorer:

1. nearest-neighbor resamples a coarser prediction to the target time length, when necessary;
2. computes the absolute difference between every predicted and reference value in radians;
3. averages over all evaluation examples, joints, and time points;
4. multiplies the aggregate MAE by `180 / π` and reports it in degrees.

Every target value therefore contributes equally to the final score. Lower is better.

### Does scoring use circular angle wrapping?

No. The scorer uses the direct absolute difference `abs(prediction - target)`. It does not wrap errors modulo `2π`.

### Which NeuralBench metric is comparable with the warm-up score?

NeuralBench's `test/mae` on the EMG2Pose `user_stage` split measures the same aggregate error in radians. Multiply it by `57.29578` to compare it with the Codabench warm-up score in degrees.

### Where is the exact scoring code?

The public **[`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/emg_pose/objective.py)** is the executable metric implementation used by Codabench. The selected evaluation dataset and leaderboard metric key are public in the **[warm-up phase config](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/phases/warmup/emg_pose/config.yaml)**.

## Scientific resources

**[Track description and active phase details](https://www.codabench.org/competitions/17984/)** · **[NeuralBench Track 04 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html)** · **[Competition implementation and worked models](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/emg_pose)**
