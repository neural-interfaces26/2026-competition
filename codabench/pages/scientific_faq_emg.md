# Scientific FAQ · Track 04

Answers to recurring scientific questions, mostly from Discord, about Track 04. The **Track description** tab and its public **[Markdown source](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/competition_emg.md)** define the phase specifications. The public **[Track 04 repository](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/emg_pose)** contains the executable task, metric, data adapter, and worked models.

## Warm-up and sealed evaluation

### What changes between the warm-up and sealed phases?

| | Warm-up | Sealed final |
|---|---|---|
| **Evaluation data** | Public [EMG2Pose `user_stage` test split](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html#split-and-model-selection), selected by the [warm-up phase config](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/phases/warmup/emg_pose/config.yaml) | Private [2026 Meta Reality Labs evaluation cohort](https://neural-interfaces26.github.io/tracks.html#dataset-track-4) |
| **Generalization tested** | Unseen combinations of users and movement stages that are each represented elsewhere in the public data | New users, new movement stages, and unseen user-stage combinations |
| **Ranking metric** | Mean absolute angular error, computed from radian predictions and reported in degrees | The same metric and unit conversion |
| **Role** | Public development proxy | Final competition ranking, as defined in the [competition timeline](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/timeline.md) |

Warm-up scores are indicative only. Although both phases use the same prediction and metric contract, their datasets and generalization shifts differ, so their scores are not directly comparable.

### Does warm-up test entirely unseen users?

No. Warm-up uses the public EMG2Pose `user_stage` test split. Its users and movement stages occur elsewhere in the public data, but their specific combinations are held out. The sealed phase adds entirely new users and movement stages.

### Will participants receive any data from the sealed cohort?

No. The raw recordings, pose trajectories, participant metadata, and labels are not released or directly accessible. During evaluation, the submitted model receives only task-formatted EMG tensor batches.

### Will more public Track 04 data be released before the sealed phase?

No additional public Track 04 release is currently announced. Public EMG2Pose remains the direct development counterpart of the hidden cohort. Participants may also use eligible external data as explained below.

## Data and preprocessing

### What is the sealed evaluation dataset?

It is a separate **2026 Meta Reality Labs cohort** recorded under the same protocol as public EMG2Pose, pairing 16-channel wrist sEMG at 2 kHz with 20 UmeTrack hand-joint angles. It is not a hidden split of the public dataset. The [website dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-4) gives the disclosed acquisition facts, while the [NeuralBench Track 04 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html) documents the public task and split.

### Which sealed-set details are not disclosed?

Only the published input, output, and preprocessing contract. The exact cohort composition, movement-stage composition, and evaluation ordering are not part of the model contract. Use the metadata supplied at evaluation time rather than hard-coding dataset counts or batch composition.

### Is preprocessing the same in warm-up and sealed evaluation?

Yes. Sealed evaluation follows the same task pipeline as warm-up: raw 16-channel sEMG at **2 kHz**, without filtering, notch filtering, baseline correction, scaling, or clipping. Evaluation uses non-overlapping **5-second windows**, and windows containing invalid inverse-kinematics target frames are excluded. The exact settings are public in the **[NeuralBench task config](https://github.com/facebookresearch/neuroai/blob/main/neuralbench-repo/neuralbench/tasks/emg/pose/config.yaml)**. The competition invokes that pipeline through its public **[dataset adapter](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/emg_pose/datasets/emg2pose.py)**.

### May we use NinaPro or other public EMG datasets?

Yes. Pre-training or training on publicly available, redistributable datasets is allowed, subject to their original licenses and the competition **[Terms](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/terms.md)**. The sealed test set and closed clinical datasets are not allowed. External corpora must be declared in the final method description for the reproducibility audit.

### May a model assume fixed input dimensions or batch size?

Use `meta["n_chans"]`, `meta["n_times"]`, `meta["sfreq"]`, and `meta["n_joints"]` when constructing the model. The evaluation signal is fixed to the published 16-channel, 2 kHz protocol, but batch size and ordering are not part of the contract.

## Predictions and scoring

### What must the model predict?

For an input batch `X` of shape `(B, C, T)`, `predict(X)` must return continuous joint-angle trajectories with shape `(B, n_joints, T')`, where `n_joints = 20`. Predictions must be in **radians**. `T'` may equal the input length or be coarser.

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
