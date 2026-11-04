# Scientific FAQ · Track 04

This page answers recurring scientific questions about the model-facing
contract. The **Track description** defines the phase specifications; the
public [`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/emg_pose/objective.py)
and active phase configuration describe the public-data implementation.
Packaging and upload instructions are in **Get Started → Submission Guide**.

## Sealed evaluation

### How does the sealed phase differ from the completed warm-up?

Warm-up has ended. Public **EMG2Pose** remains the reference training and
local-validation data. The private Meta Reality Labs cohort tests new users,
movement stages and user-stage combinations. The preprocessing, independent
batched windows and radian outputs are unchanged. Sealed ranking uses **Mean
Angular MAE per subject (deg)**, averaging subjects equally, rather than the
warm-up's pooled angular MAE.
Historical warm-up scores do not count toward the final ranking. See the
[sealed dataset description](https://neural-interfaces26.github.io/tracks.html#dataset-track-4).

### What is disclosed about the sealed cohort?

It is a separate private cohort, not a hidden split of public EMG2Pose. It
includes movement stages outside the 29 public EMG2Pose stages, while keeping
the same joint-angle regression task. Raw recordings, pose trajectories,
participant metadata, labels, exact cohort size, stage composition, and
evaluation order remain private.

## Input and preprocessing

### What does the model receive?

The model receives contiguous, non-overlapping five-second windows of
16-channel wrist sEMG. No filtering, notch filtering, baseline correction,
learned scaling, or clipping is applied. MNE-loaded volts are deterministically
converted to microvolts (`×10⁶`). Incomplete windows and windows containing
invalid inverse-kinematics target frames are excluded. These settings come
from the
[NeuralBench pose-task config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/emg/pose/config.yaml)
and are loaded by the public
[competition adapter](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/emg_pose/datasets/emg2pose.py).

Construct the model from `meta["n_chans"]`, `meta["n_times"]`,
`meta["sfreq"]`, and `meta["n_joints"]`. The published protocol is fixed at
16 channels and 2 kHz; do not hard-code an evaluation batch size.

### May NinaPro or other public EMG datasets be used?

Yes, subject to their licences and the competition
[Terms](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/terms.md).
Declare every external corpus in the final method description for the
reproducibility audit. Private sealed data and non-permitted closed datasets
may not be used.

## Batching and adaptation

### How are windows delivered? Is their order meaningful?

Sealed evaluation presents independent five-second windows in ordinary
batches. The complete current window is available, so bidirectional
processing within it is permitted. Batch size, composition, and order are not
part of the model-facing contract. A batch may mix windows or recordings, and
recording identifiers and temporal continuity are not supplied to the model.
Submissions therefore must not infer chronology from row order or depend on
state carrying across `predict` calls. The public
[`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/emg_pose/objective.py)
implements this batched contract.

### Is unsupervised test-time adaptation allowed?

Yes. A model may compute label-free statistics or update parameters, including
with gradients, from the current batch inside `predict`. It may not use labels
or assume that adjacent rows or successive calls belong to the same recording.
The method must accept varying batch sizes and remain within the worker time
and memory limits.

## Prediction and scoring

### What must `predict(X)` return?

For `X` with shape `(B, 16, T)`, return 20 joint-angle trajectories with shape
`(B, 20, T_out)` in **radians**. `T_out` may equal `T` or be coarser;
Codabench nearest-neighbor resamples predictions to the target length before
scoring. The public
[`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/emg_pose/objective.py)
contains the executable output and resampling contract.

### Should submissions return radians or degrees?

Radians. Predictions and targets remain in radians during inference and error
computation. Only the final aggregate error is converted to degrees for the
leaderboard.

### How is the ranking score computed?

The sealed ranking column is **Mean Angular MAE per subject (deg)**
(`angular_mae_subject_mean`). Codabench first computes each subject's MAE over
all their valid examples, joints and time points, then averages those MAEs
with equal weight per subject and multiplies by `180 / π`. A subject with
more evaluation windows does not receive more weight. Errors are direct
absolute differences, without wrapping modulo `2π`. The aggregation is
implemented by the
[shared metric helpers](https://github.com/neural-interfaces26/2026-competition/blob/main/benchmark_utils/metrics.py)
and exposed by the Track 04
[`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/emg_pose/objective.py).

Pooled **Angular MAE (deg)** (`angular_mae`) and the standard deviation of
per-subject MAEs (`angular_mae_subject_std`) remain diagnostic. The pooled
score weights subjects by their number of valid target values and can differ
from the sealed ranking score.

NeuralBench's `test/mae` on the public EMG2Pose `user_stage` split is the
pooled warm-up error in radians; multiply it by `180 / π` to compare with
pooled Angular MAE, not with the subject-averaged sealed ranking. Historical
public baseline scores retain their original aggregation.

## Resources

**[Track description](https://www.codabench.org/competitions/17984/)** ·
**[NeuralBench guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html)** ·
**[Competition implementation](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/emg_pose)**
