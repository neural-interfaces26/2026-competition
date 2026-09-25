# Package a NeuralBench model for Codabench

This page covers the models used by the four NeuralBench competition start
kits: EEGNet for Tracks 01 to 03 and VEMG2Pose for Track 04. NeuralBench trains
the model; Codabench only loads the trained checkpoint and runs inference.
This is the manual **NeuralBench-to-Codabench path**: Benchopt is not used to
train the model or package the submission.

## What you need

Keep the best checkpoint produced by your NeuralBench training run. Depending
on how it was exported, it may be a Lightning `best.ckpt`, a NeuralBench model
state dictionary, or a plain model state dictionary. The submission wrappers
below accept all three formats and ignore training-only entries such as loss
parameters.

If you already have an exported `.pt` file, use it directly. No key conversion
is needed. NeuralBench currently deletes its temporary best checkpoint when a
run finishes unless `delete_checkpoints_on_exit=False` is set in the run
configuration, so enable checkpoint retention or export the model before the
run is cleaned up.

## Create the ZIP

From a clone of the public
[`2026-competition`](https://github.com/neural-interfaces26/2026-competition)
repository, choose the wrapper matching the model trained by the start kit:

| Track | NeuralBench model | Submission wrapper |
|---|---|---|
| 01 - EEG-to-Image | EEGNet | `tracks/image_decoding/submission_templates/neuralbench_eegnet_submission.py` |
| 02 - BCI Decoding | EEGNet | `tracks/bci_decoding/submission_templates/neuralbench_eegnet_submission.py` |
| 03 - Sleep Onset | EEGNet | `tracks/sleep_onset/submission_templates/neuralbench_eegnet_submission.py` |
| 04 - EMG-to-Pose | VEMG2Pose | `tracks/emg_pose/submission_templates/neuralbench_vemg2pose_submission.py` |

For the matching start-kit architecture, copy the template unchanged. Rename
it to `submission.py`, add your checkpoint as `weights.pt`, and zip exactly
those two files:

```bash
mkdir my_submission
cp tracks/<track>/submission_templates/<template> my_submission/submission.py
cp /path/to/your/checkpoint my_submission/weights.pt
zip -j my_submission.zip my_submission/submission.py my_submission/weights.pt
```

Open the ZIP before uploading. `submission.py` and `weights.pt` must be at its
root, not inside a directory. Upload the ZIP through **My Submissions** on the
corresponding Codabench track.

## Optional: replay inference locally with Benchopt

Benchopt is optional in this path. After assembling `my_submission`, you may
use it to run the packaged model locally against the competition contract and
real public track data before uploading. This is an inference-only check: it
does not call `fit`, retrain the NeuralBench model, or package the ZIP.

From the repository root:

```bash
benchopt prepare tracks/<track> -d "<real data source>"
COMPET_SUBMISSION_DIR="$PWD/my_submission" \
  benchopt run tracks/<track> -d "<real data source>" \
  -s "$PWD/my_submission/submission.py"
```

Use the real data source matching the checkpoint because these trained models
have fixed input dimensions and may not fit the smaller `Simulated` dataset.
Passing this check is useful but does not replace the official Codabench run.

## Why use the dedicated wrapper?

The wrapper reconstructs the same architecture and applies the task-specific
output contract:

- Track 01 returns DINOv2-space image embeddings.
- Track 02 converts EEGNet logits to class indices.
- Track 03 returns the NeuralBench output directly in seconds. It must not use
  the separate Benchopt baseline's normalized-target conversion.
- Track 04 reconstructs VEMG2Pose, supplies its temporal left context, returns
  `(batch, joints, time)`, and keeps predictions in radians. Codabench converts
  only the final aggregate error to degrees for the leaderboard.

These inference-only wrappers deliberately omit `fit` and `save_model`.
Codabench never trains a submission. The separate Benchopt training and
automatic-packaging route documented in the Submission Guide remains
available, but it is not part of this worked example.

## Scope

These wrappers are specific to the start-kit architectures above. If you
change an architecture or its preprocessing in NeuralBench, update the copied
`submission.py` to reproduce those choices. Foundation models and downstream
adapters such as REVE, probes, and LoRA require their corresponding inference
architecture and are not covered here.
