# Scientific FAQ · Track 01

This page answers recurring scientific questions about the model-facing
contract. The **Track description** defines the phase specifications; the
public [`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/image_decoding/objective.py)
is the executable scoring reference. Packaging and upload instructions are in
**Get Started → Submission Guide**.

## Sealed evaluation

### How does the sealed phase differ from the completed warm-up?

Warm-up has ended. The official baseline training data remain **THINGS-EEG2**;
sealed evaluation uses the private Alljoined cohort, with new participants
and images and a **32-channel Emotiv montage instead of 63 channels**.
Epoch timing, preparation and embedding outputs are unchanged. The completed
Update 1 warm-up reported subject-aggregated Top-5 and Top-1 alongside pooled
diagnostics; sealed ranking uses subject-aggregated Top-5. Public warm-up
scores do not count toward the final ranking.

### What is the sealed cohort, and will any of it be released?

It is a separate cohort of 11 new participants, recorded at 256 Hz with the
same 32-channel Emotiv montage and natural-image protocol as Alljoined-1.6M.
It is not a hidden split of a public dataset. Raw EEG, images, participant
metadata, labels, exact gallery size, repetition counts, and evaluation order
remain private. The model receives only task-formatted EEG batches through the
public [competition adapter](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/image_decoding/datasets/image_studies.py).

### May sealed images overlap public data, and may I train on `stim_test`?

Sealed participants and target images are absent from the public competition
development and warm-up data. You may train on public Alljoined-1.6M trials,
including the `stim_test` trials selected by NeuralBench's
[Alljoined dataset config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/image/datasets/xu2025alljoined.yaml),
and on other data allowed by the [Terms](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/terms.md).
Do not infer the sealed gallery from any public subset.

### Will more public Track 01 data be released?

No additional Track 01 release is planned. Recommended sources include
THINGS-EEG1, THINGS-EEG2, Alljoined-1, and Alljoined-1.6M; the latter is the
closest public source to the sealed acquisition protocol. They are listed in
the [public dataset directory](https://neural-interfaces26.github.io/tracks.html#datasets).

## Input and preprocessing

### What does the model receive?

Sealed evaluation uses the 32-channel Emotiv montage and channel names of
Alljoined-1.6M, compared with 63 channels in the public THINGS-EEG2 baseline.
Both use
1-second epochs from **-0.2 to +0.8 seconds** around image onset, resampled to
**120 Hz**, with the published NeuralBench image-task filtering, baseline
correction, robust scaling, and clipping. The exact pipeline is defined by the
[NeuralBench image-task config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/image/config.yaml)
and invoked by the [competition adapter](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/image_decoding/datasets/image_studies.py).

Construct the model from `meta["n_chans"]`, `meta["ch_names"]`,
`meta["n_times"]`, and `meta["sfreq"]`. Do not hard-code the warm-up montage
or batch size; evaluation batches may change size and may mix participants.
No more specific guarantee about the EEG reference is made beyond the same
acquisition protocol.

### Is Track 01 sequential? May a model use batch statistics or retain state?

Each epoch is an independent retrieval query. Sealed examples are shuffled,
and a batch may mix participants. Statistics from the current `predict` batch
are permitted, but batch order and composition are not part of the contract.
A model must not retain state or accumulated statistics across `predict`
calls, nor assume adjacent epochs belong to the same participant.
This is also how the public [competition objective](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/image_decoding/objective.py)
iterates over evaluation batches.

## Prediction and scoring

### What must `predict(X)` return?

One **1536-dimensional DINOv2-giant embedding** per epoch, with shape
`(B, 1536)`. Submissions return embeddings, not image IDs or rankings; see the
[`predict` contract and scorer](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/image_decoding/objective.py).

### What is the candidate gallery?

It is the set of unique target-image embeddings in the evaluation partition.
Predictions and gallery embeddings are L2-normalized and ranked by cosine
similarity. A query is correct when its target is among the five highest-ranked
candidates. Top-5 determines the ranking; Top-1 is diagnostic.
The executable retrieval implementation is in the public
[`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/image_decoding/objective.py).

### How are repeated presentations scored in the sealed phase?

Predictions are first averaged over repeated presentations of the same image
within each participant. Each average is then L2-normalized and ranked against
the complete gallery. The scorer does not normalize individual predictions
before averaging.
The subject-image grouping and averaging are implemented in the public
[`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/image_decoding/objective.py)
with the shared [metric helpers](https://github.com/neural-interfaces26/2026-competition/blob/main/benchmark_utils/metrics.py).

### Which NeuralBench score is comparable?

`test/full_retrieval/top5_acc_subject-agg` follows the sealed aggregation.
Batch-level retrieval metrics rank only against a batch and are model-selection
signals, not competition scores. Pooled epoch-level Top-5 is a diagnostic,
not the sealed ranking metric.

## Resources

**[Track description](https://www.codabench.org/competitions/17974/)** ·
**[NeuralBench guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html)** ·
**[Competition implementation](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/image_decoding)**
