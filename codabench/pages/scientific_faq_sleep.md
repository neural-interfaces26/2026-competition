# Scientific FAQ · Track 03

This page answers recurring scientific questions about the model-facing
contract. The **Track description** defines the phase specifications; the
public [`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/sleep_onset/objective.py)
and active phase configuration are the executable references. Packaging and
upload instructions are in **Get Started → Submission Guide**.

## Sealed evaluation update · November 13

### Which submissions are being rescored?

Following a technical update of the Track 03 sealed evaluation, all
previously Finished Track 03 sealed submissions were rerun automatically in
place. Reruns kept the same submission IDs and used no additional quota, and the
leaderboard now reflects the updated scores. To compensate for this interruption,
all Track 03 teams receive **2 additional sealed submissions** (from 5 to 7
per team, with the 1-per-day limit unchanged). Warm-up results and the other
tracks are unaffected.

### May a model use regularities of the recordings instead of the EEG?

No. Predictions must come from the EEG. Exploiting regularities of how the
recordings were annotated or packaged is not allowed and is checked during the
reproducibility audit. Adapting within a recording, including using the time
elapsed since the recording's first window, remains allowed.

## Sealed evaluation

### How does the sealed phase differ from the completed warm-up?

Warm-up has ended. The public **Muse** release remains the reference training
and local-validation data. Sealed evaluation adds private recordings from
represented and unseen sleepers. Input preparation, chronological five-second
windows and recording resets are unchanged from the completed Muse warm-up.
The sealed score averages recording-level W-bMAE within each group, then
averages the seen and unseen group scores equally. Historical warm-up scores
do not count toward the final ranking.

### What generalization does the sealed phase test?

It combines new recordings from sleepers represented in public training data
with recordings from sleepers absent from it. Exact cohort size, identities,
group balance, recording lengths, and evaluation order remain private. The
public [competition adapter](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/sleep_onset/datasets/interaxon2026muse.py)
shows the public-data path; sealed data remain worker-side.

## Public data, target, and preprocessing

### What is in the public Muse release?

[NEMAR `nm000287`](https://nemar.org/dataset/nm000287) contains 540 at-home
recordings from 203 participants, totaling about 157.5 hours. Recordings use
TP9, AF7, AF8, and TP10 at 128 Hz and include one first-N2 annotation and
signal-quality flags. Its supplied split has 500 training recordings and 40
test recordings; every warm-up test participant is represented in training.
The release is licensed CC BY-NC-SA 4.0. NeuralBench encodes the supplied split
in its [Muse dataset config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/sleep_onset/datasets/interaxon2026muse.yaml).

### May NSRR or other public sleep datasets be used?

Yes. NSRR datasets under an active licence and other public datasets may be
used when their original terms permit this use. Declare every external corpus
in the final method description for the reproducibility audit, and follow the
competition [Terms](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/terms.md).
Private sealed data and non-permitted closed clinical datasets may not be used.

### What exactly is the target?

The supplied annotation is the earliest epoch labelled N2; the competition
does not require a full hypnogram or impose an additional multi-epoch rule.
For a window ending at time `t`, the target is `first_N2 - t` in seconds,
clipped to `[0, 600]`. The target construction is public in the
[NeuralBench stream config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/_sleep_onset_stream/config.yaml).

### What reaches `predict(X)`?

Recordings stay at their native 128 Hz and are cut into contiguous,
non-overlapping five-second windows. No filter, notch filter, learned scaling,
or clipping is applied; signals are expressed in microvolts. Each evaluation
input has shape `(1, 4, 640)`. These settings are defined by the
[NeuralBench stream config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/_sleep_onset_stream/config.yaml)
and invoked by the [competition adapter](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/sleep_onset/datasets/interaxon2026muse.py).

“Causal” refers to the model-facing contract: the model receives only the
current and previous windows of its current recording.

## Streaming and adaptation

### Where does a stream start and end?

Each sealed stream covers the part of one selected recording that precedes
first N2. There is no shuffling within a recording: windows arrive one at a
time in strict chronological order, and no post-onset window is shown. The
order of independent recordings is not exposed as a model signal; the
recording duration and future endpoint are also not passed to the model. The
public warm-up settings are visible in the [NeuralBench config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/_sleep_onset_stream/config.yaml).

### How is a new recording signaled, and may a model retain state?

Each recording starts from a fresh copy of the submitted model and calls its
optional `reset_state()`. A model may retain recurrent state, summaries,
elapsed-window count, or past windows within the current recording, subject to
the worker memory limit. All state must be cleared at the boundary. The public
[streaming evaluator](https://github.com/neural-interfaces26/2026-competition/blob/main/benchmark_utils/streaming.py)
implements the one-window calls, model copy, and reset.

### What metadata does the model receive?

`predict(X)` receives only the EEG tensor. Sleeper and recording identifiers,
seen/unseen group, labels, and future context are retained by the evaluator and
not passed to the model. Time elapsed since the stream began may be inferred
from the number of windows already received and is valid causal context.

### What must the model return?

One finite floating-point latency in seconds per input window, with shape
`(B,)`. During streamed evaluation `B = 1`. The public
[objective](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/sleep_onset/objective.py)
validates one prediction per window.

## Scoring

### How is the sealed W-bMAE ranking score computed?

Within each recording, MAE is computed in the true-target ranges **[0, 40)**,
**[40, 90)**, **[90, 300)**, and **[300, 600]** seconds. Non-empty ranges are
combined as `sum(weight × MAE) / sum(weight)` with weights **10, 5, 3, and 1**.
Recording scores are averaged equally within the seen group and within the
unseen group. The final score is `(seen_mean + unseen_mean) / 2`, so the
groups contribute equally regardless of their sizes. Windows are not pooled
across recordings to compute the ranking score. Lower is better.

NeuralBench's `test/wbmae_stream_mean` gives the public-data recording mean
used during warm-up; it has no seen/unseen group macro-average. Unweighted
bMAE and plain MAE are diagnostics. See the public
[metric helpers](https://github.com/neural-interfaces26/2026-competition/blob/main/benchmark_utils/metrics.py).

## Resources

**[Track description](https://www.codabench.org/competitions/17983/)** ·
**[Muse data on NEMAR](https://nemar.org/dataset/nm000287)** ·
**[NeuralBench guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html)** ·
**[Competition implementation](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/sleep_onset)**
