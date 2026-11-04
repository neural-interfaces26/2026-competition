# Scientific FAQ · Track 02

This page answers recurring scientific questions about the model-facing
contract. The **Track description** defines the phase specifications; the
public [`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/bci_decoding/objective.py)
and active phase configuration are the executable references. Packaging and
upload instructions are in **Get Started → Submission Guide**.

## Sealed evaluation

### How does the sealed phase differ from the completed warm-up?

Warm-up has ended. The public **PROTEUS** release remains the reference
training and local-validation data. Sealed evaluation tests later sessions
from represented participants and entirely new participants. The three-class,
41-channel preprocessing and streaming contract, session resets and
cell-averaged ranking metric match the completed PROTEUS warm-up. Historical
warm-up scores do not count toward the final ranking.

### What generalization does the sealed phase test? Is calibration provided?

It combines later sessions from participants represented in the public release
with recordings from entirely new participants. No sealed-participant calibration
session, label, or other data is supplied through `fit`: represented
participants contribute only their already-public sessions, while unseen
participants are evaluated solely through the label-free `predict` stream.
Exact cohort size, identities, session allocation, and evaluation order remain private.
The public [objective](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/bci_decoding/objective.py)
runs submissions inference-only on Codabench.

## Public data and preprocessing

### What is in the public PROTEUS release?

[NEMAR `nm000290`](https://nemar.org/dataset/nm000290) contains 112 runs from
14 sessions of 10 participants. Two participants contribute their first three
sessions and eight contribute their first session. Graz and BrainHero runs use
the same three commands as the competition. Source recordings contain 41 EEG
channels at 500 Hz and are distributed in EEG-BIDS format under CC BY 4.0. The
NeuralBench source selection is public in its
[PROTEUS dataset config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/motor_imagery/datasets/dreyer2026proteus.yaml).

Participant, session, task, interface, and run entities are available in the
public BIDS files for local splitting and analysis. They are retained by the
evaluator but not given to the model during Codabench inference.

### Will EMG or EOG channels be supplied to the model?

No. The competition input contains the **41 EEG channels only**. Do not build
the submission around the earlier 47-signal description or expect auxiliary
EMG/EOG at evaluation. The exact tensor is produced by the public
[competition adapter](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/bci_decoding/datasets/bci_studies.py)
from the [NeuralBench stream config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/_motor_imagery_stream/config.yaml).

### May resting-state recordings be used?

Yes. Eyes-open and eyes-closed runs may be used for training or analysis under
the dataset licence. They have no command cues, so the default task does not
turn them into classification examples and Codabench does not send them to
`predict`.

### Which public split supports local validation?

NeuralBench assigns participants to train, validation, and test with a
60/20/20 subject-level split and random state 33. The completed warm-up
evaluated its test participants; the split remains available for local
validation and does not reveal the sealed allocation. The parameters are in the
[NeuralBench stream config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/_motor_imagery_stream/config.yaml),
and the archived Codabench setup selects this public dataset in its
[warm-up phase config](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/phases/warmup/bci_decoding/config.yaml).

### What reaches `predict(X)`?

Each example is a 4-second window beginning 1 second after cue onset. Source
signals are notch-filtered at 50 and 60 Hz, band-pass filtered at 0.1–75 Hz at
their recorded rate, resampled to 120 Hz, and expressed in microvolts. No
learned scaling or clipping is applied. Evaluation input has shape
`(1, 41, 480)`. These values are fixed in the
[NeuralBench stream config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/_motor_imagery_stream/config.yaml).

Fixed signal preprocessing happens before streaming. “Causal” refers to the
model-facing contract: the model receives no future window or label.

### May I use my own preprocessing?

Yes. You may preprocess the public training data differently and perform
additional transformations inside `predict`. Codabench nevertheless supplies
the fixed tensor above; a submission cannot request raw 500 Hz sealed data.
Any evaluation-time transformation or fitted statistic must use only the
current or earlier windows of the current session, without labels. Public-data
use remains subject to the [Terms](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/terms.md).

## Streaming and adaptation

### Where is state reset, and in what order do windows arrive?

One stream is one **participant × session**. There is no shuffling within a
stream: runs are ordered by run number and windows are strictly chronological
within each run. The order of independent session streams is not exposed as a
model signal. State may persist across Graz and BrainHero runs in the same
session. Each new session starts from a fresh copy of the submitted model and
calls its optional `reset_state()`. The public
[streaming evaluator](https://github.com/neural-interfaces26/2026-competition/blob/main/benchmark_utils/streaming.py)
implements the one-window calls, ordering checks, model copy, and reset.

### What information does the model receive, and may it adapt?

`predict(X)` receives only the EEG tensor. Participant, session, run,
interface, and label metadata are not passed to the model. Causal, label-free
adaptation using the current and earlier windows of the current session is
allowed, including statistics-based alignment or gradient updates inside
`predict`. State, windows, statistics, and parameter updates must not cross the
session boundary. Worker time and memory limits still apply. Grouping metadata
are used internally by the [objective](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/bci_decoding/objective.py).

### What must the model return?

One integer class index per window with shape `(B,)`: `0` for kinesthetic
motor imagery, `1` for mental subtraction, and `2` for word generation. The
public [objective](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/bci_decoding/objective.py)
validates one prediction per window.

## Scoring

### How is the official score computed?

Balanced accuracy is computed separately for every participant × session ×
interface cell, then cell scores are averaged equally. Within a cell it is the
unweighted mean recall across the task classes. Pooled balanced accuracy and
plain accuracy are diagnostics only. The executable implementation is in
[`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/bci_decoding/objective.py)
and the shared [metric helpers](https://github.com/neural-interfaces26/2026-competition/blob/main/benchmark_utils/metrics.py).

### Why can NeuralBench and Codabench scores differ?

NeuralBench's `test/bal_acc_stream_mean` averages per session without
separating Graz and BrainHero. Codabench separates those interfaces before
averaging cells, so the two scores can differ even for identical predictions.

## Resources

**[Track description](https://www.codabench.org/competitions/17982/)** ·
**[PROTEUS on NEMAR](https://nemar.org/dataset/nm000290)** ·
**[NeuralBench guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html)** ·
**[Competition implementation](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/bci_decoding)**
