# Sealed phase closed

**Thank you to everyone who participated!** Registration and submissions are closed. The sealed deadline was **Saturday, November 21 at 23:59 CET (UTC+01:00), 2026**.

Submissions uploaded with a submission ID before the deadline remain eligible for evaluation, even if evaluation starts or finishes afterwards. **Please do not cancel or delete queued or running attempts.** Organizers continue monitoring and recovering platform incidents. Scores remain provisional until the reproducibility audit.

**Next come the reproducibility audit and the winners’ ceremony at the Brain and Body Workshop at NeurIPS 2026 in Sydney on December 11–12.** See the [Timeline](timeline.md) and [Participation rules](terms.md).

---

# Track 03 · Sleep Onset

For focused answers about the public release, first-N2 target, preprocessing, stream boundaries, model state, inference metadata, and scoring, read the **[Track 03 Scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_sleep.md)**.

_Estimate how long remains before the first N2 sleep epoch from wearable EEG recorded at home._

Given a continuous **four-channel, 128 Hz home EEG** recording, predict from each five-second window the number of seconds remaining until the first N2 epoch, capped at 600 seconds. Windows arrive one at a time in chronological order. Models may use past context within the current recording, but never future windows, and their state is reset between recordings. Evaluation includes both new recordings from participants represented in training and recordings from completely unseen participants.

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/sleep-onset.gif?v=20260921muse" alt="Continuous wearable EEG used to predict the time remaining until the first N2 epoch" width="640" style="display:block;max-width:100%;height:auto;margin:1.25rem auto;">
</p>

## Sealed-phase task contract

|                          |                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------- |
| **Input**                | Chronological five-second windows from four-channel Muse EEG at 128 Hz            |
| **Prediction**           | One latency in seconds until the first N2 epoch per window, capped at 600 seconds |
| **Objective**            | Estimate approaching first N2 from the current and earlier windows only           |
| **Generalization shift** | New nights from seen participants and recordings from unseen participants         |
| **Sealed split**         | Separate seen-subject and unseen-subject evaluation groups                        |

## Sealed-phase ranking metric

> **Weighted binned mean absolute error, or W-bMAE, in seconds. Lower is better.**

Codabench computes mean absolute error separately within four true time-to-onset ranges: **[0, 40)**, **[40, 90)**, **[90, 300)**, and **[300, 600]** seconds. These ranges receive severity weights of **10×, 5×, 3×, and 1×**, respectively, so errors closer to sleep onset contribute more strongly.

W-bMAE is computed within each recording over its non-empty ranges. Recording scores are averaged equally within each group, separately for **seen subjects** (new recordings from people represented in training) and **unseen subjects** (people absent from training). The final ranking metric is the macro-average of these two scores, giving equal importance to night-to-night and inter-person generalization.

See the [Scientific FAQ scoring details](scientific_faq_sleep.md#scoring).

## Training data and sealed evaluation

**Public training and baselines.** The official reference baselines use **Muse Sleep-Onset EEG ([NEMAR nm000287](https://nemar.org/dataset/nm000287))**. Its public start kit and validation split remain available for local development and submission checks. You may also use other data permitted by the [Terms](terms.md). See the [warm-up and sealed baseline results](https://neural-interfaces26.github.io/participant-guide.html#prepare-track-3) and [competition kit](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/sleep_onset).

**Sealed evaluation.** Codabench uses **private Muse recordings from represented and unseen sleepers**. Four channels at 128 Hz, microvolt inputs without additional filtering or scaling, chronological five-second windows before first N2, and recording resets match the completed Muse warm-up. Scoring additionally gives equal weight to the seen and unseen groups. The cohort is described in the [dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-3); evaluation examples and labels remain hidden.

**Completed warm-up.** Warm-up ended November 1. Its public-data scores are historical and do not count toward the final ranking. The [Scientific FAQ](scientific_faq_sleep.md) explains the transfer to sealed evaluation.

> **NeuralBench version.** This public start kit requires NeuralBench 0.4.0. In the same Python environment, run `python -m pip install --upgrade "neuralbench==0.4.0"`; older versions do not provide `_sleep_onset_stream`.

## Track resources and audit

**[Explore the full Track 03 description on the main website →](https://neural-interfaces26.github.io/tracks.html#track-3)** · **[Choose a repository workflow →](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/README.md)** · **[Use the NeuralBench start kit and public baselines →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html)** · **[Review prizes and conditions →](https://neural-interfaces26.github.io/prizes.html#award-track-3)**

The **Submission Guide** retains the packaging contract and local replay instructions for reproducibility. New uploads are closed. For eligibility and the finalist audit, read **Terms**; for audit and ceremony dates, see **Timeline**.

---

## Track partners

### Sponsor and scientific institution

<a href="https://choosemuse.com/" target="_blank" rel="noopener noreferrer" title="Muse"><img src="https://neural-interfaces26.github.io/assets/img/logos/muse.svg" alt="Muse" height="34" style="vertical-align:middle;"></a>

**Muse** sponsors Track 03, leads its scientific development, and provides its labeled training and hidden evaluation cohorts.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>
