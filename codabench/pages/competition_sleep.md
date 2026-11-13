# Track 03 · Sealed evaluation update

> **Automatic rescoring in progress (+2 extra submissions).** Following a technical update of the Track 03 sealed evaluation on November 13, all previously Finished Track 03 sealed submissions are being rerun automatically without using additional submission quota; the leaderboard updates as they finish. To compensate for this interruption, the Track 03 total sealed allowance has been increased by **2 extra submissions** (from **5 to 7 total submissions** per team; the 1-per-day limit is unchanged). Read the [Scientific FAQ](scientific_faq_sleep.md#which-submissions-are-being-rescored).

# Get ready for the sealed final

> **Track 03 is now open!**. **Registration is closed.** Approved participants should select **Sealed Final Phase** when submitting. The final deadline is **Saturday, November 21 at 23:59 CET (UTC+01:00), 2026**. For quotas and score selection, read [Terms](terms.md); for local debugging with hidden sealed logs, read the [Submission Guide](participate.md).

# Before building a submission

> To be evaluated successfully and appear on the leaderboard, every uploaded model must follow the **strict Codabench submission contract**. Before preparing and uploading your model, open **Get Started → Submission Guide** in the menu on the left. It defines the Python inference contract, output shape, trained weights, optional files, and validation workflow. The same guide and its executable examples are public in the **[2026 competition repository](https://github.com/neural-interfaces26/2026-competition)**.

---

# After uploading: when will my score appear?

> **Your submission may remain queued before evaluation starts.** All four tracks share the same evaluation queue, so your score may not appear immediately. You can leave the page and return later; refreshing the Codabench webpage will not accelerate the evaluation process.

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

## Track resources and next steps

**[Explore the full Track 03 description on the main website →](https://neural-interfaces26.github.io/tracks.html#track-3)** · **[Choose a repository workflow →](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/README.md)** · **[Use the NeuralBench start kit and public baselines →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html)** · **[Review prizes and conditions →](https://neural-interfaces26.github.io/prizes.html#award-track-3)**

For packaging and uploads, use **Get Started → Submission Guide**. For current dates and submission limits, use **Phases**. For eligibility, data use, and the reproducibility audit, use **Terms**.

---

## Track partners

### Sponsor and scientific institution

<a href="https://choosemuse.com/" target="_blank" rel="noopener noreferrer" title="Muse"><img src="https://neural-interfaces26.github.io/assets/img/logos/muse.svg" alt="Muse" height="34" style="vertical-align:middle;"></a>

**Muse** sponsors Track 03, leads its scientific development, and provides its labeled training and hidden evaluation cohorts.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>
