# Registration

> **You must first register for Track 03 - Sleep Onset before you can participate. Registration takes three steps.** Open **Get Started → Registration Guide** in the menu on the left. Follow the approval instructions and, if applicable, the additional setup required for teams.

---

# Before building a submission

> **To be evaluated successfully and appear on the leaderboard, every uploaded model must follow the strict Codabench submission contract.** Before preparing and uploading your model, open **Get Started → Submission Guide** in the menu on the left. It defines the submission contract, including the required model architecture, trained weights, optional files, and validation workflow.

---

# Track 03 · Sleep Onset

> **Sealed-phase specification.** The description, task contract, and ranking metric below define the final Muse phase. Warm-up currently uses Sleep-EDF and reports unweighted bMAE and MAE instead of the sealed W-bMAE. Phase-specific differences are detailed below.

_Estimate how long remains before the first N2 sleep epoch from wearable EEG recorded at home._

Given a short window from a continuous **four-channel, 128 Hz home EEG** recording, predict the number of seconds remaining until the first N2 epoch, capped at 600 seconds. Evaluation includes both new recordings from participants represented in training and recordings from completely unseen participants, testing robustness to both night-to-night and inter-person variability, as well as motion artifacts, impedance changes, and occasional channel dropout.

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/sleep-onset.gif?v=20260921muse" alt="Continuous wearable EEG used to predict the time remaining until the first N2 epoch" width="640" style="display:block;max-width:100%;height:auto;margin:1.25rem auto;">
</p>

## Sealed-phase task contract

|                          |                                                                           |
| ------------------------ | ------------------------------------------------------------------------- |
| **Input**                | One window from a continuous four-channel, 128 Hz home EEG recording      |
| **Prediction**           | One latency in seconds until the first N2 epoch, capped at 600 seconds    |
| **Objective**            | Estimate the transition from wakefulness to N2 sleep                      |
| **Generalization shift** | New nights from seen participants and recordings from unseen participants |
| **Sealed split**         | Separate seen-subject and unseen-subject evaluation groups                |

## Sealed-phase ranking metric

> **Weighted binned mean absolute error, or W-bMAE, in seconds. Lower is better.**

Codabench computes mean absolute error separately within four true time-to-onset ranges: **[0, 40)**, **[40, 90)**, **[90, 300)**, and **[300, 600]** seconds. These ranges receive severity weights of **10×, 5×, 3×, and 1×**, respectively, so errors closer to sleep onset contribute more strongly.

W-bMAE is reported separately for **seen subjects** (new recordings from people represented in training) and **unseen subjects** (people absent from training). The final ranking metric is the macro-average of these two scores, giving equal importance to night-to-night and inter-person generalization.

## Development, warm-up, and sealed data

**Development and training.** You may train your model on any of the organizer-recommended public datasets in the **[main website’s dataset directory](https://neural-interfaces26.github.io/tracks.html#datasets)** or other data permitted by the Terms.

**Warm-up evaluation.** Warm-up submissions on Codabench are currently being evaluated on a subset of the public **Sleep-EDF Expanded** dataset. This evaluation subset matches the test set in the [Track 03 NeuralBench start kit](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html): using random state 33, 46 participants are assigned to training, 16 to validation, and 16 to the test partition used for Codabench evaluation. This temporary proxy reports unweighted bMAE and plain MAE. The official W-bMAE and seen/unseen macro-average apply to the sealed Muse evaluation.

Sleep-EDF is both the default dataset in the **[Track 03 NeuralBench start kit](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html)** and the dataset used for the **[reported public baseline scores](https://neural-interfaces26.github.io/participant-guide.html#baseline-track-3)**. Because the test data and labels are public, training overlap, overfitting, or leakage is possible. Warm-up scores are indicative only and support submission validation and iteration. Only the sealed phase determines the final ranking.

> **Upcoming data release.** The public 2026 Muse training data described in the **[Track 03 dataset entry](https://neural-interfaces26.github.io/tracks.html#dataset-track-3)** will be released soon. Before Codabench warm-up switches from Sleep-EDF to the new Track 03 data, its scorer and the NeuralBench start kit will also be updated together to the agreed Muse warm-up W-bMAE scheme. Until then, unweighted bMAE remains the warm-up proxy.

**Sealed evaluation.** Codabench uses the private 2026 Muse cohort described in the **[main website’s dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-3)**. It combines new recordings from participants represented in training with recordings from completely unseen participants. Evaluation recordings and labels remain hidden.

## Track resources and next steps

**[Explore the full Track 03 description on the main website →](https://neural-interfaces26.github.io/tracks.html#track-3)** · **[Use the optional NeuralBench start kit and public baselines →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html)** · **[Review prizes and conditions →](https://neural-interfaces26.github.io/prizes.html#award-track-3)**

For packaging and uploads, use **Get Started → Submission Guide**. For current dates and submission limits, use **Phases**. For eligibility, data use, and the reproducibility audit, use **Terms**.

---

## Track partners

### Sponsor and scientific institution

<a href="https://choosemuse.com/" target="_blank" rel="noopener noreferrer" title="Muse"><img src="https://neural-interfaces26.github.io/assets/img/logos/muse.svg" alt="Muse" height="34" style="vertical-align:middle;"></a>

**Muse** sponsors Track 03, leads its scientific development, and provides its labeled training and hidden evaluation cohorts.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>
