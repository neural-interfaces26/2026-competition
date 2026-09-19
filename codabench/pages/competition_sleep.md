# Registration

> **You must first register for Track 03 - Sleep Onset before you can participate. Registration takes three steps.** Open **Get Started → Registration** in the menu on the left. Follow the approval instructions and, if applicable, the additional setup required for teams.

---

# Before building a submission

> **To be evaluated successfully and appear on the leaderboard, every uploaded model must follow the strict Codabench submission contract.** Before preparing and uploading your model, open **Get Started → Participation** in the menu on the left. It defines the submission contract, including the required model architecture, trained weights, optional files, and validation workflow.

---

# Track 03 · Sleep Onset

_Estimate how long remains before stable sleep begins from wearable EEG recorded at home._

Given a short window from a continuous **four-channel home EEG** recording, predict the number of seconds remaining until the first stable N2 epoch, capped at 600 seconds. Evaluation on participants absent from training tests whether sleep-onset patterns transfer across individuals despite night-to-night variability, motion artifacts, impedance changes, and occasional channel dropout.

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/sleep-onset.gif" alt="Continuous wearable EEG used to predict the time remaining until stable N2 sleep" width="640" style="display:block;max-width:100%;height:auto;margin:1.25rem auto;">
</p>

## Task contract

| | |
|---|---|
| **Input** | One window from a continuous four-channel home EEG recording |
| **Prediction** | One latency in seconds until the first stable N2 epoch, capped at 600 seconds |
| **Objective** | Estimate the transition from wakefulness to stable N2 sleep |
| **Generalization shift** | New participants and home-recording variability |
| **Sealed split** | Evaluation participants are absent from training |

## Ranking metric

> **Binned mean absolute error, or bMAE, in seconds. Lower is better.**

Codabench groups targets by their true time to onset: **[0, 40)**, **[40, 90)**, **[90, 300)**, and **[300, 600]** seconds. It computes mean absolute error within each non-empty range, then takes the unweighted mean across ranges. Short and long prediction horizons therefore contribute equally instead of being weighted by their number of windows. Plain MAE is reported separately but does not determine the ranking.

## Evaluation data

> **Warm-up configuration pending.** The public evaluation dataset will be named here once finalized. Because the warm-up data are public, they may overlap with development data. Warm-up scores validate the submission workflow and support iteration, but they do not determine the final ranking.

Only the sealed phase determines the final ranking. It uses the private 2026 Muse evaluation cohort described in the **[main website’s dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-3)**. Its participants, recordings, and labels remain hidden, preventing evaluation-set leakage.

## Track resources and next steps

**[Explore the full Track 03 description on the main website →](https://neural-interfaces26.github.io/tracks.html#track-3)** · **[Use the optional NeuralBench start kit and public baselines →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html)** · **[Review prizes and conditions →](https://neural-interfaces26.github.io/prizes.html#award-track-3)**

For packaging and uploads, use **Get Started → Participation**. For current dates and submission limits, use **Phases**. For eligibility, data use, and the reproducibility audit, use **Terms**.

---

## Track partners

### Sponsor and scientific institution

<a href="https://choosemuse.com/" target="_blank" rel="noopener noreferrer" title="Muse"><img src="https://neural-interfaces26.github.io/assets/img/logos/muse.svg" alt="Muse" height="34" style="vertical-align:middle;"></a>

**Muse** sponsors Track 03, leads its scientific development, and provides its labeled training and hidden evaluation cohorts.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>
