# Registration

> You must **first register** for Track 02 - BCI Decoding before you can participate. **Registration takes three steps.** Open **Get Started → Registration Guide** in the menu on the left. Follow the approval instructions and, if applicable, the additional setup required for teams.

---

# Before building a submission

> To be evaluated successfully and appear on the leaderboard, every uploaded model must follow the **strict Codabench submission contract**. Before preparing and uploading your model, open **Get Started → Submission Guide** in the menu on the left. It defines the submission contract, including the required model architecture, trained weights, optional files, and validation workflow.

---

# After uploading: when will my score appear?

> **Your submission may remain queued before evaluation starts.** All four tracks share the same evaluation queue, so your score may not appear immediately. You can leave the page and return later; refreshing the Codabench webpage will not accelerate the evaluation process.

---

# Track 02 · BCI Decoding

> **Sealed-phase specification.** The description, task contract, and ranking metric below define the final three-command phase. Warm-up currently uses a two-class motor-imagery task on Dreyer 2023 instead. Phase-specific differences are detailed below.

_Decode a user’s intended mental command reliably across recording sessions._

Given a short 47-channel recording window, predict one of three cued commands: **kinesthetic motor imagery, mental calculation, or word association**. The channels comprise 43 EEG, 2 EMG, and 2 EOG signals sampled at 500 Hz. Models receive labeled calibration sessions from each evaluation participant, then predict later sessions from those same participants. This longitudinal split isolates within-user, cross-session drift without additional calibration.

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/bci-decoding.gif" alt="Three cued mental commands decoded from EEG across recording sessions" width="640" style="display:block;max-width:100%;height:auto;margin:1.25rem auto;">
</p>

## Sealed-phase task contract

|                          |                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------ |
| **Input**                | Short windows from 43 EEG, 2 EMG, and 2 EOG channels at 500 Hz                       |
| **Prediction**           | One class index for motor imagery, mental calculation, or word association           |
| **Objective**            | Maintain command decoding across recording sessions                                  |
| **Generalization shift** | Later sessions from the same users                                                   |
| **Sealed split**         | Labeled early sessions provide calibration, while later-session labels remain hidden |

## Sealed-phase ranking metric

> **Balanced accuracy averaged across subject, session, and context cells. Higher is better.**

Within each subject, session, and context cell, balanced accuracy is the unweighted mean of recall across the three commands. The official score is then averaged across cells, so every subject-session-context combination contributes equally regardless of its number of evaluation windows. Plain accuracy is reported separately but does not determine the ranking.

## Development, warm-up, and sealed data

**Development and training.** You may train your model on any of the organizer-recommended public datasets in the **[main website’s dataset directory](https://neural-interfaces26.github.io/tracks.html#datasets)** or other data permitted by the Terms.

**Warm-up evaluation.** Warm-up submissions on Codabench are currently being evaluated on a subset of the public **Dreyer 2023** dataset. This evaluation subset matches the test set in the [Track 02 NeuralBench start kit](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html) when run with its Dreyer configuration: Part B participants 61 to 81 form the test partition used for Codabench evaluation. Participants 1 to 60 and 82 to 87 form the training pool, from which 20% are assigned to validation by participant using random state 33. This is a temporary two-class motor-imagery proxy, ranked by balanced accuracy computed across all evaluation windows. It does not yet use the sealed phase's three commands or subject-session-context cell aggregation. Plain accuracy is reported separately.

The **[Track 02 NeuralBench competition guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html)** selects Dreyer 2023 with `--dataset dreyer2023` to match Codabench warm-up. The underlying NeuralBench task default and the **[reported public baseline scores](https://neural-interfaces26.github.io/participant-guide.html#baseline-track-2)** use **Stieger 2021** instead. Because the warm-up test data and labels are public, training overlap, overfitting, or leakage is possible. Warm-up scores are indicative only and support submission validation and iteration. Only the sealed phase determines the final ranking.

> **Upcoming data release.** The public 2026 Graz and BrainHero training data described in the **[Track 02 dataset entry](https://neural-interfaces26.github.io/tracks.html#dataset-track-2)** will be released soon. Codabench warm-up evaluation will then switch from the temporary Dreyer task and pooled metric to the new Track 02 data, three-command output, and cell-averaged balanced accuracy used by the sealed phase.

**Sealed evaluation.** Codabench uses the private later-session split of the 2026 Graz and BrainHero dataset described in the **[main website’s dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-2)**. It is uploaded for the sealed phase, and its later-session recordings and labels remain hidden.

## Track resources and next steps

**[Explore the full Track 02 description on the main website →](https://neural-interfaces26.github.io/tracks.html#track-2)** · **[Use the optional NeuralBench start kit and public baselines →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html)** · **[Review prizes and conditions →](https://neural-interfaces26.github.io/prizes.html#award-track-2)**

For packaging and uploads, use **Get Started → Submission Guide**. For current dates and submission limits, use **Phases**. For eligibility, data use, and the reproducibility audit, use **Terms**.

---

## Track partners

### Sponsor and scientific institutions

<a href="https://ai.meta.com/research/brain-ai/" target="_blank" rel="noopener noreferrer" title="Meta FAIR Brain and AI"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/meta_brainai.png" alt="Meta FAIR Brain and AI" height="34" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="30" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.isae-supaero.fr/" target="_blank" rel="noopener noreferrer" title="ISAE-SUPAERO"><img src="https://neural-interfaces26.github.io/assets/img/logos/isae-supaero.svg" alt="ISAE-SUPAERO" height="30" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.labri.fr/" target="_blank" rel="noopener noreferrer" title="LaBRI and University of Bordeaux"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/bordeaux.png" alt="LaBRI and University of Bordeaux" height="30" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://ethz.ch/" target="_blank" rel="noopener noreferrer" title="ETH Zurich"><img src="https://neural-interfaces26.github.io/assets/img/logos/eth-zurich.svg" alt="ETH Zurich" height="30" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://institutducerveau-icm.org/" target="_blank" rel="noopener noreferrer" title="Paris Brain Institute"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/icm.png" alt="Paris Brain Institute" height="30" style="vertical-align:middle;"></a>

**Meta FAIR Brain & AI** sponsors Track 02 and its cash awards. **Inria Bordeaux** leads the scientific development and provides the 2026 dataset, with contributions from **ISAE-SUPAERO**, **LaBRI and the University of Bordeaux**, **ETH Zurich**, and the **Paris Brain Institute**.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>
