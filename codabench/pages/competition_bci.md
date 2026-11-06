# Get ready for the sealed final

> **Registration closed.** Approved participants should select **Sealed Final Phase** for submissions from **Thursday, November 5 at 00:01 to Saturday, November 21 at 23:59 CET (UTC+01:00), 2026**. No submissions are accepted during the November 2–4 transition. For quotas and score selection, read [Terms](terms.md); for local debugging with hidden sealed logs, read the [Submission Guide](participate.md).

# Before building a submission

> To be evaluated successfully and appear on the leaderboard, every uploaded model must follow the **strict Codabench submission contract**. Before preparing and uploading your model, open **Get Started → Submission Guide** in the menu on the left. It defines the Python inference contract, output shape, trained weights, optional files, and validation workflow. The same guide and its executable examples are public in the **[2026 competition repository](https://github.com/neural-interfaces26/2026-competition)**.

---

# After uploading: when will my score appear?

> **Your submission may remain queued before evaluation starts.** All four tracks share the same evaluation queue, so your score may not appear immediately. You can leave the page and return later; refreshing the Codabench webpage will not accelerate the evaluation process.

---

# Track 02 · BCI Decoding

For focused answers about the public release, split, preprocessing, stream boundaries, adaptation, inference metadata, and scoring aggregation, read the **[Track 02 Scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md)**.

_Decode a user’s intended mental command reliably across recording sessions._

Given a short 41-channel EEG window, predict one of three cued commands: **kinesthetic motor imagery, mental subtraction, or word generation**. Source recordings are sampled at 500 Hz and transformed by the published competition pipeline into 4-second, 120 Hz windows in microvolts. Sealed evaluation combines later sessions from represented participants with recordings from participants absent from the public release, testing both session drift and inter-person transfer.

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/bci-decoding.gif?v=20261104sealed" alt="Three cued mental commands decoded from EEG across recording sessions" width="640" style="display:block;max-width:100%;height:auto;margin:1.25rem auto;">
</p>

## Sealed-phase task contract

|                          |                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------ |
| **Input**                | Four-second windows from 41 EEG channels, resampled to 120 Hz                         |
| **Prediction**           | One class index for motor imagery, mental subtraction, or word generation             |
| **Objective**            | Maintain command decoding across sessions and participants                            |
| **Generalization shift** | Later sessions from represented participants and entirely new participants            |
| **Sealed split**         | Public early sessions support development; all sealed labels remain hidden            |

## Sealed-phase ranking metric

> **Balanced accuracy averaged across subject, session, and context cells. Higher is better.**

Within each subject, session, and context cell, balanced accuracy is the unweighted mean of recall across the three commands. The official score is then averaged across cells, so every subject-session-context combination contributes equally regardless of its number of evaluation windows. Plain accuracy is reported separately but does not determine the ranking.

See the [Scientific FAQ scoring details](scientific_faq_bci.md#scoring).

## Training data and sealed evaluation

**Public training and baselines.** The official reference baselines use **PROTEUS BCI Bordeaux ([NEMAR nm000290](https://nemar.org/dataset/nm000290))**. Its public start kit and validation split remain available for local development and submission checks. You may also use other data permitted by the [Terms](terms.md). See the [baseline results](https://neural-interfaces26.github.io/participant-guide.html#baseline-track-2) and [competition kit](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/bci_decoding).

**Sealed evaluation.** Codabench uses **later sessions and new participants from the PROTEUS acquisition programme**. The three commands, 41 EEG channels, four-second inputs at 120 Hz, chronological one-window calls and session resets match the completed PROTEUS warm-up. No labelled calibration data are supplied at evaluation. The cohort is described in the [dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-2); evaluation examples and labels remain hidden.

**Completed warm-up.** Warm-up ended November 1. Its public-data scores are historical and do not count toward the final ranking. The [Scientific FAQ](scientific_faq_bci.md) explains the transfer to sealed evaluation.

> **NeuralBench version.** This public start kit requires NeuralBench 0.4.0. In the same Python environment, run `python -m pip install --upgrade "neuralbench==0.4.0"`; older versions do not provide `_motor_imagery_stream`.

## Track resources and next steps

**[Explore the full Track 02 description on the main website →](https://neural-interfaces26.github.io/tracks.html#track-2)** · **[Choose a repository workflow →](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/README.md)** · **[Use the NeuralBench start kit and public baselines →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html)** · **[Review prizes and conditions →](https://neural-interfaces26.github.io/prizes.html#award-track-2)**

For packaging and uploads, use **Get Started → Submission Guide**. For current dates and submission limits, use **Phases**. For eligibility, data use, and the reproducibility audit, use **Terms**.

---

## Track partners

### Sponsor and scientific institutions

<a href="https://ai.meta.com/research/brain-ai/" target="_blank" rel="noopener noreferrer" title="Meta FAIR Brain and AI"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/meta_brainai.png" alt="Meta FAIR Brain and AI" height="34" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="30" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.isae-supaero.fr/" target="_blank" rel="noopener noreferrer" title="ISAE-SUPAERO"><img src="https://neural-interfaces26.github.io/assets/img/logos/isae-supaero.svg" alt="ISAE-SUPAERO" height="30" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.labri.fr/" target="_blank" rel="noopener noreferrer" title="LaBRI and University of Bordeaux"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/bordeaux.png" alt="LaBRI and University of Bordeaux" height="30" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://ethz.ch/" target="_blank" rel="noopener noreferrer" title="ETH Zurich"><img src="https://neural-interfaces26.github.io/assets/img/logos/eth-zurich.svg" alt="ETH Zurich" height="30" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://institutducerveau-icm.org/" target="_blank" rel="noopener noreferrer" title="Paris Brain Institute"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/icm.png" alt="Paris Brain Institute" height="30" style="vertical-align:middle;"></a>

**Meta FAIR Brain & AI** sponsors Track 02 and its cash awards. **Inria Bordeaux** leads the scientific development and provides the 2026 dataset, with contributions from **ISAE-SUPAERO**, **LaBRI and the University of Bordeaux**, **ETH Zurich**, and the **Paris Brain Institute**.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>
