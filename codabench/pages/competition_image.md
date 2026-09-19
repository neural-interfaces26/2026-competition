# Registration

> **You must first register for Track 01 - EEG-to-Image before you can participate. Registration takes three steps.** Open **Get Started → Registration** in the menu on the left. Follow the approval instructions and, if applicable, the additional setup required for teams.

---

# Before building a submission

> **To be evaluated successfully and appear on the leaderboard, every uploaded model must follow the strict Codabench submission contract.** Before preparing and uploading your model, open **Get Started → Participation** in the menu on the left. It defines the submission contract, including the required model architecture, trained weights, optional files, and validation workflow.

---

# Track 01 · EEG-to-Image

_Identify a viewed natural image from a single EEG response._

Given one multichannel EEG epoch recorded during natural-image viewing, predict a **1536-dimensional DINOv2-giant image embedding**. Codabench compares that prediction with the frozen embeddings of the candidate image gallery. The sealed split contains participants and images absent from training, testing cross-participant and cross-stimulus transfer rather than memorization of a fixed catalogue.

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/eeg-to-image.gif" alt="EEG response ranked against a gallery of candidate natural images" width="640" style="display:block;max-width:100%;height:auto;margin:1.25rem auto;">
</p>

## Task contract

| | |
|---|---|
| **Input** | One multichannel EEG epoch |
| **Prediction** | One 1536-dimensional DINOv2-giant embedding |
| **Objective** | Retrieve the viewed image from the complete candidate gallery |
| **Generalization shift** | New participants and images |
| **Sealed split** | Evaluation participants and images are absent from training |

## Ranking metric

> **Top-5 retrieval accuracy. Higher is better.**

The candidate pool is the set of unique target-image embeddings in the evaluation split. For each EEG epoch, Codabench L2-normalizes the predicted embedding and every candidate embedding, ranks their cosine similarities, and counts the prediction as correct when the viewed image is among the five highest-ranked candidates. The score is the mean of these outcomes across all evaluation epochs. Top-1 accuracy is reported separately but does not determine the ranking.

## Evaluation data

> **Warm-up configuration pending.** The public evaluation dataset will be named here once finalized. Because the warm-up data are public, they may overlap with development data. Warm-up scores validate the submission workflow and support iteration, but they do not determine the final ranking.

Only the sealed phase determines the final ranking. It uses the private 2026 Track 01 cohort described in the **[main website’s dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-1)**. Its evaluation examples and labels remain hidden, preventing evaluation-set leakage.

## Track resources and next steps

**[Explore the full Track 01 description on the main website →](https://neural-interfaces26.github.io/tracks.html#track-1)** · **[Use the optional NeuralBench start kit and public baselines →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html)** · **[Review prizes and conditions →](https://neural-interfaces26.github.io/prizes.html#award-track-1)**

For packaging and uploads, use **Get Started → Participation**. For current dates and submission limits, use **Phases**. For eligibility, data use, and the reproducibility audit, use **Terms**.

---

## Track partners

### Sponsor and scientific institutions

<a href="https://www.alljoined.com/" target="_blank" rel="noopener noreferrer" title="Alljoined"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/alljoined.png" alt="Alljoined" height="34" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://ai.meta.com/research/brain-ai/" target="_blank" rel="noopener noreferrer" title="Meta FAIR Brain and AI"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/meta_brainai.png" alt="Meta FAIR Brain and AI" height="34" style="vertical-align:middle;"></a>

**Alljoined** sponsors Track 01 and provides its 2026 hidden evaluation cohort. **Meta FAIR Brain & AI** leads the track’s scientific development.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>
