# Registration

> **You must first register for Track 04 - EMG-to-Pose before you can participate. Registration takes three steps.** Open **Get Started → Registration** in the menu on the left. Follow the approval instructions and, if applicable, the additional setup required for teams.

---

# Before building a submission

> **To be evaluated successfully and appear on the leaderboard, every uploaded model must follow the strict Codabench submission contract.** Before preparing and uploading your model, open **Get Started → Participation** in the menu on the left. It defines the submission contract, including the required model architecture, trained weights, optional files, and validation workflow.

---

# Track 04 · EMG-to-Pose

_Predict continuous hand motion from electrical activity recorded at the wrist._

Given a window of **16-channel wrist surface EMG**, predict continuous trajectories for **20 UmeTrack hand-joint angles** in degrees. The sealed evaluation covers users and movement stages absent from training, plus unseen combinations of users and stages. This tests transfer across anatomy, wristband placement, and movement context.

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/emg-to-pose.gif" alt="Wrist surface EMG decoded into continuous hand-pose trajectories" width="640" style="display:block;max-width:100%;height:auto;margin:1.25rem auto;">
</p>

## Task contract

|                          |                                                                    |
| ------------------------ | ------------------------------------------------------------------ |
| **Input**                | One temporal window of 16-channel wrist sEMG                       |
| **Prediction**           | Continuous trajectories for 20 UmeTrack joint angles in degrees    |
| **Objective**            | Reconstruct hand pose throughout the input window                  |
| **Generalization shift** | New users, new movement stages, and unseen user-stage combinations |
| **Sealed split**         | Evaluation examples and labels remain hidden                       |

## Ranking metric

> **Mean absolute angular error in degrees. Lower is better.**

Codabench computes the absolute angular difference between every predicted and reference value, then averages over all evaluation examples, joints, and time points. If a model predicts on a coarser time axis, its output is nearest-neighbor resampled to the target length before scoring.

## Development, warm-up, and sealed data

**Development and training.** You may train your model on any of the organizer-recommended public datasets in the **[main website’s dataset directory](https://neural-interfaces26.github.io/tracks.html#datasets)** or other data permitted by the Terms.

**Warm-up evaluation.** Warm-up submissions on Codabench are currently being evaluated on a subset of the public **EMG2Pose** dataset. This evaluation subset matches the test set in the [Track 04 NeuralBench start kit](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html): rows marked `generalization = user_stage` form the Codabench evaluation partition and contain unseen combinations of users and movement stages represented elsewhere in the public data. EMG2Pose also underlies the **[reported NeuroPose baseline scores](https://neural-interfaces26.github.io/participant-guide.html#baseline-track-4)**. Because the data and labels are public, leakage is possible and warm-up scores are indicative only. Only the sealed phase determines the final ranking.

**Sealed evaluation.** Codabench switches to the private 2026 Meta Reality Labs cohort described in the **[main website’s dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-4)**. This cohort is uploaded only for the sealed phase, and its evaluation examples and labels remain hidden.

## Track resources and next steps

**[Explore the full Track 04 description on the main website →](https://neural-interfaces26.github.io/tracks.html#track-4)** · **[Use the optional NeuralBench start kit and public baseline →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html)** · **[Review prizes and conditions →](https://neural-interfaces26.github.io/prizes.html#award-track-4)**

For packaging and uploads, use **Get Started → Participation**. For current dates and submission limits, use **Phases**. For eligibility, data use, and the reproducibility audit, use **Terms**.

---

## Track partners

### Sponsor and scientific institutions

<a href="https://about.meta.com/reality-labs/" target="_blank" rel="noopener noreferrer" title="Meta Reality Labs"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/meta_reality.png" alt="Meta Reality Labs" height="34" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.imperial.ac.uk/" target="_blank" rel="noopener noreferrer" title="Imperial College London"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/imperial.png" alt="Imperial College London" height="34" style="vertical-align:middle;"></a>

**Meta Reality Labs** sponsors Track 04, co-leads its scientific development, and provides its 2026 hidden evaluation cohort. **Imperial College London** contributes to the scientific development and organization.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>
