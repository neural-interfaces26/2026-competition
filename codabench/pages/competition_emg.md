# Sealed phase closed

**Thank you to everyone who participated!** Registration and submissions are closed. The sealed deadline was **Saturday, November 21 at 23:59 CET (UTC+01:00), 2026**.

Submissions uploaded with a submission ID before the deadline remain eligible for evaluation, even if evaluation starts or finishes afterwards. **Please do not cancel or delete queued or running attempts.** Organizers continue monitoring and recovering platform incidents. Scores remain provisional until the reproducibility audit.

**Next come the reproducibility audit and the winners’ ceremony at the Brain and Body Workshop at NeurIPS 2026 in Sydney on December 11–12.** See the [Timeline](timeline.md) and [Participation rules](terms.md).

---

# Track 04 · EMG-to-Pose

For focused answers about the sealed cohort, preprocessing, external data, batching, model state, and radian-to-degree scoring, read the **[Track 04 Scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_emg.md)**.

_Predict continuous hand motion from electrical activity recorded at the wrist._

Given a window of **16-channel wrist surface EMG**, predict continuous trajectories for **20 UmeTrack hand-joint angles**. **Models must return radians; Codabench converts the resulting MAE to degrees for the leaderboard.** The sealed evaluation covers users and movement stages absent from training, plus unseen combinations of users and stages. This tests transfer across anatomy, wristband placement, and movement context.

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/emg-to-pose.gif" alt="Wrist surface EMG decoded into continuous hand-pose trajectories" width="640" style="display:block;max-width:100%;height:auto;margin:1.25rem auto;">
</p>

## Sealed-phase task contract

|                          |                                                                    |
| ------------------------ | ------------------------------------------------------------------ |
| **Input**                | One temporal window of 16-channel wrist sEMG                       |
| **Prediction**           | Continuous trajectories for 20 UmeTrack joint angles in radians    |
| **Objective**            | Reconstruct hand pose throughout the input window                  |
| **Delivery**             | Independent five-second windows in ordinary evaluation batches     |
| **Generalization shift** | New users, new movement stages, and unseen user-stage combinations |
| **Sealed split**         | Evaluation examples and labels remain hidden                       |

## Sealed-phase ranking metric

> **Mean Angular MAE per subject (deg). Lower is better.**

For each subject, Codabench averages the absolute prediction error over that subject's valid evaluation examples, joints, and time points. It then averages these subject scores equally and converts the result from radians to degrees. Each subject therefore has equal weight, regardless of how much evaluation data they contribute. This ranking score is `angular_mae_subject_mean`. Pooled Angular MAE and the standard deviation of subject scores are diagnostics. If a model predicts on a coarser time axis, its output is nearest-neighbor resampled to the target length before scoring.

See the [Scientific FAQ scoring details](scientific_faq_emg.md#prediction-and-scoring).

## Training data and sealed evaluation

**Public training and baselines.** The official reference baselines use **EMG2Pose**. Its public start kit and validation split remain available for local development and submission checks. You may also use other data permitted by the [Terms](terms.md). See the [warm-up and sealed baseline results](https://neural-interfaces26.github.io/participant-guide.html#prepare-track-4) and [competition kit](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/emg_pose).

**Sealed evaluation.** Codabench uses the private **2026 Meta Reality Labs cohort**, including new users, new movement stages and unseen user-stage combinations. The 16-channel, five-second input, ordinary independent batches and radian output match the completed EMG2Pose warm-up. Sealed ranking uses the equal-weight mean of per-subject MAEs rather than the warm-up's pooled MAE. Batch size and order are not guaranteed, and state must not carry across `predict` calls. The cohort is described in the [dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-4); evaluation examples and labels remain hidden.

**Completed warm-up.** Warm-up ended November 1. Its public-data scores are historical and do not count toward the final ranking. The [Scientific FAQ](scientific_faq_emg.md) explains the transfer to sealed evaluation.

## Track resources and audit

**[Explore the full Track 04 description on the main website →](https://neural-interfaces26.github.io/tracks.html#track-4)** · **[Choose a repository workflow →](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/README.md)** · **[Use the NeuralBench start kit and public baseline →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html)** · **[Review prizes and conditions →](https://neural-interfaces26.github.io/prizes.html#award-track-4)**

The **Submission Guide** retains the packaging contract and local replay instructions for reproducibility. New uploads are closed. For eligibility and the finalist audit, read **Terms**; for audit and ceremony dates, see **Timeline**.

---

## Track partners

### Sponsor and scientific institutions

<a href="https://about.meta.com/reality-labs/" target="_blank" rel="noopener noreferrer" title="Meta Reality Labs"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/meta_reality.png" alt="Meta Reality Labs" height="34" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.imperial.ac.uk/" target="_blank" rel="noopener noreferrer" title="Imperial College London"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/imperial.png" alt="Imperial College London" height="34" style="vertical-align:middle;"></a>

**Meta Reality Labs** sponsors Track 04, co-leads its scientific development, and provides its 2026 hidden evaluation cohort. **Imperial College London** contributes to the scientific development and organization.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>
