# Register in three steps

**Every participant must complete these steps, including each member of a team.**

1. Create a **[Codabench account](https://www.codabench.org/accounts/signup)** and sign in.
2. Open **[My Submissions](https://www.codabench.org/competitions/17984/#/participate-tab)**, accept the terms and conditions, then click **Register**. Your request will initially appear as **pending**.
3. Complete and submit the **[registration form](https://forms.gle/p3t2V25nuQtVXyj9A)**. **If you are entering multiple tracks**, submit the form only once and select, in the form, every track for which you have registered.

You may access the form using a different email account, but make sure the email address you enter in the form matches the one associated with your Codabench account, as it is used to approve your registration automatically.

Return to **My Submissions** to confirm your registration. Refresh the page if needed.

# What if you are part of a team?

**If you are participating individually, you can disregard this section.**

0. Each team member must complete the three steps above individually (you won't have to indicate your team in the form).
1. Internally, choose a Team Leader.
2. The team leader must create one **[Codabench organization](https://www.codabench.org/profiles/organization/create/)** for the team, save it, and edit it to **add the other team members to it**.

Remark 1: Each participant may belong to only one team (Codabench Organization) and cannot join or submit on behalf of multiple teams.

Remark 2: Submissions and submission quotas remain individual and are not shared across the team. During both the warm-up and sealed phases, each member submits through their own account, but their submissions are attributed to the team.

Remark 3: If the team wins, the team leader is responsible for submitting the code for the reproducibility audit and distributing the prize among team members.

---

# Before preparing a submission

Before building or uploading a model, **read the Submission Guidelines in the Participation tab** (in the menu on the left). It is the authoritative source for the ZIP structure, `submission.py` contract, trained weights, and validation workflow.

---

# EEG/EMG Foundation Challenge 2026 · Track 04 - EMG-to-Pose

_Predict continuous hand motion from the electrical activity recorded by a wrist-worn muscle interface._

Track 04 asks models to regress trajectories of **20 hand-joint angles** from **16-channel wrist surface EMG**. It evaluates whether an interface trained on existing recordings remains accurate for new users, new movement stages, and user-stage combinations absent from training.

**More information:** **[Explore Track 04 on the main competition website →](https://neural-interfaces26.github.io/tracks.html#track-4)** · **[Open the preparation guide on NeuralBench →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html)**

---

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/emg-to-pose.gif" alt="Wrist surface EMG decoded into continuous hand-pose trajectories" width="640" style="display:block;max-width:100%;height:auto;margin:1.25rem auto;">
</p>

## Track at a glance

|                    |                                                                    |
| ------------------ | ------------------------------------------------------------------ |
| **Input**          | 16-channel wrist sEMG                                              |
| **Output**         | 20 UmeTrack joint-angle trajectories                               |
| **Evaluation**     | New users, new movement stages, and unseen user-stage combinations |
| **Ranking metric** | Mean absolute angular error in degrees                             |
| **Direction**      | Lower is better                                                    |

---

## Scientific task

### From muscle activity to hand kinematics

**Wearable hand interfaces need to decode movement without requiring every user to complete a large personal calibration session.**

Track 04 measures whether learned EMG representations transfer across differences in anatomy, wristband placement, and movement context. The evaluation separately examines performance on new users, new movement stages, and user-stage combinations that were not represented during training.

### What the model predicts

The model receives temporal windows of wrist sEMG and returns a dense sequence for each of the **20 UmeTrack joint angles**. The output describes the evolution of the hand pose throughout the window rather than assigning a single gesture label.

Predictions made on a coarser temporal grid are nearest-neighbor resampled to the target time axis before scoring.

### Ranking metric

> **Mean absolute angular error, reported in degrees. Lower is better.**

Absolute errors are averaged across predicted joints and time. The resulting score measures the angular difference between the predicted and reference hand trajectories.

NeuralBench logs the corresponding local `test/mae` in **radians**. Multiply this value by `57.29578` to express it in degrees.

---

## Development and evaluation data

### Training and hidden evaluation sets

Public EMG2Pose recordings support training and local validation. A separate 2026 cohort following the same signal-to-kinematics protocol is reserved entirely for hidden competition evaluation.

### Public development data: EMG2Pose

The direct training counterpart is the public **EMG2Pose** dataset. It is registered in NeuralBench as `Salter2024Emg2pose` and available from NEMAR as `NM000281`.

- **193 participants**
- **25,253 recordings**
- **370 hours**
- **29 movement stages**
- **16 wrist EMG channels**
- **2 kHz sampling rate**
- **BIDS format**
- **20 UmeTrack joint-angle targets**

Wrist sEMG is paired with trajectories of the same 20 UmeTrack joint angles used by the competition task. NeuralBench implements the paper’s sequence-to-sequence regression setting.

### 2026 hidden evaluation: EMG2Pose evaluation cohort

A new cohort provided by **Meta Reality Labs** follows the same protocol as the public EMG2Pose dataset, pairing 16-channel wrist sEMG with 20 UmeTrack joint-angle trajectories. Its labels remain confidential throughout the competition.

The evaluation covers:

- **Users absent from training**
- **Movement stages absent from training**
- **User-stage combinations absent from training**
- **Confidential evaluation labels**

This controlled split tests whether a model transfers beyond the anatomy, wristband placement, and kinematic contexts represented in its training data.

> **Data use:** Public EMG2Pose is released under CC BY-NC-SA 4.0. The UmeTrack hand model used for forward kinematics is released under CC BY-NC 4.0. Both licenses are non-commercial. Competition data conditions are defined in the _Terms_ tab.

---

## Prepare with NeuralBench

### Reproduce the public baseline first

NeuralBench provides the matching `emg pose` task, the public EMG2Pose data pipeline, and the **VEMG2Pose regression baseline**.

Reproducing this baseline validates the complete local workflow:

1. Download the public EMG2Pose dataset
2. Prepare and cache the signals and targets
3. Verify the pipeline with a local debug run
4. Reproduce the published regression baseline
5. Compare the resulting score with the reference result

The local baseline score is a **preparation reference**, not an official hidden-evaluation result.

**[Follow the Track 04 preparation guide on NeuralBench →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html)**

**[Read the NeuralBench hand-pose task reference →](https://facebookresearch.github.io/neuroai/neuralbench/tasks/emg/pose.html)**

---

## Track 04 awards

### $6,000 cash prize pool

| Final position | Cash prize |
| -------------- | ---------: |
| **1st place**  | **$2,000** |
| **2nd place**  | **$2,000** |
| **3rd place**  | **$2,000** |

Awards are subject to participant eligibility and the reproducibility audit described in the _Terms_ tab.

**[Review the Track 04 awards and conditions on the main competition website →](https://neural-interfaces26.github.io/prizes.html#award-track-4)**

---

## Track partners

### Track sponsor and scientific institutions

<a href="https://about.meta.com/reality-labs/" target="_blank" rel="noopener noreferrer" title="Meta Reality Labs"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/meta_reality.png" alt="Meta Reality Labs" height="34" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.imperial.ac.uk/" target="_blank" rel="noopener noreferrer" title="Imperial College London"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/imperial.png" alt="Imperial College London" height="34" style="vertical-align:middle;"></a>

**Meta Reality Labs** sponsors Track 04, co-leads its scientific development, and provides the 2026 hidden evaluation cohort. **Imperial College London** contributes to the track’s scientific development and organization.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>
