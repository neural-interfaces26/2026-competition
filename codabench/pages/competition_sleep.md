# Register in three steps

**Every participant must complete these steps, including each member of a team.**

1. Create a **[Codabench account](https://www.codabench.org/accounts/signup)** and sign in.
2. Open **[My Submissions](https://www.codabench.org/competitions/17983/#/participate-tab)**, accept the terms and conditions, then click **Register**. Your request will initially appear as **pending**.
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

# EEG/EMG Foundation Challenge 2026 · Track 03 - Sleep onset

_Estimate how long remains before stable sleep begins from continuous wearable EEG recorded at home._

Track 03 asks models to predict, at each point in a **continuous four-channel EEG recording**, the number of seconds remaining until the first stable N2 epoch. It evaluates whether sleep-onset patterns learned from existing recordings generalize to participants who were never seen during training.

**More information:** **[Explore Track 03 on the main competition website →](https://neural-interfaces26.github.io/tracks.html#track-3)** · **[Open the preparation guide on NeuralBench →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html)**

---

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/sleep-onset.gif" alt="Continuous wearable EEG used to predict the time remaining until stable N2 sleep" width="640">
</p>

## Track at a glance

|                    |                                                   |
| ------------------ | ------------------------------------------------- |
| **Input**          | Continuous four-channel home EEG                  |
| **Output**         | Seconds remaining until the first stable N2 epoch |
| **Evaluation**     | Participants absent from training                 |
| **Ranking metric** | Binned mean absolute error in seconds             |
| **Direction**      | Lower is better                                   |

---

## Scientific task

### From wearable EEG to sleep-onset timing

**Reliable sleep-onset estimates from lightweight home headbands could make longitudinal sleep monitoring less dependent on full laboratory polysomnography.**

The task is to predict, at each point in a continuous four-channel home EEG recording, the number of seconds remaining until the first stable N2 epoch. Because recordings are collected in everyday settings, models must remain robust to night-to-night variability, motion artifacts, changes in electrode impedance, and occasional channel dropout.

Evaluation is performed on unseen participants. This tests whether patterns associated with the transition from wakefulness to sleep generalize across individuals rather than relying on participant-specific characteristics.

### What the model predicts

The model receives consecutive windows from a continuous wearable EEG recording and returns **one sleep-onset latency in seconds for each window**.

The target represents the time remaining until the recording reaches its first stable N2 epoch. The task therefore focuses on the temporal transition into sleep rather than reconstructing a complete sleep-stage hypnogram.

The NeuralBench preparation task uses five-second windows and targets capped at 600 seconds before onset.

### Ranking metric

> **Binned mean absolute error, reported in seconds. Lower is better.**

Absolute prediction errors are computed within four time-to-onset ranges:

- **0 to 40 seconds**
- **40 to 90 seconds**
- **90 to 300 seconds**
- **300 to 600 seconds**

The final **binned mean absolute error**, or **bMAE**, gives equal weight to these four ranges. This prevents the most frequent prediction horizons from dominating the score and ensures that performance close to onset and farther from onset contributes equally.

NeuralBench reports the corresponding local metric as `test/bmae`.

---

## Development and evaluation data

### Public preparation data and matched Muse competition cohorts

Before the competition opens, public polysomnography datasets provide the EEG signals and sleep annotations needed to reproduce the task and its baselines. These datasets support workflow validation, but their clinical hardware differs from the wearable Muse recordings used during the competition.

When submissions open, a labeled **Muse Sleep-Onset training set** will provide the direct training counterpart to a separate **Muse hidden-evaluation set**. Both cohorts use the same four-channel wearable hardware, home-recording protocol, and sleep-onset target.

### Public preparation data

The default NeuralBench dataset is **Sleep-EDF Expanded**, registered as `Kemp2000Analysis`. It contains overnight polysomnography from 78 participants, with up to two nights recorded per participant.

NeuralBench also supports additional public sleep datasets recommended by the organizers, including:

- **Sleep-EDF Expanded**
- **PhysioNet Challenge 2018**
- **HMC Sleep Staging**

These datasets allow participants to:

- Validate continuous EEG loading
- Generate time-to-onset targets
- Reproduce the bMAE evaluation
- Train EEGNet and foundation-model baselines
- Test participant-level data splits

These public datasets use different recording systems from the competition data. Their role before the competition is to provide a reproducible preparation environment rather than to replicate the final wearable-device distribution exactly.

### 2026 labeled training data: Muse Sleep-Onset training set

The competition training set is provided by **Muse** and contains continuous EEG recorded at home with a four-channel Muse headband.

It provides:

- **Four-channel wearable EEG**
- **256 Hz sampling rate**
- **Multiple recordings per participant**
- **Sleep-onset annotations**
- **Labeled data for training and local validation**
- **BIDS format**

The training and hidden-evaluation cohorts are expected to contain **approximately 200 participants or fewer in total**, with multiple recordings per participant. The final participant count and allocation between the two cohorts remain to be confirmed.

### 2026 hidden evaluation: Muse Sleep-Onset evaluation set

A separate Muse cohort is reserved for hidden competition evaluation. It follows the same hardware, home-recording protocol, and target definition as the labeled Muse training set.

The evaluation set contains:

- **Participants absent from the training set**
- **Multiple home recordings per participant**
- **The same four-channel Muse EEG configuration**
- **The same sleep-onset prediction objective**
- **Confidential evaluation labels**

This participant-level separation tests whether the model captures physiological patterns that generalize across individuals while remaining robust to night-to-night variability and the artifacts encountered in everyday recordings.

> **Data use:** Original dataset licenses and access conditions remain in force. Conditions applying to the Muse competition data are defined in the _Terms_ tab.

---

## Prepare with NeuralBench

### Reproduce the public baselines first

NeuralBench provides the matching `eeg sleep_onset` task, the public sleep-data pipeline, the target construction, and two model baselines:

- **EEGNet**, a compact task-specific convolutional model
- **REVE**, an EEG foundation model adapted to sleep-onset prediction

Reproducing these baselines validates the complete local workflow:

1. Download the default public Sleep-EDF dataset
2. Prepare and cache the continuous EEG windows
3. Verify the task with a short debug run
4. Reproduce the EEGNet baseline
5. Reproduce the REVE foundation-model baseline
6. Compare the resulting bMAE scores with the reference results

The public baseline scores are **preparation references**, not official hidden-evaluation results.

When the competition opens, the same NeuralBench task will provide access to the labeled Muse training data. Participants can then retrain or adapt their models to the matched four-channel wearable recordings before packaging a trained submission for Codabench.

**[Follow the Track 03 preparation guide on NeuralBench →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html)**

**[Read the NeuralBench sleep-onset task reference →](https://facebookresearch.github.io/neuroai/neuralbench/tasks/eeg/sleep_onset.html)**

---

## Track 03 awards

### Up to $6,000 in cash awards

| Final position |                                  Award |
| -------------- | -------------------------------------: |
| **1st place**  | **$2,000 or a remote Muse internship** |
| **2nd place**  |                             **$2,000** |
| **3rd place**  |                             **$2,000** |

The winning team may choose between the **$2,000 first-place cash prize** and a **remote internship with Muse**. Final internship arrangements will be announced separately.

Awards are subject to participant eligibility and the reproducibility audit described in the _Terms_ tab.

**[Review the Track 03 awards and conditions on the main competition website →](https://neural-interfaces26.github.io/prizes.html#award-track-3)**

---

## Track partners

### Track sponsor and scientific institution

<a href="https://choosemuse.com/" target="_blank" rel="noopener noreferrer" title="Muse"><img src="https://neural-interfaces26.github.io/assets/img/logos/muse.svg" alt="Muse" height="34" style="vertical-align:middle;"></a>

**Muse** sponsors Track 03, leads its scientific development, and provides the labeled training and hidden-evaluation cohorts recorded with its wearable EEG platform.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>
