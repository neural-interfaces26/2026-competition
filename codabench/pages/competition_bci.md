---

# Register in three steps

**Every participant must complete these steps, including each member of a team.**

1. Create a **[Codabench account](https://www.codabench.org/accounts/signup)** and sign in.
2. Open **[My Submissions](https://www.codabench.org/competitions/17982/#/participate-tab)**, accept the terms and conditions, then click **Register**. Your request will initially appear as **pending**.
3. Complete and submit the **[registration form](https://forms.gle/p3t2V25nuQtVXyj9A)**. **If you are entering multiple tracks**, submit the form only once and select, in the form, every track for which you have registered.

You may access the form using a different email account, but make sure the email address you enter in the form matches the one associated with your Codabench account, as it is used to approve your registration automatically.

Return to **My Submissions** to confirm your registration. Refresh the page if needed..

# What if you are part of a team?

**If you are participating individually, you can disregard this section.**

0. Each team member must complete the three steps above individually (you won't have to indicate your team in the form). 
1. Internally, choose a Team Leader. 
2. The team leader must create one **[Codabench organization](https://www.codabench.org/profiles/organization/create/)** for the team, save it, and edit it to **add the other team members to it**. 

Remark 1: Each participant may belong to only one team (Codabench Organization) and cannot join or submit on behalf of multiple teams.

Remark 2: Submissions and submission quotas remain individual and are not shared across the team. During both the warm-up and sealed phases, each member submits through their own account, but their submissions are attributed to the team.

Remark 3: If the team wins, the team leader is responsible for submitting the code for the reproducibility audit and distributing the prize among team members.

---


# EEG/EMG Foundation Challenge 2026 · Track 02 - BCI Decoding

*Decode a user’s intended mental command reliably across recording sessions.*

Track 02 asks models to classify **three cued mental commands** from short EEG windows: kinesthetic motor imagery, mental calculation, and word association. It evaluates whether subject-specific decoding learned from earlier sessions remains accurate during later sessions without additional per-session calibration.

**More information:** **[Explore Track 02 on the main competition website →](https://neural-interfaces26.github.io/tracks.html#track-2)** · **[Open the preparation guide on NeuralBench →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html)**

---

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/bci-decoding.gif" alt="Three cued mental commands decoded from EEG across recording sessions" width="640" style="display:block;max-width:100%;height:auto;margin:1.25rem auto;">
</p>

## Track at a glance

| | |
|---|---|
| **Input** | Short windows sampled at 500 Hz from 43 EEG, 2 EMG, and 2 EOG channels |
| **Output** | One of three cued mental commands |
| **Evaluation** | Later sessions from users represented during calibration |
| **Ranking metric** | Balanced accuracy across subject, session, and context cells |
| **Direction** | Higher is better |

---

## Scientific task

### From initial calibration to durable BCI control

**A practical brain-computer interface should remain reliable when the same user returns on another day without requiring a complete recalibration.**

Track 02 isolates within-user, cross-session variability through a longitudinal six-session protocol. It tests robustness to changes in time of day, electrode replacement, physiology, recording context, and mental strategy while preserving access to labeled calibration sessions from each evaluation participant.

### What the model predicts

The model receives a short EEG window recorded while the participant performs one of three cued commands:

- **Kinesthetic motor imagery**
- **Mental calculation**
- **Word association**

It returns the active command as a three-class prediction. Models train on earlier labeled sessions and are evaluated on later sessions from the same participants.

Per-participant models are permitted when they use the provided metadata. Retraining or adapting on hidden later-session labels is not permitted.

### Ranking metric

> **Balanced accuracy averaged across subject, session, and context cells. Higher is better.**

Balanced accuracy gives equal importance to each command even when class counts differ. Scores are computed within the relevant subject, session, and recording-context cells before aggregation, preventing large cells from dominating the final ranking.

The corresponding NeuralBench metric key is `test/bal_acc`.

---

## Development and evaluation data

### A self-contained longitudinal competition dataset

The 2026 Graz and BrainHero dataset provides both labeled training data and hidden evaluation data. Additional public datasets cover complementary aspects of motor imagery, mental calculation, word association, and cross-session decoding.

### 2026 training and hidden evaluation: Graz and BrainHero EEG

The competition dataset is provided by **Inria Bordeaux** and contains:

- **20 participants**
- **6 sessions per participant**
- **3 cued mental commands**
- **47 channels: 43 EEG, 2 EMG, and 2 EOG**
- **500 Hz sampling rate**
- **Approximately 80 hours**
- **BIDS format**

The dataset is divided into two participant groups:

- For **10 training participants**, all six sessions are released with labels.
- For **10 evaluation participants**, sessions 1 to 3 provide labeled calibration data.
- Sessions 4 to 6 from the evaluation participants form the hidden test set.
- Hidden-session labels remain confidential throughout the competition.

The Graz and BrainHero contexts introduce protocol variation while retaining the same three-command decoding objective.

### Public development data

No single public dataset reproduces the complete three-command, six-session competition design. NeuralBench therefore provides several complementary datasets:

- **Stieger 2021**, registered as `Stieger2021Continuous`
  - 62 participants
  - 615 hours
  - 60-channel continuous motor-imagery EEG
  - Four classes covering left hand, right hand, both hands, and rest

- **Dreyer 2023**, registered as `Dreyer2023Large`
  - 87 participants
  - 127 hours
  - 27 EEG channels
  - Two-class motor imagery

- **Scherer 2015**, registered as `Scherer2015Individually`
  - 9 participants
  - 14 hours
  - Five cued mental tasks including arithmetic and letter association
  - Cross-session evaluation

- **Zyma 2019**, registered as `Zyma2019Electroencephalograms`
  - 36 participants
  - Mental calculation against a resting baseline

These public datasets are preparation resources. The official Graz and BrainHero dataset is the only source that combines all three commands with the exact longitudinal evaluation split.

> **Data use:** Each public dataset retains its original license. Conditions for the Graz and BrainHero competition dataset are defined in the *Terms* tab.

---

## Prepare with NeuralBench

### Validate the workflow before the competition data arrive

NeuralBench provides public analogues for the Track 02 task together with **EEGNet** and **REVE** baselines. The default `eeg motor_imagery` configuration uses Stieger 2021 and validates the model pipeline, but it does not reproduce the official three-command, cross-session competition split.

Use NeuralBench to:

1. Run the default motor-imagery task on Stieger 2021
2. Explore the complementary motor-imagery and mental-task datasets
3. Verify preprocessing, model outputs, and balanced-accuracy scoring
4. Reproduce the EEGNet or REVE baseline
5. Switch to the official Graz and BrainHero configuration when it becomes available

The public-data baseline is a **preparation reference**, not an official hidden-evaluation result. Official evaluation uses sessions 4 to 6 from the ten evaluation participants.

**[Follow the Track 02 preparation guide on NeuralBench →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html)**

**[Read the NeuralBench motor-imagery task reference →](https://facebookresearch.github.io/neuroai/neuralbench/tasks/eeg/motor_imagery.html)**

*The Participation tab contains the submission contract, local smoke-test workflow, packaging instructions, and upload procedure.*

---

## Track 02 awards

### $6,000 cash prize pool

| Final position | Cash prize |
|---|---:|
| **1st place** | **$2,000** |
| **2nd place** | **$2,000** |
| **3rd place** | **$2,000** |

Awards are subject to participant eligibility and the reproducibility audit described in the *Terms* tab.

**[Review the Track 02 awards and conditions on the main competition website →](https://neural-interfaces26.github.io/prizes.html#award-track-2)**

---

## Track partners

### Track sponsor and scientific institutions

<a href="https://ai.meta.com/research/brain-ai/" target="_blank" rel="noopener noreferrer" title="Meta FAIR Brain and AI"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/meta_brainai.png" alt="Meta FAIR Brain and AI" height="34" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="30" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.isae-supaero.fr/" target="_blank" rel="noopener noreferrer" title="ISAE-SUPAERO"><img src="https://neural-interfaces26.github.io/assets/img/logos/isae-supaero.svg" alt="ISAE-SUPAERO" height="30" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.labri.fr/" target="_blank" rel="noopener noreferrer" title="LaBRI and University of Bordeaux"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/bordeaux.png" alt="LaBRI and University of Bordeaux" height="30" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://ethz.ch/" target="_blank" rel="noopener noreferrer" title="ETH Zurich"><img src="https://neural-interfaces26.github.io/assets/img/logos/eth-zurich.svg" alt="ETH Zurich" height="30" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://institutducerveau-icm.org/" target="_blank" rel="noopener noreferrer" title="Paris Brain Institute"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/icm.png" alt="Paris Brain Institute" height="30" style="vertical-align:middle;"></a>

**Meta FAIR Brain & AI** sponsors Track 02 and its cash awards. **Inria Bordeaux** leads the track’s scientific development and provides the 2026 competition dataset, with contributions from **ISAE-SUPAERO**, **LaBRI and the University of Bordeaux**, **ETH Zurich**, and the **Paris Brain Institute**.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>