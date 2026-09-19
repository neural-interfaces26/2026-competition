# Register in three steps

**Every participant must complete these steps, including each member of a team.**

1. Create a **[Codabench account](https://www.codabench.org/accounts/signup)** and sign in.
2. Open **[My Submissions](https://www.codabench.org/competitions/17974/#/participate-tab)**, accept the terms and conditions, then click **Register**. Your request will initially appear as **pending**.
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

# EEG/EMG Foundation Challenge 2026 · Track 01 - EEG-to-Image

_Identify the natural image viewed by a participant from a single EEG response._

Track 01 asks models to predict a **1536-dimensional visual embedding** from a **single EEG epoch** recorded during natural-image viewing. It evaluates whether neural representations learned from training images transfer to entirely new images rather than memorizing a fixed stimulus catalogue.

**More information:** **[Explore Track 01 on the main competition website →](https://neural-interfaces26.github.io/tracks.html#track-1)** · **[Open the preparation guide on NeuralBench →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html)**

---

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/eeg-to-image.gif" alt="EEG response ranked against a gallery of candidate natural images" width="640" style="display:block;max-width:100%;height:auto;margin:1.25rem auto;">
</p>

## Track at a glance

|                    |                                                             |
| ------------------ | ----------------------------------------------------------- |
| **Input**          | A single multichannel EEG epoch                             |
| **Output**         | A 1536-dimensional DINOv2-giant image embedding             |
| **Evaluation**     | Images and participants absent from training                |
| **Ranking metric** | Top-5 retrieval accuracy against the complete candidate set |
| **Direction**      | Higher is better                                            |

---

## Scientific task

### From brain responses to visual content

**A useful visual decoder should recognize novel content rather than memorize a fixed catalogue of images encountered during training.**

Track 01 measures whether EEG representations preserve enough visual information to retrieve images that were never shown during model training. Research-grade THINGS-EEG recordings and consumer-grade Alljoined recordings introduce variability in participants and hardware while the frozen visual embedding space keeps the retrieval objective consistent.

### What the model predicts

The model receives one EEG epoch recorded while a participant views a natural image. It returns a **1536-dimensional embedding** aligned with the frozen `facebook/dinov2-giant` representation of that image.

The predicted embedding is compared with every image in the held-out candidate gallery. The model therefore retrieves visual content through a shared representation space rather than selecting from a fixed set of training labels.

### Ranking metric

> **Top-5 retrieval accuracy against the complete held-out candidate set. Higher is better.**

A prediction is correct when the viewed image appears among the five highest-ranked candidates. Accuracy is computed over the full test gallery and averaged across participants.

The comparable NeuralBench metric is `test/full_retrieval/top5_acc_subject-agg`. The validation metric `val/batch_top5_acc` only ranks images within individual batches and should not be compared with the competition score.

---

## Development and evaluation data

### Public development datasets and a matched hidden cohort

Four public EEG datasets support training and local validation. The hidden 2026 evaluation cohort follows the same hardware and natural-image protocol as **Alljoined-1.6M**, which is the closest direct training counterpart.

### Public development data

- **THINGS-EEG1**, registered as `Grootswagers2022Human`
  - 50 participants
  - 46 hours
  - 63 or 128 EEG channels
  - 1 kHz sampling rate

- **THINGS-EEG2**, registered as `Gifford2022Large`
  - 10 participants
  - 87 hours
  - 63 EEG channels
  - 1 kHz sampling rate
  - Default NeuralBench dataset for the `eeg image` task

- **Alljoined-1**, registered as `Xu2024Alljoined`
  - 8 participants
  - 64 EEG channels
  - 512 Hz sampling rate
  - Natural-image viewing protocol

- **Alljoined-1.6M**, registered as `Xu2025Alljoined`
  - 20 participants
  - 130 hours
  - 32-channel Emotiv EEG
  - 256 Hz sampling rate
  - Direct training counterpart to the hidden evaluation cohort

Together, these datasets cover research-grade and consumer-grade acquisition while preserving a common image-decoding objective.

### 2026 hidden evaluation: Alljoined evaluation cohort

A new **11-participant cohort** provided by **Alljoined** is reserved entirely for hidden evaluation. It uses the same 32-channel Emotiv hardware and natural-image protocol as Alljoined-1.6M.

The evaluation covers:

- **Participants absent from training**
- **Images absent from training**
- **Confidential image identities and evaluation labels**
- **The same hardware and protocol as the direct training counterpart**

This controlled cross-stimulus split tests whether a model retrieves visual content beyond the concepts and image identities represented in its training data.

> **Data use:** Each public dataset retains its original license. Conditions for the hidden competition cohort are defined in the _Terms_ tab.

---

## Prepare with NeuralBench

### Reproduce the public baselines first

NeuralBench provides the matching `eeg image` task, the four public development datasets, the frozen DINOv2-giant target space, and **EEGNet** and **REVE** baselines.

Reproducing these baselines validates the complete local workflow:

1. Download one or more public image-viewing EEG datasets
2. Prepare and cache the EEG epochs and image embeddings
3. Verify the pipeline with a local debug run
4. Reproduce the EEGNet or REVE baseline
5. Compare the full-gallery retrieval score with the reference result

If you begin with only one dataset, **Alljoined-1.6M** is the closest match to the hidden evaluation cohort.

The local baseline score is a **preparation reference**, not an official hidden-evaluation result.

**[Follow the Track 01 preparation guide on NeuralBench →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html)**

**[Read the NeuralBench image-decoding task reference →](https://facebookresearch.github.io/neuroai/neuralbench/tasks/eeg/image.html)**

---

## Track partners

### Track sponsor and scientific institutions

<a href="https://www.alljoined.com/" target="_blank" rel="noopener noreferrer" title="Alljoined"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/alljoined.png" alt="Alljoined" height="34" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://ai.meta.com/research/brain-ai/" target="_blank" rel="noopener noreferrer" title="Meta FAIR Brain and AI"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/meta_brainai.png" alt="Meta FAIR Brain and AI" height="34" style="vertical-align:middle;"></a>

**Alljoined** sponsors Track 01, contributes the public Alljoined datasets, provides the 2026 hidden evaluation cohort, and supports the internship opportunity. **Meta FAIR Brain & AI** leads the track’s scientific development and contributes its EEG representation-learning expertise.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>
