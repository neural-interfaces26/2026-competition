# Registration

> You must **first register** for Track 01 - EEG-to-Image before you can participate. **Registration takes three steps.** Open **Get Started → Registration Guide** in the menu on the left. Follow the approval instructions and, if applicable, the additional setup required for teams.

---

# Before building a submission

> To be evaluated successfully and appear on the leaderboard, every uploaded model must follow the **strict Codabench submission contract**. Before preparing and uploading your model, open **Get Started → Submission Guide** in the menu on the left. It defines the Python inference contract, output shape, trained weights, optional files, and validation workflow. The same guide and its executable examples are public in the **[2026 competition repository](https://github.com/neural-interfaces26/2026-competition)**.

---

# After uploading: when will my score appear?

> **Your submission may remain queued before evaluation starts.** All four tracks share the same evaluation queue, so your score may not appear immediately. Once started, Track 01 normally needs 2–3 additional minutes to load and prepare the evaluation data, plus the model's inference time. You can leave the page and return later; refreshing the Codabench webpage will not accelerate the evaluation process.

---

# Track 01 · EEG-to-Image

> **Sealed-phase specification.** The description, task contract, and ranking metric below define the final sealed phase. Warm-up uses a closely matched Top-5 retrieval proxy on public THINGS-EEG2 data. Its candidate set and aggregation are detailed below.

_Identify a viewed natural image from a single EEG response._

Given one multichannel EEG epoch recorded during natural-image viewing, predict a **1536-dimensional DINOv2-giant image embedding**. Codabench compares that prediction with the frozen embeddings of the candidate image gallery. The sealed split contains participants and images absent from training, testing cross-participant and cross-stimulus transfer rather than memorization of a fixed catalogue.

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/eeg-to-image.gif" alt="EEG response ranked against a gallery of candidate natural images" width="640" style="display:block;max-width:100%;height:auto;margin:1.25rem auto;">
</p>

## Sealed-phase task contract

|                          |                                                               |
| ------------------------ | ------------------------------------------------------------- |
| **Input**                | One multichannel EEG epoch                                    |
| **Prediction**           | One 1536-dimensional DINOv2-giant embedding                   |
| **Objective**            | Retrieve the viewed image from the complete candidate gallery |
| **Generalization shift** | New participants and images                                   |
| **Sealed split**         | Evaluation participants and images are absent from training   |

## Sealed-phase ranking metric

> **Top-5 retrieval accuracy. Higher is better.**

For each EEG epoch, Codabench L2-normalizes the predicted embedding and the frozen DINOv2-giant embeddings in its held-out candidate gallery, then ranks their cosine similarities. A query is correct when the viewed image is among the five highest-ranked candidates. Predictions for repeated presentations of the same image are aggregated within each subject before retrieval, then the subject-level results are averaged so that every subject contributes equally. Top-1 accuracy is reported separately but does not determine the ranking.

## Development, warm-up, and sealed data

**Development and training.** You may train your model on any of the organizer-recommended public datasets in the **[main website’s dataset directory](https://neural-interfaces26.github.io/tracks.html#datasets)** or other data permitted by the Terms. The public **[Track 01 competition directory](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/image_decoding)** contains the executable contract and metric, dataset adapters, editable worked solvers, Benchopt configs, and NeuralBench checkpoint wrapper.

**Warm-up evaluation.** Warm-up submissions on Codabench are currently being evaluated on a subset of the public **THINGS-EEG2** dataset. This evaluation subset matches the test set in the [Track 01 NeuralBench start kit](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html): timelines marked as test form the Codabench evaluation partition, while 20% of the training timelines are assigned to validation using random state 33. Codabench ranks each epoch against the unique target-image embeddings in that test partition and reports pooled Top-5 accuracy across all evaluation epochs. Unlike the sealed metric and NeuralBench's headline `test/full_retrieval/top5_acc_subject-agg`, this temporary proxy does not aggregate repeated presentations within subjects. Top-1 accuracy is reported separately.

THINGS-EEG2 also provides the **[reported public baseline scores](https://neural-interfaces26.github.io/participant-guide.html#baseline-track-1)**. Because the test data and labels are public, leakage is possible and warm-up scores are indicative only. Only the sealed phase determines the final ranking.

**Sealed evaluation.** Codabench switches to the private 2026 Alljoined cohort described in the **[main website’s dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-1)**. This cohort is uploaded only for the sealed phase, and its participants, images, and labels remain hidden.

## Track resources and next steps

**[Explore the full Track 01 description on the main website →](https://neural-interfaces26.github.io/tracks.html#track-1)** · **[Choose a repository workflow →](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/README.md)** · **[Use the NeuralBench start kit and public baselines →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html)** · **[Review prizes and conditions →](https://neural-interfaces26.github.io/prizes.html#award-track-1)**

For packaging and uploads, use **Get Started → Submission Guide**. For current dates and submission limits, use **Phases**. For eligibility, data use, and the reproducibility audit, use **Terms**.

---

## Track partners

### Sponsor and scientific institutions

<a href="https://www.alljoined.com/" target="_blank" rel="noopener noreferrer" title="Alljoined"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/alljoined.png" alt="Alljoined" height="34" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://ai.meta.com/research/brain-ai/" target="_blank" rel="noopener noreferrer" title="Meta FAIR Brain and AI"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/meta_brainai.png" alt="Meta FAIR Brain and AI" height="34" style="vertical-align:middle;"></a>

**Alljoined** sponsors Track 01 and provides its 2026 hidden evaluation cohort. **Meta FAIR Brain & AI** leads the track’s scientific development.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>
