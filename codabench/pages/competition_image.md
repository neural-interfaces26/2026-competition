# Sealed phase closed

**Thank you to everyone who participated!** Registration and submissions are closed. The sealed deadline was **Saturday, November 21 at 23:59 CET (UTC+01:00), 2026**.

Submissions uploaded with a submission ID before the deadline remain eligible for evaluation, even if evaluation starts or finishes afterwards. **Please do not cancel or delete queued or running attempts.** Organizers continue monitoring and recovering platform incidents. Scores remain provisional until the reproducibility audit.

**Next come the reproducibility audit and the winners’ ceremony at the Brain and Body Workshop at NeurIPS 2026 in Sydney on December 11–12.** See the [Timeline](timeline.md) and [Participation rules](terms.md).

---

# Track 01 · EEG-to-Image

For focused answers about the sealed cohort, public-data use, preprocessing, batching, model state, and retrieval aggregation, read the **[Track 01 Scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_image.md)**.

_Identify a viewed natural image from a single EEG response._

Given one multichannel EEG epoch recorded during natural-image viewing, predict a **1536-dimensional DINOv2-giant image embedding**. Codabench compares that prediction with the frozen embeddings of the candidate image gallery. The sealed split contains participants and images absent from training, testing cross-participant and cross-stimulus transfer rather than memorization of a fixed catalogue.

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/eeg-to-image.gif" alt="EEG response ranked against a gallery of candidate natural images" width="640" style="display:block;max-width:100%;height:auto;margin:1.25rem auto;">
</p>

> **Sealed montage contract.** The sealed input has **32 Emotiv channels**, compared with **63 channels in THINGS-EEG2**. [**Alljoined-1.6M**](https://nemar.org/dataset/nm000134) is the public reference for this recording setup. Use it to reproduce and verify the submitted model’s channel handling; see the [montage adaptation guidance](scientific_faq_image.md#how-should-i-adapt-a-model-trained-on-a-different-montage).

## Sealed-phase task contract

|                          |                                                               |
| ------------------------ | ------------------------------------------------------------- |
| **Input**                | 32-channel Emotiv EEG, one second at 120 Hz: `(B, 32, 120)`      |
| **Prediction**           | One 1536-dimensional DINOv2-giant embedding                   |
| **Objective**            | Retrieve the viewed image from the complete candidate gallery |
| **Generalization shift** | New participants and images                                   |
| **Sealed split**         | Evaluation participants and images are absent from training   |

## Sealed-phase ranking metric

> **Top-5 retrieval accuracy. Higher is better.**

For each participant and target image, Codabench first averages the predicted embeddings across repeated presentations. It then L2-normalizes the resulting embedding and the frozen DINOv2-giant embeddings in the held-out candidate gallery before ranking their cosine similarities. A query is correct when the viewed image is among the five highest-ranked candidates. Top-5 accuracy is pooled over the resulting participant-image queries. Top-1 accuracy is reported separately but does not determine the ranking.

See the [Scientific FAQ scoring details](scientific_faq_image.md#prediction-and-scoring).

## Training data and sealed evaluation

**Public training and baselines.** The reported reference baselines use **THINGS-EEG2**, with 63 EEG channels. These public scores describe the development task; they do not establish compatibility with the sealed montage. You may also use other data permitted by the [Terms](terms.md). See the [baseline results](https://neural-interfaces26.github.io/participant-guide.html#baseline-track-1) and [competition kit](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/image_decoding).

**Public acquisition reference.** [**Alljoined-1.6M (NEMAR nm000134)**](https://nemar.org/dataset/nm000134) documents the consumer-grade Emotiv recording setup and provides the closest public match to the sealed cohort. Use its [NeuralBench dataset configuration](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/image/datasets/xu2025alljoined.yaml) to prepare matching public inputs for training, adaptation and local validation. The Scientific FAQ defines the competition's preprocessing, model inputs and scoring.

**Sealed evaluation.** Codabench uses the private **2026 Alljoined cohort**, with new participants and target images absent from the public development data. One-second epochs at 120 Hz and the 1536-dimensional embedding output are shared with the public task; the electrode montage changes. Evaluation examples and labels remain hidden. See the [dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-1).

**Completed warm-up.** Warm-up ended November 1. Its public-data scores are historical and do not count toward the final ranking. The [Scientific FAQ](scientific_faq_image.md) explains the transfer to sealed evaluation.

## Track resources and audit

**[Explore the full Track 01 description on the main website →](https://neural-interfaces26.github.io/tracks.html#track-1)** · **[Choose a repository workflow →](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/README.md)** · **[Use the NeuralBench start kit and public baselines →](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html)** · **[Review prizes and conditions →](https://neural-interfaces26.github.io/prizes.html#award-track-1)**

The **Submission Guide** retains the packaging contract and local replay instructions for reproducibility. New uploads are closed. For eligibility and the finalist audit, read **Terms**; for audit and ceremony dates, see **Timeline**.

---

## Track partners

### Sponsor and scientific institutions

<a href="https://www.alljoined.com/" target="_blank" rel="noopener noreferrer" title="Alljoined"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/alljoined.png" alt="Alljoined" height="34" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://ai.meta.com/research/brain-ai/" target="_blank" rel="noopener noreferrer" title="Meta FAIR Brain and AI"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/meta_brainai.png" alt="Meta FAIR Brain and AI" height="34" style="vertical-align:middle;"></a>

**Alljoined** sponsors Track 01 and provides its 2026 hidden evaluation cohort. **Meta FAIR Brain & AI** leads the track’s scientific development.

### Organizing institutions

<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>
