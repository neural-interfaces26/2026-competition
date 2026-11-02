# Get ready for the sealed final

> **Warm-up has ended. Registration remains open until November 5 at 23:59 CET (UTC+01:00).** Follow the [Registration Guide](registration.md), then select **Sealed Final Phase** for submissions from **Thursday, November 5 at 00:01 to Saturday, November 21 at 23:59 CET (UTC+01:00), 2026**. No submissions are accepted during the November 2–4 transition. For quotas and score selection, read [Terms](terms.md); for local debugging with hidden sealed logs, read the [Submission Guide](participate.md).

# Before building a submission

> To be evaluated successfully and appear on the leaderboard, every uploaded model must follow the **strict Codabench submission contract**. Before preparing and uploading your model, open **Get Started → Submission Guide** in the menu on the left. It defines the Python inference contract, output shape, trained weights, optional files, and validation workflow. The same guide and its executable examples are public in the **[2026 competition repository](https://github.com/neural-interfaces26/2026-competition)**.

---

# After uploading: when will my score appear?

> **Your submission may remain queued before evaluation starts.** All four tracks share the same evaluation queue, so your score may not appear immediately. Once started, Track 01 normally needs 2–3 additional minutes to load and prepare the evaluation data, plus the model's inference time. You can leave the page and return later; refreshing the Codabench webpage will not accelerate the evaluation process.

---

# Track 01 · EEG-to-Image

For focused answers about the sealed cohort, public-data use, preprocessing, batching, model state, and retrieval aggregation, read the **[Track 01 Scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_image.md)**.

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

For each participant and target image, Codabench first averages the predicted embeddings across repeated presentations. It then L2-normalizes the resulting embedding and the frozen DINOv2-giant embeddings in the held-out candidate gallery before ranking their cosine similarities. A query is correct when the viewed image is among the five highest-ranked candidates. Top-5 accuracy is pooled over the resulting participant-image queries. Top-1 accuracy is reported separately but does not determine the ranking.

See the [Scientific FAQ scoring details](scientific_faq_image.md#prediction-and-scoring).

## Training data and sealed evaluation

**Public training and baselines.** The official reference baselines use **THINGS-EEG2**. Its public start kit and validation split remain available for local development and submission checks. You may also use other data permitted by the [Terms](terms.md). See the [baseline results](https://neural-interfaces26.github.io/participant-guide.html#baseline-track-1) and [competition kit](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/image_decoding).

**Sealed evaluation.** Codabench uses the private **2026 Alljoined cohort**, with new participants and target images absent from the public development data. Its EEG uses the 32-channel Emotiv montage, rather than the 63-channel THINGS-EEG2 montage. The one-second epoch, 120 Hz preparation and embedding-output contract remain the same. Build the model from the supplied channel metadata rather than hard-coding the public montage. The cohort is described in the [dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-1); evaluation examples and labels remain hidden.

**Completed warm-up.** Warm-up ended November 1. Its public-data scores are historical and do not count toward the final ranking. The [Scientific FAQ](scientific_faq_image.md) explains the transfer to sealed evaluation.

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
