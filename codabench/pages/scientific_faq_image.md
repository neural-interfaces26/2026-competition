# Scientific FAQ · Track 01

Answers to recurring scientific questions, mostly from Discord, about Track 01. The **Track description** tab and its public **[Markdown source](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/competition_image.md)** define the phase specifications. The public **[Track 01 repository](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/image_decoding)** contains the executable task, metric, data adapters, and worked models.

## Warm-up and sealed evaluation

### What changes between the warm-up and sealed phases?

| | Warm-up | Sealed final |
|---|---|---|
| **Evaluation data** | Public [THINGS-EEG2 test split](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html#split-and-model-selection), selected by the [warm-up phase config](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/phases/warmup/image_decoding/config.yaml) | Private [2026 Alljoined evaluation cohort](https://neural-interfaces26.github.io/tracks.html#dataset-track-1) |
| **Generalization tested** | New images, with the public participants also represented in training | New participants and new images |
| **Top-5 metric** | One retrieval query per EEG epoch, pooled across the evaluation set by the current public [`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/image_decoding/objective.py) | Repeated presentations averaged per participant and image, producing one retrieval query per pair, as implemented for sealed evaluation in [PR #45](https://github.com/neural-interfaces26/2026-competition/pull/45) |
| **Role** | Public development proxy | Final competition ranking, as defined in the [competition timeline](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/timeline.md) |

Warm-up scores are indicative only and are not directly comparable with the sealed score or NeuralBench's subject-aggregated headline metric.

### Does the sealed phase test new participants, new images, or both?

Both. Every sealed participant and target image is absent from the public competition development and warm-up datasets. The final task therefore tests cross-participant and cross-stimulus generalization.

### Will participants receive any data from the sealed cohort?

No. The raw EEG, images, participant metadata, and labels are not released or directly accessible. During evaluation, the submitted model receives only preprocessed EEG tensor batches.

### Will more public Track 01 data be released before the sealed phase?

No additional Track 01 release is planned. The recommended public sources are THINGS-EEG1, THINGS-EEG2, Alljoined-1, and Alljoined-1.6M. Alljoined-1.6M is the closest public source to the hidden cohort.

## Data and preprocessing

### What is the sealed evaluation dataset?

It is a separate **2026 Alljoined cohort of 11 new participants**, recorded at 256 Hz with the same 32-channel Emotiv montage and the same natural-image acquisition protocol as public Alljoined-1.6M. It is not a hidden split of the public dataset. The [website dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-1) gives the disclosed acquisition facts, while the [NeuralBench Track 01 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html#where-the-competition-data-diverges) explains how the public datasets relate to this cohort.

### Which sealed-set details will not be disclosed?

The exact target images, gallery size, number of repetitions per image, and evaluation ordering will remain hidden. This is deliberate: none is required to satisfy the model contract, and disclosing them would create avoidable test-set information leakage and sealed-set-specific tuning. Alljoined-1.6M is the reference for the acquisition protocol and data structure, but its gallery and repetition counts are not guarantees for the sealed cohort.

### Should models expect a different or reduced channel montage?

No. Sealed evaluation uses the same 32-channel Emotiv montage and channel names as Alljoined-1.6M. No alternative or reduced montage is planned. Models should nevertheless construct their input layer from the metadata supplied at evaluation time.

### Is preprocessing the same in warm-up and sealed evaluation?

Yes. Preprocessing applied in sealed evaluation, like in warm-up, is 1.0-second epochs from **-0.2 to +0.8 seconds** around image onset, resampled to **120 Hz**, with the same filtering, baseline correction, robust scaling, and clipping. The exact settings are public in the **[NeuralBench task config](https://github.com/facebookresearch/neuroai/blob/main/neuralbench-repo/neuralbench/tasks/eeg/image/config.yaml)** and its **[defaults](https://github.com/facebookresearch/neuroai/blob/main/neuralbench-repo/neuralbench/defaults/config.yaml)**. The competition invokes that pipeline through its public **[dataset adapter](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/image_decoding/datasets/image_studies.py)**.

### May a model assume fixed channels, window length, or batch size?

Use `meta["n_chans"]`, `meta["ch_names"]`, `meta["n_times"]`, and `meta["sfreq"]` when constructing the model. The sealed montage and preprocessing are fixed as described above, but evaluation batch size may change.

### May `predict(X)` use statistics from its current test batch?

Yes, but do not rely on batch order or composition. The current warm-up ordering is not part of the contract. Sealed batches are shuffled, may mix participants, and may use a different batch size. A submission must not assume that adjacent epochs belong to one participant.

## Predictions and scoring

### What must the model predict?

For an input batch `X` of shape `(B, C, T)`, `predict(X)` must return one **1536-dimensional DINOv2-giant embedding** per epoch, with shape `(B, 1536)`. Codabench performs retrieval; submissions do not return image IDs or rankings.

### What is the candidate gallery?

It is the set of unique target-image embeddings in the evaluation partition. A query is correct when its viewed image is among the five gallery items with the highest cosine similarity. Its sealed size remains hidden as explained above.

### How are repeated presentations scored in the sealed phase?

For each participant and target image, the scorer:

1. averages the submitted embeddings across repeated presentations;
2. L2-normalizes the resulting embedding and the gallery embeddings;
3. ranks the complete gallery by cosine similarity;
4. computes Top-5 accuracy over the resulting participant-image queries.

The scorer does **not** normalize each repeated prediction before averaging. Top-1 accuracy is reported separately but does not determine the ranking.

### Which NeuralBench metric is comparable with the competition score?

`test/full_retrieval/top5_acc_subject-agg` follows the sealed aggregation. `val/batch_top5_acc` ranks only against items in one validation batch and is a model-selection signal, not a competition score. The current warm-up score is also different because it evaluates individual epochs without repeat aggregation.

### Where is the exact scoring code?

The public **[`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/image_decoding/objective.py)** is the executable implementation deployed by Codabench. The active warm-up deployment uses its pooled per-epoch proxy. The sealed aggregation is being finalized publicly in **[PR #45](https://github.com/neural-interfaces26/2026-competition/pull/45)** and will be merged and deployed before the sealed phase opens. Phase metric keys are mapped in the public **[Codabench config](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/phases/warmup/image_decoding/config.yaml)**.

## Scientific resources

**[Track description and active phase details](https://www.codabench.org/competitions/17974/)** · **[NeuralBench Track 01 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html)** · **[Competition implementation and worked models](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/image_decoding)**
