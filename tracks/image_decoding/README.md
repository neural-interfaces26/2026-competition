# Track 01: EEG-to-Image

Decode which image a participant viewed from one EEG epoch by retrieving its
embedding from a candidate image pool.

## Model contract

- **Input:** torch tensor `(B, C, T)` on `meta["device"]`.
- **Output:** `predict(X) -> (B, D)` float image embeddings, where
  `D = meta["n_outputs"]`.
- **Metric implemented here:** top-5 cosine-similarity retrieval accuracy,
  with top-1 reported alongside.
- **Benchopt objective:** `Image-decoding`.

See the Codabench **Track description** for the active phase data and official
ranking specification.

## Public data choices

| Benchopt `-d` selector | Data |
|---|---|
| `Image[study=gifford2022large]` | THINGS-EEG2, the default public proxy |
| `Image[study=grootswagers2022human]` | Grootswagers 2022 |
| `Image[study=xu2024alljoined]` | Alljoined 2024 |
| `Image[study=xu2025alljoined]` | Alljoined 2025 |
| `Simulated` | tiny synthetic contract check, with no download |

## Worked examples

### NeuralBench start kit

Use NeuralBench to explore the neurophysiology task, preprocessing, public
split, and reference model pipeline:

```bash
pip install neuralbench
neuralbench eeg image --download
neuralbench eeg image --prepare
neuralbench eeg image -m eegnet --debug
```

Reference development results on THINGS-EEG2, not Codabench warm-up scores:

| Baseline | Top-5 accuracy |
|---|---|
| Chance | 2.22 ± 0.31 |
| EEGNet | 28.13 ± 0.14 |
| REVE frozen probe | 84.75 ± 0.38 |

[Open the Track 01 NeuralBench guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html).
After training its EEGNet, use the shared
[NeuralBench-to-Codabench bridge](../README.md#package-a-neuralbench-checkpoint).

### Benchopt competition kit

Use the shared [Benchopt workflow](../README.md#develop-and-package-with-benchopt)
with `<track> = image_decoding`. `training.yml` selects THINGS-EEG2 and
exports the `Torch-Linear` submission.

Editable solvers live in [`solvers/`](solvers/):

| Solver | Purpose |
|---|---|
| `MeanEmbedding` | uploadable constant floor |
| `Mean-Ridge` | scikit-learn linear baseline with joblib weights |
| `Torch-Linear` | PyTorch linear baseline with `fit` and `save_model` |
| `EEGNet-CLIP` | end-to-end EEGNet with a retrieval loss |

## Adapt your own model

The track metadata adds `n_outputs`, the image-embedding dimension, to the
shared submission metadata. Your model must return `(B, n_outputs)` floats in
the target embedding space.

Override the training config's dataset with another selector from the table
above or a custom Benchopt dataset file:

```bash
benchopt run tracks/image_decoding \
  --config tracks/image_decoding/training.yml \
  -d "Image[study=xu2024alljoined]" -s MyModel
```

The [Submission Guide](../../codabench/pages/participate.md) defines the full
contract and ZIP layout.
