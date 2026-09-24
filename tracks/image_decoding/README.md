# Track 01 — EEG-to-Image

Decode which image a subject viewed from a single EEG epoch, by **retrieval**
in a frozen image-embedding space (e.g. DINOv2).

- **Input:** torch batch `X` `(B, C, T)`, already on `meta["device"]`.
- **Output:** `predict(X) -> (B, D)` — predicted image *embeddings*, ranked
  against the test candidate pool by cosine similarity.
- **Ranking metric:** top-5 retrieval accuracy (top-1 reported alongside).
- **Objective:** `Image-decoding` · output size `meta["n_outputs"]` = `D`.

## Data

Pick what you train on with `-d` (one-time `benchopt prepare` download):

| `-d` selector | What it is |
|---|---|
| `Image[study=gifford2022large]` | THINGS-EEG2 (Gifford2022Large) — public proxy, **default** |
| `Image[study=grootswagers2022human]`, `Image[study=xu2024alljoined]`, `Image[study=xu2025alljoined]` | alternative public studies |
| `Simulated` | tiny synthetic set — contract check only, no download |

## Two starting kits

Both paths finish with the same upload: a `submission.py` + weights. Pick by
your architecture — see the [participant guide](https://neural-interfaces26.github.io/participant-guide.html).

### Benchopt — run experiments & package (this repo)

The track is a [benchopt](https://benchopt.github.io) benchmark — the same code
Codabench runs. Prepare the data and train the linear baseline into a
ready-to-upload submission with the starter config:

```bash
benchopt prepare tracks/image_decoding --config tracks/image_decoding/starter.yml
benchopt run     tracks/image_decoding --config tracks/image_decoding/starter.yml
```

Baselines (each a solver **and** a valid submission) live in
[`solvers/`](solvers/) — full details in [`solvers/README.md`](solvers/README.md):

| Solver | What it shows |
|---|---|
| `MeanEmbedding` | the contract with no weights and no training (retrieval floor) |
| `Mean-Ridge` | a scikit-learn model; `save_model` writes a joblib dump |
| `Torch-Linear` | the same model in PyTorch, with its own Adam loop in `fit` |
| `EEGNet-CLIP` | braindecode EEGNet trained with a CLIP retrieval loss, end-to-end |

### NeuralBench — tasks, datasets & reference baselines

[NeuralBench](https://facebookresearch.github.io/neuroai/neuralbench/) defines
the task, public splits, and reference baselines. Reproduce the start kit:

```bash
pip install neuralbench
neuralbench eeg image --download
neuralbench eeg image --prepare
neuralbench eeg image -m eegnet --debug
```

Reference results on THINGS-EEG2 (`Gifford2022Large`) — development data,
**not** Codabench warm-up scores:

| Baseline | Top-5 accuracy |
|---|---|
| Chance | 2.22 ± 0.31 |
| EEGNet | 28.13 ± 0.14 |
| REVE (frozen probe) | 84.75 ± 0.38 |

Full guide: [Track 01 on NeuralBench ↗](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html).

## Develop your own model

A submission is one `submission.py` with `class Solver(CompetSolver)`:

- **`load_model(meta) -> model`** (required) — build your architecture, load the
  shipped weights from `meta["submission_dir"]`, return a model exposing
  `predict(X)` (in eval mode, on `meta["device"]`).
- `fit(model, train_loader)` (optional) — train locally; Codabench never calls it.
- `save_model(model, path)` (optional) — write weights; a training run then
  packages `outputs/<Solver.name>/` (`submission.py` + weights), ready to zip.

`meta` provides `n_chans`, `n_times`, `sfreq`, `ch_names`, `chs_info`,
`n_outputs` (= `D`), and `device`. `predict(X)` must return `(B, D)` float
embeddings in the target space.

Fastest loop — copy a baseline from `solvers/`, edit `load_model` / `fit`, then:

```bash
benchopt run tracks/image_decoding -s MyModel -o "Image-decoding[training=True]"
```

Full submission contract, `meta` keys, and packaging:
[`codabench/pages/participate.md`](../../codabench/pages/participate.md) and the
repo [README](../../README.md#develop--train-your-model-with-benchopt).
