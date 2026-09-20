# How to submit your model

## Submission rules

Upload a **fully trained model as a ZIP** through the **My Submissions** tab.
Codabench will mount the extracted files as read-only, run the model in inference
mode, compute the track metric, and publish the score. It does not train
your model.

Your typical `my_submission.zip` contains:

```text
my_submission.zip
├── submission.py   # required: all Python inference code
├── weights.pt      # trained parameters, with any name or format
└── ...             # optional non-Python artifact like config.json
```

- Put every file at the **root of the ZIP**.
- `submission.py` must contain `class Solver(CompetSolver)`, the model
  architecture, model-specific preprocessing, and all Python inference code.
  Additional Python modules are not supported. See below for more details.
- Include every weight and optional non-Python artifact the model needs. Read
  them from `meta["submission_dir"]` in `submission.py`.

Nothing will be installed during evaluation, so `submission.py` may only
import what the image already carries: `torch`, `torchvision`, `torchaudio`,
`numpy`, `pandas`, `scikit-learn`, `mne`, `moabb`, `braindecode`, `benchopt`
and the `neuralset` / `neuralfetch` / `neuralbench` data stack. During
`load_model` and `predict`, do not train, do not download competition data,
and do not write into the submission directory.

---

## Submit in five steps

1. **Train and validate locally** with an optional NeuralBench start kit,
   directly through Benchopt, or with your own pipeline. Save the trained
   weights.
2. **Create `submission.py`** following the contract below.
3. **Create the ZIP.** For the example above, run:

   ```bash
   zip -j my_submission.zip submission.py weights.pt
   ```

   Add `config.json` or any other non-Python artifact your model needs.

4. **Check it locally first.** A failed upload still costs you one of the
   day's submissions, and most failures are caught in seconds:

   ```bash
   cp my_submission/* tracks/<track>/solvers/
   benchopt run tracks/<track> -d Simulated -s MyModel
   ```

   See [Test your submission locally](#test-your-submission-locally) for
   what this does and does not cover.

5. **Upload.** In **My Submissions**, select the active phase, upload the
   ZIP, and wait for **Finished**. If it fails, start with the first error
   in the ingestion log.

---

## The `submission.py` contract

In `submission.py`, you write ordinary PyTorch code. Benchopt provides the evaluation wrapper, but
you do not need to understand its internals. In `submission.py`, Codabench expects two components:

1. A named `class Solver(CompetSolver)` implementing `load_model(meta)`.
2. A model returned by `load_model` and exposing `predict(X)`.

### Contract 1: `class Solver(CompetSolver)` loads the trained model

In `class Solver(CompetSolver)`, **`load_model(self, meta) -> model` is
required**. It reconstructs the trained architecture, loads the shipped
weights, moves the model to the evaluation device, and returns it in
evaluation mode:

```python
import torch

from benchmark_utils.base_solver import CompetSolver

class Solver(CompetSolver):
    name = "MyModel"

    requirements = []

    def load_model(self, meta):
        # MyModel must be defined in this submission.py file or imported
        # from a dependency.
        model = MyModel(
            n_chans=meta["n_chans"],
            n_times=meta["n_times"],
        )
        weights = meta["submission_dir"] / "weights.pt"
        state_dict = torch.load(
            weights,
            map_location=meta["device"],
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model.to(meta["device"]).eval()

    # Optional local-training hooks. Codabench never calls them.
    # See "Practice 2: Train and package with Benchopt" below.
    def fit(self, model, train_loader):
        ...

    def save_model(self, model, path):
        ...
```

Each track's [`solvers/`](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks)
directory holds working implementations of this contract on real data —
the closest reference for your own `submission.py`.

Remark: `meta` is a plain Python dictionary created and passed to `load_model`
automatically. You do not create or upload it. It provides:

- `submission_dir`: read-only path to the files extracted from your ZIP
- `device`: CPU or GPU used for the model and input batches
- `n_chans`, `n_times`, `sfreq`, `ch_names`, `chs_info`: input description
- `n_outputs`, `n_classes`, or `n_joints`: track-specific output size

A fixed-size architecture normally uses `meta["n_chans"]` and
`meta["n_times"]`. A shape-agnostic model may infer these dimensions from `X`.

### Contract 2: The built model generates predictions

After `Solver.load_model(meta)` returns your model, Codabench calls that
required model's `predict(X)` method for every evaluation batch.

`X` is a PyTorch tensor already on `meta["device"]`, with shape `(B, C, T)`:

- `B`: batch size
- `C`: signal channels
- `T`: time samples

`predict(X)` must return the output required by the track:

| Track             | `predict(X)` must return                   | Output-size key           | Ranking metric           |
| ----------------- | ------------------------------------------ | ------------------------- | ------------------------ |
| 01 - EEG-to-Image | image embeddings `(B, D)`                  | `meta["n_outputs"]` = `D` | top-5 retrieval accuracy |
| 02 - BCI Decoding | one class index per window `(B,)`          | `meta["n_classes"]`       | balanced accuracy        |
| 03 - Sleep Onset  | seconds to sleep onset `(B,)` as floats    | `meta["n_outputs"]` = `1` | binned MAE               |
| 04 - EMG-to-Pose  | joint angles `(B, n_joints, T)` in degrees | `meta["n_joints"]`        | mean angular MAE         |

A PyTorch model can implement `predict` directly:

```python
class MyModel(torch.nn.Module):
    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return self(X)
```

---

## Behind the scenes: How Codabench and Benchopt evaluate your submission

[Benchopt](https://benchopt.github.io) runs each public track benchmark and
computes its metrics. Codabench manages uploads, workers, and leaderboards.

Only the sealed evaluation data and labels are hidden. The evaluation code
and model contract remain public:

|                                 | Warm-up phase                    | Sealed final phase                       |
| ------------------------------- | -------------------------------- | ---------------------------------------- |
| Evaluation data                 | development or public proxy data | held-out cohort, never released          |
| Benchmark code and worker image | public and fixed                 | same                                     |
| Objective and metric            | public                           | same                                     |
| `meta` keys and tensor contract | documented contract              | same contract, runtime values may differ |
| Submission ZIP                  | your trained model               | unchanged                                |

For portability, **use only `meta` and the provided batches**. Do not rely on
undocumented dataset names, paths, subject identifiers, or fixed dimensions.

---

## Get some practice

Track 03 ships a three-step progression in its
[`solvers/`](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/sleep_onset/solvers)
directory: upload something trivial, then let Benchopt package a model you
trained, then reproduce a published baseline. The other tracks ship their
`CompetSolver` references in the same place —
[Track 01](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/image_decoding/solvers),
[Track 02](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/bci_decoding/solvers),
[Track 04](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/emg_pose/solvers)
— from simple baselines (`MeanLogReg`, `Median`, `MeanEmbedding`,
`MeanPose`) to EEGNet variants.

### Practice 1: Test Codabench with a dummy submission

Build the ZIP from Track 03's `01_dummy_submission/` folder and upload
it through **My Submissions**:

```bash
python tools/make_examples.py --track sleep_onset
```

Its predictions are deliberately meaningless. A successful run confirms
ZIP ingestion, weight loading, inference, scoring and leaderboard
publication before you package your own model.

### Practice 2: Train and package with Benchopt

Benchopt already evaluates your submission; it can package it too.
Rather than assembling the ZIP by hand as in step 3 above, implement
two optional methods and a local run writes it for you — see
`03_train_and_package_with_benchopt/`. During that run:

- `fit(model, train_loader)` trains the model
- `save_model(model, path)` saves its weights
- `CompetSolver` packages the solver and weights into an upload-ready ZIP

Codabench never calls `fit` or `save_model` — the server is
inference-only — so they cost you nothing at evaluation time. If you
train and package another way, omit both.

Use these track and objective names:

| Track | `<track>`        | `<objective>`    |
| ----- | ---------------- | ---------------- |
| 01    | `image_decoding` | `Image-decoding` |
| 02    | `bci_decoding`   | `BCI-decoding`   |
| 03    | `sleep_onset`    | `Sleep-onset`    |
| 04    | `emg_pose`       | `EMG-pose`       |

```bash
benchopt prepare tracks/<track>   # prepare the data
benchopt run tracks/<track> -s MyModel -o "<objective>[training=True]"
```

The training run creates:

```text
tracks/<track>/outputs/submission_<model-name>.zip
```

### Optional practice 3: Reproduce the NeuralBench start kit

Track 03's `02_eegnet_startkit_submission/` folder holds an
inference-only solver and its trained weights, produced from the
NeuralBench start kit. `python tools/make_examples.py --track
sleep_onset` builds its ZIP. Upload it, then read how the architecture
and preprocessing are exposed through `submission.py`.

To reproduce the baseline yourself, follow the corresponding NeuralBench
guide. Each guide provides the task, public data pipeline, preprocessing, and
reference model:

| Track             | NeuralBench preparation guide                                                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 01 - EEG-to-Image | [Open the Track 01 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html) |
| 02 - BCI Decoding | [Open the Track 02 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html)   |
| 03 - Sleep Onset  | [Open the Track 03 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html)  |
| 04 - EMG-to-Pose  | [Open the Track 04 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html)  |

NeuralBench trains the model. Codabench can evaluate the result only after its
architecture and inference preprocessing are exposed through the `Solver`
contract and its trained weights are packaged in the ZIP.

1. Follow the track guide and reproduce the baseline on permitted data.
2. Package its `submission.py` and trained weights.
3. Upload the ZIP through **My Submissions** and confirm that it finishes and
   receives a score.

## Test your submission locally

Local testing is optional, but catches missing files, imports, and incorrect
output shapes.

Download the **starting kit** from the competition's *Files* tab — it holds
the benchmark, this track's baselines and the example submissions — or clone
the competition repository. Both unpack to the same `tracks/<track>/` layout,
so from either root:

```bash
benchopt install tracks/<track>  # add --gpu if your setup requires CUDA
cp my_submission/* tracks/<track>/solvers/
benchopt run tracks/<track> -d Simulated -s MyModel
benchopt test tracks/<track> --skip-install
```

Replace `MyModel` with `Solver.name`. `Simulated` requires no download.
`benchopt test` also exercises small configurations with different dimensions
where applicable. Neither command produces an official score. Dropping your
files next to the track's own baselines is also the easiest way to compare
against them: selectors are case-insensitive globs, so
`-s MyModel -s "eegnet*"` runs yours against every EEGNet baseline the
track ships.

---

## Common submission errors

- `submission.py` or the weights are inside an extra directory in the ZIP
- `class Solver` is missing, renamed, or cannot be imported
- the filename loaded in `load_model` does not match the uploaded weight file
- the model imports a package that is not available in the worker image
- the model or input is placed on the wrong device
- `predict(X)` returns the wrong shape, type, or unit for the track
- the submission tries to train, download data, or write into its read-only
  directory during server evaluation

If ingestion fails, start with the **first error in the log**. Later messages
such as a missing `results.parquet` usually mean that inference already
failed and no results file could be created.

---

## Optional: use a large pretrained EEG model

`braindecode` ships pretrained models including BENDR, BIOT, CBraMod, and
SignalJEPA. When a checkpoint is too large for the submission ZIP,
`Model.from_pretrained(...)` can retrieve its published weights from the
Hugging Face Hub. Prefer shipping weights when feasible because the download
runs for every submission and counts against the one-hour evaluation limit.

---

## Develop further with Benchopt

Starting kits are standard [Benchopt](https://benchopt.github.io)
benchmarks. Benchopt supports parameter sweeps, cached reruns, interactive
reports with `benchopt plot`, reproducible YAML configurations, and local or
SLURM execution. These features are optional. See the
[starting-kit README](https://github.com/neural-interfaces26/2026-competition#develop--train-your-model-with-benchopt)
for the full workflow. For AI tools, `benchopt sync-skills --global` installs
Benchopt solver conventions.
