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

Nothing is installed during evaluation, so `submission.py` may only import
what the worker image already carries — `torch` and the rest of the stack
pinned in
[`requirements.txt`](https://github.com/neural-interfaces26/2026-competition/blob/main/requirements.txt).
During `load_model` and `predict`, do not train, do not download competition
data, and do not write into the submission directory.

---

## Submit in five steps

1. **Train and validate locally** with an optional NeuralBench start kit,
   directly through Benchopt, or with your own pipeline. Save the trained
   weights. If your Benchopt solver implements the optional `fit` and
   `save_model` methods described below, its training run writes the solver
   and weights into a ready-to-upload submission folder for you.
2. **Create `submission.py`** following the contract below.
3. **Create the ZIP.** For the example above, run:

   ```bash
   zip -j my_submission.zip submission.py weights.pt
   ```

   Add `config.json` or any other non-Python artifact your model needs.

4. **Check it locally first.** A failed upload still costs you one of the
   day's submissions, and most failures are caught in seconds:

   ```bash
   cp my_submission/submission.py tracks/<track>/solvers/my_submission.py
   COMPET_SUBMISSION_DIR="$PWD/my_submission" \
       benchopt run tracks/<track> -d Simulated -s MyModel
   ```

   See [Test your submission locally](#test-your-submission-locally) for
   what this does and does not cover.

5. **Upload.** In **My Submissions**, select the active phase, upload the
   ZIP, and wait for **Finished**. If it fails, start with the first error
   in the ingestion log.

---

## The `submission.py` contract

In `submission.py`, you write ordinary Python or PyTorch code. Benchopt
provides the evaluation wrapper, but you do not need to understand its
internals. Codabench expects two components:

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

| Track             | `predict(X)` must return                   | Output-size key           | Final sealed metric      |
| ----------------- | ------------------------------------------ | ------------------------- | ------------------------ |
| 01 - EEG-to-Image | image embeddings `(B, D)`                  | `meta["n_outputs"]` = `D` | top-5 retrieval accuracy |
| 02 - BCI Decoding | one class index per window `(B,)`          | `meta["n_classes"]`       | balanced accuracy        |
| 03 - Sleep Onset  | seconds to sleep onset `(B,)` as floats    | `meta["n_outputs"]` = `1` | weighted binned MAE      |
| 04 - EMG-to-Pose  | joint angles `(B, n_joints, T)` in degrees | `meta["n_joints"]`        | mean angular MAE         |

Warm-up proxy metrics may differ. The **Track description** tab gives the
active warm-up metric and the final sealed specification for each track.

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

Only the sealed evaluation data and labels are hidden. The submission
contract remains stable, while the evaluation data and, for proxy warm-ups,
the task or metric may differ by phase:

|                                 | Warm-up phase                    | Sealed final phase                       |
| ------------------------------- | -------------------------------- | ---------------------------------------- |
| Evaluation data                 | development or public proxy data | held-out cohort, never released          |
| Benchmark framework and worker image | public competition stack      | same competition stack                   |
| Task and ranking metric         | public proxy; see Track description | final specification; see Track description |
| `meta` keys and tensor contract | documented contract              | same contract, runtime values may differ |
| Submission ZIP format           | same contract                     | same contract                            |

Tracks 01–03 currently have phase-specific proxy details. Track 04 uses the
same prediction task and metric in both phases, on different evaluation data.

For portability, **use only `meta` and the provided batches**. Do not rely on
undocumented dataset names, paths, subject identifiers, or fixed dimensions.

---

## Get some practice

The canonical worked implementations live in each track's `solvers/`
directory: [Track 01](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/image_decoding/solvers),
[Track 02](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/bci_decoding/solvers),
[Track 03](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/sleep_onset/solvers),
and [Track 04](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/emg_pose/solvers).

These files show the submission contract, but they are at different stages:
some are weightless floors, some implement local training and export, and
neural models need a trained checkpoint before they represent a baseline.
The `examples/` directory is not stored in Git. It is created only inside a
generated starting-kit archive by `tools/make_starting_kit.py`. The generator
includes trained weights only when they already exist under the track's
`outputs/` directory; otherwise it warns that the example is untrained.

Track 03 currently provides the most complete progression:

| Baseline | Solver | What it shows |
|---|---|---|
| `Median` | `median_baseline.py` | the contract with no weights and no training |
| `Mean-Ridge` | `mean_ridge.py` | a scikit-learn model, weights as a joblib dump |
| `Torch-Linear` | `torch_linear.py` | the same model in PyTorch, with its own `fit` loop |
| `EEGNet` | `eegnet_reg.py` | the NeuralBench-compatible architecture and inference wrapper; trained weights are published separately |

### Practice 1: Check the platform with a constant baseline

Start with a weightless floor where one is provided:

- Track 01: `mean_embedding.py`
- Track 03: `median_baseline.py`
- Track 04: `mean_pose.py`

Copy the selected file to `submission.py`, place it at the root of a ZIP,
and upload it through **My Submissions**. A successful run confirms ZIP
ingestion, inference, scoring, and leaderboard publication before you package
your own model. For Track 02, train and package `eegnet.py` as described in
Practice 2. Its `mean_logreg.py` currently illustrates `fit`, but does not yet
serialize and reload trained weights, so do not upload it untrained.

### Practice 2: Train and package with Benchopt

Benchopt already evaluates your submission; it can package it too.
Rather than assembling the ZIP by hand as in step 3 above, implement
two optional methods and a local run writes it for you — `torch_linear.py`
and `mean_ridge.py` are the worked examples. During that run:

- `fit(model, train_loader)` trains the model
- `save_model(model, path)` saves its weights
- `CompetSolver` writes the solver and weights into a submission folder

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

For a solver implementing `save_model`, the training run writes a
ready-to-upload submission folder (`submission.py` plus its weights). Zip its
contents to upload:

```text
tracks/<track>/outputs/<model-name>/
```

Benchopt does more than package: parameter sweeps, cached reruns, plots and
SLURM execution are summarised in
[Develop further with Benchopt](#develop-further-with-benchopt) at the end
of this page.

### Optional practice 3: Reproduce the NeuralBench start kit

The EEGNet solver files currently provided for Tracks 01–03 show how an
architecture and its inference preprocessing are exposed through the
`Solver` contract. Track 04 still needs the equivalent wrapper for its
NeuralBench reference model. The official trained checkpoints are not
currently stored in this repository. Until they are published, an untrained
or dummy checkpoint can test the worker, but it is not a reference baseline.

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

Clone or download the
[competition repository](https://github.com/neural-interfaces26/2026-competition).
If a generated track starting kit is also provided, it uses the same
`tracks/<track>/` layout and additionally contains derived submission ZIPs
under `examples/`. From the repository or an extracted kit root:

```bash
benchopt install tracks/<track>  # add --gpu if your setup requires CUDA
cp my_submission/submission.py tracks/<track>/solvers/my_submission.py
COMPET_SUBMISSION_DIR="$PWD/my_submission" \
    benchopt run tracks/<track> -d Simulated -s MyModel
```

Replace `MyModel` with `Solver.name`. `COMPET_SUBMISSION_DIR` makes
`meta["submission_dir"]` point at the folder containing your weights and
optional artifacts. `Simulated` requires no download.
It is a quick contract check, not an official score. A fixed-size checkpoint
trained for the real task may be incompatible with the smaller simulated
dimensions; validate such a model on the public track data instead. Dropping
your solver next to the track's own baselines is also the easiest way to
compare against them: selectors are case-insensitive globs, so
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
