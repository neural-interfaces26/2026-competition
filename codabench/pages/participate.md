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

**Pretrained EEG foundation models.** `braindecode` ships several (BENDR,
BIOT, CBraMod, SignalJEPA, ...) and its `Model.from_pretrained(...)` pulls
the published weights from the HuggingFace Hub, which is the way to use a
checkpoint too large to ship in your ZIP. Prefer shipping the weights when
you can: a Hub download runs on every submission and counts against the
one-hour evaluation limit.

---

## Submit in four steps

1. **Train and validate locally** (with the benchopt starting kit or your
   own pipeline). Save the trained weights.
2. **Create `submission.py`** following the contract below.
3. **Create the ZIP.** For the example above, run:

   ```bash
   zip -j my_submission.zip submission.py weights.pt
   ```

   Add `config.json` or any other non-Python artifact your model needs.

4. **Upload and verify.** In **My Submissions**, select the active phase,
   upload the ZIP, and wait for **Finished**. If it fails, start with the first
   error in the ingestion log.

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
    # See "Optional: train through Benchopt" below.
    def fit(self, model, train_loader):
        ...

    def save_model(self, model, path):
        ...
```

Every track ships working implementations of this contract in its
`solvers/` directory — real `CompetSolver` subclasses on real data, from
trivial baselines (`MeanLogReg`, `Median`, `MeanEmbedding`, `MeanPose`) to
EEGNet variants:
[image_decoding](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/image_decoding/solvers),
[bci_decoding](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/bci_decoding/solvers),
[sleep_onset](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/sleep_onset/solvers),
[emg_pose](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/emg_pose/solvers).
They are the closest thing to a reference answer for your own
`submission.py`.

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

## Get some practice 1: Validate a complete Codabench submission

**Goal: upload a minimal working model before adapting the same structure to
your trained model.**

The repository's [worked examples](https://github.com/neural-interfaces26/2026-competition/tree/main/examples)
provide a small code-and-weights submission for each track. Choose the folder
for your track and follow its README to create the ZIP.

Upload the example only to its matching competition. A successful run
validates the ZIP structure, weight loading, inference, scoring, and
leaderboard publication. These examples are technical checks, not reference
baselines.

---

## Get some practice 2: Reproduce and submit a NeuralBench baseline

**Goal: apply the workflow from Practice 1 to a trained reference model.**
Use your track's NeuralBench starting kit to reproduce its public baseline,
then submit that model to Codabench during the warm-up phase.

The guides provide the track tasks, public data pipelines, preprocessing, and
reference baselines:

| Track             | NeuralBench preparation guide                                                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 01 - EEG-to-Image | [Open the Track 01 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track1_eeg_to_image.html) |
| 02 - BCI Decoding | [Open the Track 02 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html)   |
| 03 - Sleep Onset  | [Open the Track 03 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html)  |
| 04 - EMG-to-Pose  | [Open the Track 04 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html)  |

NeuralBench produces the trained model. Codabench can evaluate it only when
its architecture and inference preprocessing are exposed through the
`Solver` contract above and its weights are included in the ZIP.

1. Follow the track guide and reproduce the baseline on permitted data.
2. Package its `submission.py` and trained weights.
3. Upload the ZIP through **My Submissions** and confirm that it finishes and
   receives a score.

If the NeuralBench solver already implements `fit` and `save_model`, the
training run performs step 2 and creates:

```text
tracks/<track>/outputs/submission_<model-name>.zip
```

With another training pipeline, save the parameters yourself and load them
from `meta["submission_dir"]` in `load_model`.

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

## Optional: test your submission locally

Local testing is optional, but catches missing files, imports, and incorrect
output shapes.

From a checkout of the competition repository:

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

### Optional: train through Benchopt

During local training, optional `fit(model, train_loader)` trains the model and
`save_model(model, path)` writes its artifacts. `CompetSolver` then packages
them with `submission.py` into an upload-ready ZIP.

Use these track and objective names:

| Track | `<track>`        | `<objective>`    |
| ----- | ---------------- | ---------------- |
| 01    | `image_decoding` | `Image-decoding` |
| 02    | `bci_decoding`   | `BCI-decoding`   |
| 03    | `sleep_onset`    | `Sleep-onset`    |
| 04    | `emg_pose`       | `EMG-pose`       |

To train and export through Benchopt, run:

```bash
benchopt prepare tracks/<track>   # prepare the data
benchopt run tracks/<track> -s MyModel -o "<objective>[training=True]"
```

If you train outside NeuralBench, omit these optional methods.

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

## Optional: develop further with Benchopt

Starting kits are standard [Benchopt](https://benchopt.github.io)
benchmarks. Benchopt supports parameter sweeps, cached reruns, interactive
reports with `benchopt plot`, reproducible YAML configurations, and local or
SLURM execution. These features are optional. See the
[starting-kit README](https://github.com/neural-interfaces26/2026-competition#develop--train-your-model-with-benchopt)
for the full workflow. For AI tools, `benchopt sync-skills --global` installs
Benchopt solver conventions.
