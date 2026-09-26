# Track benchmarks and model workflows

Each directory here is a standalone Benchopt benchmark and the executable
definition used by the corresponding Codabench competition. Its README gives
the track-specific tensor contract, public data choices, reference results,
and available worked models:

- [Track 01: EEG-to-Image](image_decoding/README.md)
- [Track 02: BCI Decoding](bci_decoding/README.md)
- [Track 03: Sleep Onset](sleep_onset/README.md)
- [Track 04: EMG-to-Pose](emg_pose/README.md)

## What every track contains

| Path | Responsibility |
|---|---|
| `objective.py` | prediction contract and metric executed by Benchopt and Codabench |
| `datasets/` | public or simulated task data exposed through the NeuralBench data layer |
| `solvers/` | editable worked models, from constant floors to neural baselines |
| `submission_templates/` | inference wrappers for checkpoints trained by the NeuralBench start kits |
| `starter.yml` | zero-download contract check on simulated data |
| `training.yml` | default public-data training and packaging workflow |
| `benchmark_utils` | link to the shared contract and utilities at the repository root |

There are two complementary worked routes. NeuralBench provides the
neurophysiology task and model environment. Benchopt runs reproducible
experiments directly against the competition benchmark and can export a
submission. You may also train entirely in your own pipeline. Every route
ends with the same Codabench ZIP contract.

## Develop and package with Benchopt

Use this route to run an included solver or your own model against the public
competition benchmark. Complete the optional
[local setup](../README.md#optional-local-benchopt-setup), then replace
`<track>` with one of the directory names above.

Start with the simulated contract check:

```bash
benchopt install tracks/<track> --config tracks/<track>/starter.yml -y
benchopt run tracks/<track> --config tracks/<track>/starter.yml
```

To exercise training, checkpoint export, inference reload, and ZIP packaging
without downloading public data, run:

```bash
python tools/debug_submission.py --track <track>
```

This uses a tiny synthetic dataset with the real warm-up tensor dimensions.
The resulting ZIP is structurally uploadable to Codabench, but its score has
no scientific meaning.

Then install the real-data configuration, prepare its public dataset, and
train the default linear baseline:

```bash
benchopt install tracks/<track> --config tracks/<track>/training.yml -y
benchopt prepare tracks/<track> --config tracks/<track>/training.yml
benchopt run tracks/<track> --config tracks/<track>/training.yml
```

The training run exports `submission.py` and its weights to
`tracks/<track>/outputs/Torch-Linear/`. Zip the contents of that directory,
not the directory itself. To develop your own model, copy a solver, give it a
unique `Solver.name`, and override the configured solver with `-s MyModel`.
The selected track README documents alternative datasets and output shapes.

Each worked file under `solvers/` is both an executable Benchopt solver and a
valid source for `submission.py`. Benchopt also supports parameter grids,
cached reruns, YAML experiment configs, HTML reports and `benchopt plot`,
local or SLURM parallelism, `benchopt info`, and `benchopt test`. The optional
`benchopt sync-skills --global` command installs its solver conventions for
compatible coding assistants. See the
[Benchopt documentation](https://benchopt.github.io/) for these development
features.

## Package a NeuralBench checkpoint

Use this route after training the reference architecture through a
[NeuralBench competition start kit](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/index.html).
NeuralBench trains the model; the wrapper in this repository reconstructs it
for Codabench inference. Benchopt is not required for training or packaging
this route. These wrappers are inference-only and intentionally omit
`fit` and `save_model`.

Retain or export the best checkpoint before NeuralBench cleans up the run.
Set `delete_checkpoints_on_exit=False` in the run configuration if you want
NeuralBench to keep its temporary best checkpoint.
The wrappers accept a Lightning `best.ckpt`, a NeuralBench-exported model
state dictionary, or a plain model state dictionary.

| Track | Start-kit model | Copy unchanged as `submission.py` |
|---|---|---|
| 01 | EEGNet | `image_decoding/submission_templates/neuralbench_eegnet_submission.py` |
| 02 | EEGNet | `bci_decoding/submission_templates/neuralbench_eegnet_submission.py` |
| 03 | EEGNet | `sleep_onset/submission_templates/neuralbench_eegnet_submission.py` |
| 04 | VEMG2Pose | `emg_pose/submission_templates/neuralbench_vemg2pose_submission.py` |

From the repository root:

```bash
mkdir my_submission
cp tracks/<track>/submission_templates/<template> my_submission/submission.py
cp /path/to/your/checkpoint my_submission/weights.pt
zip -j my_submission.zip my_submission/submission.py my_submission/weights.pt
```

Open the ZIP before uploading. Both files must be at its root. The wrappers
apply the track-specific output contract: image embeddings for Track 01,
class indices for Track 02, seconds for Track 03, and radian pose predictions
with the required temporal context for Track 04.

These wrappers match the unmodified start-kit architectures. If you change an
architecture or its preprocessing, adapt the copied `submission.py` so that
inference reproduces those choices. Foundation models, probes, LoRA adapters,
and other downstream variants require their corresponding inference
architecture rather than these start-kit wrappers.

### Optional local replay

Benchopt can replay the packaged checkpoint against the competition contract
and matching public data without retraining or repackaging it:

```bash
benchopt prepare tracks/<track> -d "<real data source>"
COMPET_SUBMISSION_DIR="$PWD/my_submission" \
  benchopt run tracks/<track> -d "<real data source>" \
  -s "$PWD/my_submission/submission.py"
```

Use the real data source matching the checkpoint. Fixed-size trained models
usually do not match the smaller simulated dimensions.

## Use your own pipeline

Train with any permitted data and codebase, then expose the trained model
through the same `submission.py` contract. The official
[Submission Guide](../codabench/pages/participate.md) defines the required
class, metadata, output shapes, ZIP layout, and optional local validation.
