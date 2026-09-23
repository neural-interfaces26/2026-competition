# Neural Interfaces 2026 — competition benchmarks & Codabench bundles

The code behind the [NeurIPS 2026 neural-interfaces
competition](https://neural-interfaces26.github.io): **4 tracks**, each a
standalone [benchopt](https://benchopt.github.io) benchmark and its own
[Codabench](https://www.codabench.org) competition, sharing the data layer
(built on
[neuralset / neuralbench](https://facebookresearch.github.io/neuroai/)) and
the submission contract.

| Track | Benchmark | Task | Metric |
|---|---|---|---|
| 1 | `tracks/image_decoding` | decode the viewed image from an EEG epoch (retrieval) | top-5 accuracy |
| 2 | `tracks/bci_decoding` | cued mental-command classification | balanced accuracy |
| 3 | `tracks/sleep_onset` | regress seconds to the first N2 epoch | weighted binned MAE (W-bMAE, s) |
| 4 | `tracks/emg_pose` | regress hand-joint angles from wrist EMG | angular MAE (°) |

Each track runs on its own (`benchopt run tracks/<name>`), so participants
test locally with the *same* code the competition runs. A submission is a
**trained model** — plain PyTorch, shipped as code + weights — evaluated
inference-only on the platform. See
[`codabench/pages/participate.md`](codabench/pages/participate.md) for the
submission how-to.

## Structure

```
benchmark_utils/        shared components (data loading, submission contract,
                        baselines, metrics)
tracks/
  image_decoding/       track 1 benchmark
  bci_decoding/         track 2 benchmark
  sleep_onset/          track 3 benchmark
  emg_pose/             track 4 benchmark (EMG2Pose + simulated smoke data)
codabench/              shared ingestion/scoring programs, one competition
                        config per track, competition pages
solution/<track>/       sample submission per track
tools/                  bundle builder, data preparation, Docker helpers
```

## Run locally

```bash
# zero-download smoke test, any track. Runs are inference-only by default
# (like the platform); the objective's training variant trains the
# solvers' fit first — this is how the baselines are trained.
benchopt run tracks/bci_decoding -d Simulated -o "BCI-decoding[training=True]"

# benchopt test on the tiny configs
benchopt test tracks/bci_decoding --skip-install

# install a track's requirements (add --gpu for a CUDA setup)
benchopt install tracks/bci_decoding

# real data (one-time download; large for some tracks). Downloads land in
# benchopt's data folder — tracks/<t>/data by default, or $BENCHOPT_DATA_HOME.
benchopt prepare tracks/bci_decoding -d "BCI[study=tangermann2012]"

# test a submission: drop its files (code + weights) into the track's
# solvers/ and run it like any benchopt solver
cp solution/bci_decoding/* tracks/bci_decoding/solvers/
benchopt run tracks/bci_decoding -d Simulated -s Sample-BCI
```

## Develop & train your model with benchopt

The tracks are plain [benchopt](https://benchopt.github.io) benchmarks, so
your whole development loop lives in one tool. Start from a baseline: copy
one from `tracks/<t>/solvers/`, rename it, edit `load_model` / `fit`, and

```bash
benchopt run tracks/bci_decoding -d Simulated -s my-solver \
    -o "BCI-decoding[training=True]"
```

trains and scores it exactly like the platform will. `-d Simulated` needs no
download but is only a contract check — swap it for a real study (e.g.
`-d "BCI[study=dreyer2023]"`) to train on competition data;
[`participate.md`](codabench/pages/participate.md) lists each track's studies
and which one the warm-up scores. Along the way, benchopt gives you the things
you end up wanting when iterating on a model:

- **Hyperparameter grids, one flag** — declare `parameters` on your solver
  and sweep them inline: `-s "my-solver[lr=[1e-4,1e-3],n_epochs=[20,50]]"`
  runs every combination and collects them in one results file.
- **Caching** — completed runs are cached; rerunning after adding a variant
  only computes what is new (`--no-cache` to force).
- **Interactive reports** — every run writes an HTML dashboard next to the
  parquet results; `benchopt plot` (or `--all` to merge runs) compares your
  variants and the baselines visually.
- **Reproducible experiment files** — pin datasets, solvers and seed in a
  yaml and `benchopt run --config my_config.yml`; the competition phases are
  driven by exactly such files.
- **Parallel & cluster runs** — `-j 4` fans out locally,
  `--parallel-config slurm.yml` sends the grid to a SLURM cluster.
- **AI-assistant ready** — `benchopt sync-skills --global` installs
  benchopt's agent skill (Claude Code, Copilot, ...), so your coding
  assistant knows the solver/dataset conventions when it writes one for you.
- **Introspection** — `benchopt info tracks/<t>` lists solvers/datasets and
  their parameters; `benchopt test tracks/<t> --skip-install` sanity-checks
  a new solver against the tiny test configs.
- **Submission artifact** — implement `save_model(model, path)` next to your
  `fit` and every training run writes a ready-to-zip submission folder at
  `tracks/<track>/outputs/<name>/` (your solver as `submission.py` + the
  weights it saved). `solution/bci_decoding/submission.py` shows the full
  load/fit/save triple.

The platform evaluation (`codabench/ingestion_program/ingestion.py` +
`scoring_program/scoring.py`) is a thin wrapper around that same
`benchopt run` — inference-only, driven by the phase's `config.yaml`.

## Run in Docker

One recipe ([`tools/Dockerfile`](tools/Dockerfile)) builds a single image for
the four tracks and both phases — the same image serves participants and the
Codabench workers. It carries the environment only: the benchmark and the
phase config travel together in the phase bundle mounted on
`/app/input_data`, and data lives outside the image, read from
`$BENCHOPT_DATA_HOME` (`/app/data`).

`tools/run_docker.py` is the platform evaluation, to the letter: it builds
the image, assembles the phase bundle and runs the ingestion and scoring
programs on a submission.

```bash
python tools/run_docker.py --track sleep_onset              # the sample
python tools/run_docker.py --track sleep_onset \
    --submission my_submission.zip --data ~/neural-data
```

The first run on a real dataset is slow — it downloads and prepares the data
into `--data` — and every later run reuses it.

## Build & CI

- `python tools/create_bundle.py --all` produces one `bundle_<track>.zip` per
  track, ready to upload to Codabench.
- `tools/build_image.sh --push` builds and pushes the Docker image.
- CI runs `benchopt test` on the 4 tracks plus lint, and an end-to-end
  Docker test of the ingestion/scoring programs.
