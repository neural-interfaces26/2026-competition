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
| 3 | `tracks/sleep_onset` | regress seconds to the first stable N2 epoch | binned MAE (s) |
| 4 | `tracks/emg_pose` | regress hand-joint angles from wrist EMG | angular MAE (°) |

Each track runs on its own (`benchopt run tracks/<name>`), so participants
test locally with the *same* code the competition runs. A submission is a
**trained model** — plain PyTorch, shipped as code + weights — evaluated
inference-only on the platform. See
[`codabench/pages/participate.md`](codabench/pages/participate.md) for the
submission how-to, and [`design.md`](design.md) for the architecture and its
rationale.

## Structure

```
benchmark_utils/            shared components (data loading, submission contract,
                        baselines, metrics)
tracks/
  image_decoding/       track 1 benchmark
  bci_decoding/         track 2 benchmark
  sleep_onset/          track 3 benchmark
  emg_pose/             track 4 benchmark (simulated data only for now)
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

trains and scores it exactly like the platform will. Along the way, benchopt
gives you the things you end up wanting when iterating on a model:

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
  `fit` and every training run ends with a ready-to-upload
  `outputs/submission_<name>.zip` (your solver as `submission.py` + the
  weights it saved). `solution/bci_decoding/submission.py` shows the full
  load/fit/save triple.

The platform evaluation (`codabench/ingestion_program/ingestion.py` +
`scoring_program/scoring.py`) is a thin wrapper around that same
`benchopt run` — inference-only, driven by the phase's `config.yaml`
(see [`design.md`](design.md)).

## Run in Docker

One recipe ([`tools/Dockerfile`](tools/Dockerfile)) builds a single image
for the four tracks and both phases — the same image serves participants and
the Codabench workers. It carries the environment only: the benchmark and the
phase config travel together in the phase bundle mounted on
`/app/input_data`, and the ingestion/scoring programs come from this
checkout. Data lives *outside* the image too, read from
`$BENCHOPT_DATA_HOME` (`/app/data`), so bind-mount any host folder there.

```bash
tools/build_image.sh [--push]    # tommoral/neural-compet:v1
IMG=tommoral/neural-compet:v1
PHASE=codabench/phases/warmup/sleep_onset

# one-time download of a track's public dataset into a host folder
docker run -v ~/neural-data:/app/data -v $PWD/$PHASE:/app/input_data $IMG \
    benchopt prepare /app/input_data/benchmark -d Sleep-EDF

# run your submission (code + weights) against the phase's benchmark
docker run --gpus all -v ~/neural-data:/app/data -v $PWD/$PHASE:/app/input_data \
    -v $PWD/my_submission:/sub $IMG \
    bash -c 'cp /sub/* /app/input_data/benchmark/solvers/ &&
             benchopt run /app/input_data/benchmark -d Sleep-EDF -s my-solver'
```

The platform evaluation is that same run, inference-only, driven by
`ingestion.py`: mount the programs as `/compet`, your submission as
`/app/ingested_program` and a results folder as `/app/output` to reproduce it
to the letter.

```bash
docker run --gpus all -v ~/neural-data:/app/data -v $PWD/$PHASE:/app/input_data \
    -v $PWD/codabench:/compet -v $PWD/my_submission:/app/ingested_program \
    -v $PWD/results:/app/output $IMG \
    python3 /compet/ingestion_program/ingestion.py
```

## Build & CI

- `python tools/create_bundle.py --all` produces one `bundle_<track>.zip` per
  track, ready to upload to Codabench.
- `tools/build_image.sh --push` builds and pushes the Docker image.
- CI runs `benchopt test` on the 4 tracks plus lint, and an end-to-end
  Docker test of the ingestion/scoring programs.
