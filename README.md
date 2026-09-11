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
compet_core/            shared components (data loading, submission contract,
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
# zero-download smoke test, any track
benchopt run tracks/bci_decoding -d Simulated

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

The platform evaluation (`codabench/ingestion_program/ingestion.py` +
`scoring_program/scoring.py`) is a thin wrapper around that same
`benchopt run` — inference-only, driven by the phase's `config.yaml`
(see [`design.md`](design.md)).

## Run in Docker

One recipe ([`tools/Dockerfile`](tools/Dockerfile)) builds one image per
track — the same image serves participants and the Codabench workers. The
track's benchmark is baked in at `$COMPET_BENCHMARK_DIR`
(`/compet/benchmark`); data always lives *outside* the image, read from
`$BENCHOPT_DATA_HOME` (`/data`), so bind-mount any host folder there.

```bash
tools/build_images.sh [--push]   # tommoral/neural-compet-<track>:v1, all tracks
IMG=tommoral/neural-compet-sleep_onset:v1

# one-time download of a track's public dataset into a host folder
docker run -v ~/neural-data:/data $IMG \
    benchopt prepare /compet/benchmark -d Sleep-EDF

# run your submission (code + weights) against the embedded benchmark
docker run --gpus all -v ~/neural-data:/data -v $PWD/my_submission:/sub $IMG \
    bash -c 'cp /sub/* /compet/benchmark/solvers/ &&
             benchopt run /compet/benchmark -d Sleep-EDF -s my-solver'
```

The platform evaluation is that same run, inference-only, driven by
`/compet/ingestion_program/ingestion.py` and the phase config baked in at
`/app/input_data` (dev phase by default; Codabench mounts the live phase's
over it) — mount your submission as `/app/ingested_program` and a results
folder as `/app/output` to reproduce it to the letter.

## Codabench worker setup (organizers)

The self-hosted compute queue evaluates submissions inside the track image.
On the worker server, per track:

```bash
docker pull tommoral/neural-compet-sleep_onset:v1

# stage the phase data once, with the same image participants use
mkdir -p /srv/neural-data
docker run -v /srv/neural-data:/data tommoral/neural-compet-sleep_onset:v1 \
    benchopt prepare /compet/benchmark -d Sleep-EDF
```

Then configure the compute worker so submission containers run with
`-v /srv/neural-data:/data` (and the nvidia runtime for GPU tracks), and set
the competition's docker image to the track image. Each phase's `input_data`
dataset on Codabench provides the `config.yaml` (dataset selection, seed,
scoring columns — see [`design.md`](design.md)); sealed final-phase splits
ship there as `datasets/*.py`, never in this repo.

## Build & CI

- `python tools/create_bundle.py --all` produces one `bundle_<track>.zip` per
  track, ready to upload to Codabench.
- `tools/build_images.sh --push` builds and pushes the 4 track Docker images.
- CI runs `benchopt test` on the 4 tracks plus lint, and an end-to-end
  Docker test of the ingestion/scoring programs.

## Setting up a compute worker

On a GPU VM with `/data` mounted, copy `tools/setup_worker.sh` and an env
file with the queue's `BROKER_URL` (and `COMPET_PHASE` when several phases
exist), then run it as root. The script stages the tracks' datasets in
`/data` before starting the worker (all 4 tracks by default; pass track
names as extra arguments to restrict, e.g. for testing). The full staging
downloads for hours, so detach it:

```bash
sudo nohup ./setup_worker.sh /etc/codabench-worker.env \
    > setup_worker.log 2>&1 &
tail -f setup_worker.log
```

Re-running the script is cheap (already-staged data is only re-validated) —
do so after enabling a dataset in a phase config and rebuilding the images.
