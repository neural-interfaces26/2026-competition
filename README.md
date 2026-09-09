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

## Design choices

- **Each track is a standalone benchopt benchmark** (`tracks/<name>/`),
  runnable on its own (`benchopt run tracks/<name>`). Participants test
  locally with the *same* code the competition runs.
- **A submission is a trained model.** It ships as a benchopt solver
  (`submission.py` subclassing `compet_core.base_solver.CompetSolver` +
  weight files); the platform evaluates **inference-only**
  (`COMPET_INFERENCE_ONLY=1` — the solver's optional `fit` is a local
  training convenience, never run on the server).
- **Pure-PyTorch submissions.** Batches reach the model as plain torch
  tensors `(X, y, info)` already on `meta["device"]`; no benchopt / neuralset
  / neuralbench types cross the solver boundary.
- **CPU and GPU.** The objectives declare cpu/gpu requirement variants
  (`requirements = {"cpu": [...], "gpu": [...]}`): `benchopt install
  tracks/<t>` sets up a CPU env (what CI uses), `benchopt install tracks/<t>
  --gpu` a CUDA one. The code is device-agnostic — batches follow
  `meta["device"]` (auto-detected, override with `COMPET_DEVICE`).
- **The data layer reuses the official neuralbench pipelines.**
  `compet_core.nb_task.load_task` instantiates the task configs shipped in
  the `neuralbench` wheel (study, split, segmenter, target extractors,
  samplers) and adapts the loaders to the competition contract. Every track
  also has a zero-download `Simulated` dataset for smoke tests.
- **Everything runs in ingestion; scoring only parses.** The shared
  `codabench/ingestion_program` runs the bundled benchmark on the submission
  via `benchopt.run_benchmark` and stores the raw results dataframe; the
  shared `codabench/scoring_program` turns its float metric columns into
  `scores.json`.

## Structure

```
compet_core/            shared package: data utils, nb_task (neuralbench
                        wrapper), base_solver (submission contract),
                        baselines, metrics, linear_probe
tracks/
  image_decoding/       objective (top-5 retrieval) + THINGS-EEG2 proxy
  bci_decoding/         objective (balanced acc) + Stieger2021/Tangermann2012/
                        Dreyer2023 studies + MOABB-MI legacy dataset
  sleep_onset/          objective (binned MAE) + Sleep-EDF proxy
  emg_pose/             objective (angular MAE) + Simulated only (the real
                        Salter2024 emg2pose loader is upstream work)
codabench/
  ingestion_program/    shared, inference-only; bundles ship one track as
                        benchmark/
  scoring_program/      shared, parses the results dataframe
  competition_*.yaml    one Codabench config per track
  pages/                competition pages
solution/<track>/       sample submission per track
tools/                  create_bundle --track, setup_data --track,
                        Dockerfile, run_docker
```

## Run locally

No install needed — each track's `benchmark_utils` locates the shared
`compet_core` package from the repo (or bundle) root. `pip install -e .` is
optional (IDE/import convenience).

```bash
# zero-download smoke test, any track
benchopt run tracks/bci_decoding -d Simulated

# benchopt test on the tiny configs
benchopt test tracks/bci_decoding --skip-install

# end-to-end ingestion + scoring on Simulated (mirrors Codabench)
python codabench/ingestion_program/ingestion.py \
    --submission-dir solution/bci_decoding --output-dir ingestion_res \
    --benchmark-dir tracks/bci_decoding --datasets Simulated
python codabench/scoring_program/scoring.py \
    --prediction-dir ingestion_res --output-dir scoring_res

# real data (one-time download; large for some tracks)
python tools/setup_data.py --track bci_decoding
```

See `codabench/pages/participate.md` for writing and testing a submission.

## Build & CI

- `python tools/create_bundle.py --all` produces one `bundle_<track>.zip` per
  track to upload to Codabench (the track's benchmark is shipped as
  `benchmark/`, next to `compet_core/` and the shared programs).
- `.github/workflows/benchmark.yml` runs `benchopt test` on the 4 tracks
  (matrix over `benchmark_dir`, via the reusable
  `benchopt/template_benchmark` workflows) plus lint;
  `test-docker.yml` builds the Docker image and runs ingestion/scoring
  end-to-end on Simulated. Each track's `test_config.py` skips the tests
  that cannot run on CI runners (real-data installs/downloads pinning CUDA
  builds); those paths are validated on the cluster instead.

Note: the benchmarks require benchopt ≥ 1.9.2 (currently the `main` branch);
the CI release-version job will be re-enabled once it is on PyPI.
