# EEG Models Competition 2026 — Codabench bundle

A [Codabench](https://www.codabench.org) competition to evaluate **EEG
foundation models** (via linear probing) against **task-specific specialist
models**, powered by [benchopt](https://benchopt.github.io). It ships two tasks
spanning the two data regimes EEG decoding needs:

- **Motor imagery** (MOABB BNCI2014_001) — *epoched*: one label per trial.
- **Sleep staging / onset detection** (Sleep-EDF) — *continuous / dense*: a
  per-time-step label sequence over a long window.

## Design choices

- **The benchmark is a standalone benchopt benchmark** (`benchmark/`), runnable
  on its own (`benchopt run benchmark/`) and developed independently of the
  Codabench bundle. Participants test locally with the *same* code the
  competition runs — only data paths / settings change, via config.
- **Two tracks via an Objective switch** `track ∈ {linear_probe, general}`:
  - *linear_probe* (foundation models): a submission ships only a frozen
    encoder; the infra auto-fits a scikit-learn linear head. This is the
    track's constraint — labels never reach the encoder.
  - *general* (specialists): a submission trains a task-specific model directly.
- **A submission is a benchopt solver.** Foundation-model submissions subclass
  `benchmark_utils.base_solver.CompetEEGSolver` and implement `load_model` +
  `time_embed`/`embed` (see `solution/submission.py`). The base class handles
  track gating and fitting the linear probe.
- **One contract for both regimes.** The encoder produces a *temporal*
  embedding `(B, C, T) → (B, T', D)`; the linear head is applied per time
  position. Epoched tasks pool over `T'` (one label/window); dense tasks keep
  the sequence (`(B, T', K+1)`) and upsample predictions to the raw time axis.
- **Torch end-to-end.** Dataloaders yield torch tensors so braindecode/REVE-style
  models run directly; thanks to scikit-learn's **array API** the torch features
  flow straight through the linear head (on-device, no numpy round-trip), with a
  numpy fallback. (`SCIPY_ARRAY_API=1` must be set before scipy import — the
  ingestion and probe modules set it.)
- **neuralset is confined to data loading.** neuralset / neuralfetch (Study →
  events → Segmenter → SegmentDataset) live only behind
  `benchmark_utils/neuralset_task.py` and the `datasets/`; everything is
  converted to torch tensors at that boundary, so nothing downstream is tied to
  the data stack.
- **Everything runs in ingestion; scoring only parses.** `ingestion_program`
  runs the benchmark on the submission via `benchopt.run_benchmark` (programmatic
  API, no subprocess) and stores the **raw** benchopt results dataframe with
  `save_results`. `scoring_program` reads it back with `read_results` and emits
  `scores.json`. Keeping the raw dataframe lets us store predictions later.

## Structure

```
benchmark/              standalone benchopt benchmark
  objective.py          EEGObjective: `track` switch + epoched/dense metrics
  datasets/             simulated (zero-dep), moabb_mi, sleep_edf
  solvers/              linear_probe (reference FM), general (reference specialist)
  benchmark_utils/      base_solver, linear_probe, data, baselines, neuralset_task
ingestion_program/      runs the benchmark on a submission -> raw results dataframe
scoring_program/        parses the dataframe -> scores.json
solution/submission.py  sample foundation-model submission (REVE)
tools/                  setup_data (benchopt prepare), create_bundle, Dockerfile
dev_phase/              placeholder data dirs (data is loaded by the benchmark)
pages/                  Codabench competition pages
```

## Run locally

```bash
# zero-download smoke test (both tracks, both regimes)
benchopt run benchmark/ -d Simulated

# benchopt test on the tiny config
benchopt test benchmark/ -k Simulated --skip-install

# end-to-end ingestion + scoring on Simulated (mirrors Codabench)
python ingestion_program/ingestion.py --submission-dir solution/ \
    --output-dir ingestion_res --datasets Simulated
python scoring_program/scoring.py --prediction-dir ingestion_res \
    --output-dir scoring_res

# real data (downloads MOABB + Sleep-EDF once)
python tools/setup_data.py
```

See `pages/participate.md` for writing and testing a submission.

## Build & CI

- `python tools/create_bundle.py` produces `bundle.zip` (includes `benchmark/`)
  to upload to Codabench.
- `.github/workflows/benchmark.yml` runs `benchopt test` on `benchmark/` (via
  the reusable `benchopt/template_benchmark` workflows, with
  `benchmark_dir: ./benchmark`) plus lint. `.github/workflows/test.yml` builds
  the Docker image and runs the ingestion/scoring end-to-end.
