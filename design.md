# Context — EEG Models Competition 2026 (for future sessions)

Orientation doc for anyone (human or agent) resuming work on this repo. Pairs
with `README.md` (user-facing design) — this file adds *why*, current state,
and the rough edges.

## What this is

A Codabench competition to evaluate **EEG foundation models** (frozen encoder +
linear probing) against **task-specific specialist models**, using **benchopt**
as the evaluation engine. The full competition targets **4 tasks**; this repo is
a **proof-of-concept with 2**, chosen to exercise both data regimes:

1. **Motor imagery** (MOABB BNCI2014_001) — *epoched* (one label per trial).
2. **Sleep staging / onset detection** (Sleep-EDF) — *continuous / dense*
   (per-time-step label sequence).

Eventual tasks also include **THINGS-EEG image decoding** and **EMG2qwerty
typing** — the epoched/dense contract was designed to absorb them.

## Architecture (two layers)

- `benchmark/` — a **standalone benchopt benchmark** (runnable on its own,
  `benchopt run benchmark/`). This is where all the science lives.
- The **Codabench bundle** around it: `ingestion_program/` runs the benchmark on
  a submission via `benchopt.run_benchmark` (programmatic, no subprocess) and
  stores the raw results dataframe; `scoring_program/` parses it to
  `scores.json`; `tools/` builds the bundle; `pages/` are the competition pages.

## Key design choices (and why)

- **Two tracks via an `Objective` parameter** `track ∈ {linear_probe, general}`
  (the user wanted the switch at the objective level). Solvers `skip()` the
  track they don't belong to.
- **A submission is a benchopt solver.** FM submissions subclass
  `benchmark_utils/base_solver.py::CompetEEGSolver` and implement
  `load_model` + `time_embed` (+ optional `embed`); the base fits the linear
  probe. **Specialist submissions are task-specific:** they subclass
  `CompetEEGGeneralSolver` (same file), set a `task` class attribute, and
  implement `load_model` (returns a `fit(loader)`/`predict(X)` model); the base
  gates on `track == general` *and* `task == self.task`. Reference solvers live
  in `solvers/`: `LinearProbe` (random-projection FM) and one specialist per
  task `general_{mi,sleep}` (sklearn baseline). Sample submissions:
  `solution/submission.py` (FM, name `REVE`) and
  `solution/submission_general.py` (specialist, name `MI-LogVar`, `task="mi"`).
- **One contract for both regimes:** encoder returns a *temporal* embedding
  `(B, C, T) → (B, T', D)`. Epoched → mean-pool over `T'` → one label/window;
  dense → per-position head `(B, T', K+1)`, predictions upsampled to raw `T`.
  Dense targets use class `K` (last index) as background/null.
- **Torch end-to-end + scikit-learn array API.** Dataloaders yield torch
  tensors so braindecode models run directly; features pass through the sklearn
  linear head on-device via `array_api_dispatch` (needs `SCIPY_ARRAY_API=1`
  before scipy import — set in `ingestion.py` and `benchmark_utils/linear_probe.py`
  via `os.environ.setdefault`), with a **numpy fallback** if unavailable.
- **neuralset confined to data loading.** neuralset/neuralfetch usage lives only
  in `benchmark_utils/neuralset_task.py` + the `datasets/`, converted to torch at
  the boundary. Pipeline: `Study(path, query).run()` → events df →
  `Segmenter.apply` → `SegmentDataset` → `load_all`. `EegExtractor` for signal,
  `LabelEncoder` (aggregation="trigger") for epoched labels, manual event
  rasterization for dense per-step labels.
- **Everything runs in ingestion; scoring only parses.** Ingestion stores the
  **raw** benchopt dataframe with `benchopt.results.save_results` and scoring
  reloads with `read_results` — chosen so prediction artefacts can be persisted
  later (packed object columns survive the round-trip).
- **Leaderboard keys** are `{track}_{task}_{metric}` so FM and specialist tracks
  never collide.

## Current state (validated)

- `benchopt run benchmark/ -d Simulated` — both tracks × both regimes; metrics
  learnable (epoched acc 1.0; dense staging ~0.93–0.99).
- `benchopt test benchmark/ -k Simulated --skip-install` — 7 passed.
- End-to-end `ingestion → scoring` with the `REVE` submission solver on
  Simulated → correct `linear_probe_simulated_*` scores; submission file
  auto-copied into `solvers/` and cleaned up.
- `flake8` clean (max-line-length 79); `create_bundle.py` ships `benchmark/`.
- CI: `.github/workflows/benchmark.yml` (reusable benchopt test, `benchmark_dir:
  ./benchmark`) + `test.yml` (Docker ingestion/scoring).

## TODOs / questionable choices / open issues

**Not yet run against real data (highest risk):**
- `datasets/moabb_mi.py` and `datasets/sleep_edf.py` are implemented against the
  neuralfetch/neuralset API but **never executed** (no network downloads in the
  dev env). Unverified specifics: the `query` string format
  (`subject == "Tangermann2012Review/1"`), `Segmenter` trigger queries, and the
  segment-metadata access in `neuralset_task.py::_segment_meta` /
  `_dense_targets` (`seg.timeline`, `seg.start`). **Run `benchopt prepare
  benchmark/` then `benchopt run benchmark/ -d MOABB-MI -d Sleep-EDF` to validate.**
- `braindecode`/REVE **cannot import on this machine** (`No module named
  'torchaudio.functional'`), so the REVE path is untested; the sample submission
  falls back to a random-projection encoder. Fix the torch/torchaudio env, then
  test `REVE.from_pretrained("brain-bzh/reve-base")`.

**Design gaps to revisit:**
- `chs_info` is currently `None` in the datasets' meta, but **REVE needs channel
  3D coordinates** at construction. Wire it from the MNE `raw.info` (the
  neuralfetch studies set a standard_1020 montage) before real REVE runs.
- `onset_f1` is a **placeholder** (binary event-vs-background F1), not a real
  onset/transition metric. For sleep, also reconsider whether dense long windows
  vs the standard 30 s epochs is the right framing.
- **Lazy loading** is only partial: `neuralset_task` materializes via
  `load_all()` then keeps tensors in `ArrayWindows` (in-memory). Fine for the POC
  subsets; wrap the `SegmentDataset` directly for large-scale data.
- `dev_phase/input_data` & `reference_data` are **placeholders** (data is loaded
  by the benchmark, not shipped). Codabench data isolation / hidden test labels
  is **explicitly deferred**.
- `competition.yaml` dates and leaderboard columns are placeholders.
- CI installs the full benchmark env (neuralset/neuralfetch are on PyPI); the
  real-task CI path is unverified — consider `extra_args: "-k Simulated"` if the
  full install is flaky.

## Pointers

- Plan: `~/.claude/plans/rustling-weaving-lobster.md`.
- neuralset/braindecode design docs: `~/Work/collaborations/brainai/docs/internal/design/`
  (`studies.md`, `dataloader-pipeline.md`, `extractor-pipeline.md`).
- Reusable skills written for this stack:
  `~/Work/prog/bash_conf/ai-skills/{neuro-ai-stack,sklearn-array-api}/`.
- Reference benchmarks: `../bci` (MOABB via braindecode), `../benchmark_tsfm`
  (the linear-probe/encoder patterns this was adapted from).
- Previous Codabench competition: `../compet2025_codabench`.
