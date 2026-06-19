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

- **Two tracks via an `Objective` parameter** `track ∈ {linear_probe, specific}`
  (the user wanted the switch at the objective level). Solvers `skip()` the
  track they don't belong to.
- **A submission is a benchopt solver.** FM submissions subclass
  `benchmark_utils/base_solver.py::CompetEEGSolver` and implement
  `load_model` + `time_embed` (+ optional `embed`); the base fits the linear
  probe. **Specialist submissions are task-specific:** they subclass
  `CompetEEGSpecificSolver` (same file), set a `task` class attribute, and
  implement `load_model` (returns a `fit(loader)`/`predict(X)` model); the base
  gates on `track == specific` *and* `task == self.task`. Reference solvers live
  in `solvers/`: `LinearProbe` (random-projection FM) and one specialist per
  task `specific_{mi,sleep}` (sklearn baseline). Sample submissions:
  `solution/submission_fm.py` (FM, name `REVE`) and
  `solution/submission_specific.py` (specialist, name `MI-LogVar`, `task="mi"`).
- **One contract for both regimes:** encoder returns a *temporal* embedding
  `(B, C, T) → (B, T', D)`. Epoched → mean-pool over `T'` → one label/window;
  dense → per-position head `(B, T', K+1)`, predictions upsampled to raw `T`.
  Dense targets use class `K` (last index) as background/null.
- **Torch end-to-end + scikit-learn array API.** Dataloaders yield torch
  tensors so braindecode models run directly; features pass through the sklearn
  linear head on-device via `array_api_dispatch` (needs `SCIPY_ARRAY_API=1`
  before scipy import — set in `ingestion.py` and `benchmark_utils/linear_probe.py`
  via `os.environ.setdefault`), with a **numpy fallback** if unavailable.
- **GPU support.** `data.get_device()` auto-detects CUDA (override with
  `COMPET_EEG_DEVICE`); each dataset advertises it via `meta["device"]` *and*
  moves every batch (`X`, `y`) onto it at loading time (the loaders'
  `collate_fn`). So a model only has to place *itself* on `meta["device"]` in
  `load_model` (the FM base exposes it as `self.device`) — `X`/`y` already
  match, the array-API head then runs on-device, and `to_numpy` (`.cpu()`)
  handles the metric boundary. Device-locality fixes: `RandomProjectionEncoder`
  moves its projection to `X.device`; the `_resample`/`resample_labels` index is
  built on the sequence's device. Specialists handle the device internally —
  `solvers/specific_eegnet.py` (`EEGNet-MI`) is the reference: a real torch
  training loop, braindecode EEGNet with a tiny conv-net fallback.
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

- `benchopt run benchmark/ -d Simulated` — both tracks × both task fixtures
  (`task ∈ {mi, sleep}`, mapping to the epoched/dense regimes); metrics
  learnable (epoched acc 1.0; dense staging ~0.93–0.99).
- `benchopt run benchmark/ -d MOABB-MI` — **real data, both tracks** (epoched).
  Lazy `SegmentDataset` loading + recording-level group split confirmed working
  end-to-end; baselines sit just above chance (4-class, group split is harder
  than a random split — by design).
- `benchopt test benchmark/ -k Simulated --skip-install` — 9 passed, 1 skipped.
- End-to-end `ingestion → scoring` with the `REVE` submission solver on
  Simulated → correct `linear_probe_simulated_*` scores; submission file
  auto-copied into `solvers/` and cleaned up.
- `flake8` clean (max-line-length 79); `create_bundle.py` ships `benchmark/`.
- CI: `.github/workflows/benchmark.yml` (reusable benchopt test, `benchmark_dir:
  ./benchmark`) + `test.yml` (Docker ingestion/scoring).

## TODOs / questionable choices / open issues

**Real-data status:**
- `datasets/moabb_mi.py` (epoched) is **validated on real data** — query format,
  `Segmenter` trigger queries, and `neuralset_task` segment-metadata access
  (`seg.timeline`, `seg.start`) all confirmed.
- `datasets/sleep_edf.py` (dense) is **still unverified on real data**: it shares
  the validated lazy/group-split machinery, but `build_dense` rasterization
  (`_dense_targets`) hasn't run. Verifying it needs the **full** Sleep-EDF
  download (see below), which is large/slow — run `benchopt run benchmark/ -d
  Sleep-EDF` once the data is present.
- **neuralfetch downloads the whole study, not the queried subject.**
  `study.download()` (called by `Dataset.prepare()`) fetches *all* subjects
  (MOABB loops the full `subject_list`; Sleep-EDF uses `SUBJECTS_REC_1/2`,
  ~78 subjects / 153 timelines). The `query` only filters at `run()` time, and
  `run()`'s integrity check (`_info.num_timelines`) **requires every timeline
  present**, so a subject-scoped download is *not possible* with these study
  classes. For dev, prefer `Simulated` (zero-download); real downloads are a
  one-time, whole-dataset cost.
- `braindecode`/REVE **cannot import on this machine** (`No module named
  'torchaudio.functional'`), so the REVE path is **unrun** (not just untested).
  Per the no-fallback convention, the `REVE` sample submission and the
  `EEGNet-MI` specialist now import braindecode at module top, so benchopt
  reports them as *not installed* here rather than silently substituting an
  encoder. The wiring is in place — fix the torch/torchaudio env, then run
  `benchopt run benchmark/ -d Simulated -s REVE`. Known REVE gotchas already
  handled: `from_pretrained` needs `n_outputs` (head built in `__init__`,
  unused with `return_features=True`) and `chs_info` (channel names → 3D
  positions); features come back `(B, C, n_patches, D)` and are mean-pooled
  over channels to `(B, T', D)` in `time_embed`.

**Design gaps to revisit:**
- `chs_info` **(done).** All datasets now populate `meta["chs_info"]` as
  `[{"ch_name": ...}]` via `data.chs_info_from_names`. REVE resolves 3D
  positions from these *names* through its own bank (it does **not** need raw
  coordinates), so name fidelity is what matters: real datasets read names from
  the `MneTimedArray` header (`ds[0].data["eeg"].ch_names`); `Simulated` emits
  standard 10-20 names so REVE can smoke-test on it. Sleep-EDF channels are
  bipolar (`EEG Fpz-Cz`, `EEG Pz-Oz`), which REVE's single-electrode position
  bank can't resolve (it crashes with a cryptic `IndexError` when *no* name
  matches); the `REVE` submission handles this by mapping each channel to its
  **anode** electrode (`_reve_channel`: strip `EEG `, take the first of the
  pair → `Fpz`, `Pz`) and validates against the bank with a clear error. The
  anode is an approximation — midpoint-of-pair positions would be more
  faithful if needed.
- `onset_f1` is a **placeholder** (binary event-vs-background F1), not a real
  onset/transition metric. For sleep, also reconsider whether dense long windows
  vs the standard 30 s epochs is the right framing.
- **Lazy loading (done).** `neuralset_task.build_{epoched,dense}` now return the
  prepared `SegmentDataset` un-materialized; `data.SegmentWindows` wraps it and
  extracts each window's signal on demand (`ds[i]`), and `data.group_split` +
  `ds.select(idx)` do a recording-level (no-leak) train/test split. The
  in-memory `ArrayWindows`/`make_loader` path is still used by `Simulated`.
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
