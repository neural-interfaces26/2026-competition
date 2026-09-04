# Context — Neural Interfaces 2026 competition code (for future sessions)

Orientation doc for anyone (human or agent) resuming work on this repo. Pairs
with `README.md` (user-facing design) — this file adds *why*, current state,
and the rough edges.

## What this is

The code for the NeurIPS 2026 neural-interfaces competition
(https://neural-interfaces26.github.io): **4 Codabench competitions**, one per
track, each powered by a **standalone benchopt benchmark** under `tracks/`,
with shared components in the installable `compet_core` package (repo root).
Formerly a 2-task proof-of-concept with a foundation-model (linear-probe)
track — that track was **dropped** (2026-09) and the repo restructured into
this monorepo.

## Key design choices (and why)

- **One benchmark per track, no `track` objective parameter.** Each
  competition holds a single task, so leaderboard keys are plain metric names
  and solvers need no gating.
- **A submission is a trained model, evaluated inference-only.** The user
  decided (2026-09-04): no training and no fine-tuning on the platform. The
  contract (`compet_core/base_solver.py::CompetSolver`) is
  `load_model(meta) -> model with predict(X)`; the optional
  `fit(model, train_loader)` only runs locally (ingestion sets
  `COMPET_INFERENCE_ONLY=1`, checked in `CompetSolver.run`). Weights ship
  next to `submission.py`; `meta["weights_dir"]` points there
  (`COMPET_SUBMISSION_DIR`, set by ingestion; defaults to the solver file's
  dir locally).
- **Data layer = neuralbench task configs.** `compet_core/nb_task.py` builds
  `neuralbench.data.Data` from the configs shipped in the wheel
  (`neuralbench/tasks/<modality>/<task>/config.yaml` + per-dataset overlays),
  bypassing `~/.neuralbench/config.json` and exca cluster infra (explicit
  `data_dir`, local execution). Loaders are re-wrapped to the competition
  contract (`(X, y, info)` torch batches on `meta["device"]`) so submissions
  stay pure-PyTorch. neuralbench pins `torch==2.6` + needs Python ≥ 3.12.
- **`compet_core` needs no install.** Each track ships a
  `benchmark_utils/__init__.py` that walks up from the benchmark dir to the
  first parent holding `compet_core/` (repo root in a checkout, bundle root
  on Codabench) and puts it on `sys.path`; every benchmark module does
  `import benchmark_utils` before `from compet_core import ...`. This is what
  lets CI work on the **private** repo (a `pip::git+...` requirement cannot
  be installed there — that was the first CI failure). The root
  `pyproject.toml` remains for optional `pip install -e .` convenience.
- **Bundles**: `tools/create_bundle.py --track <t>` ships the track's
  benchmark under the canonical `benchmark/` name + `compet_core/` + shared
  ingestion/scoring + `codabench/competition_<t>.yaml` (as
  `competition.yaml`) + `solution/<t>/`.
- The legacy direct-neuralset path (`compet_core/neuralset_task.py`,
  `tracks/bci_decoding/datasets/moabb_mi.py`) is kept alongside the
  neuralbench path until the latter is fully validated on real data; then it
  can be dropped.

## Per-track notes

- **bci_decoding** — `datasets/bci_studies.py` wraps the `eeg/motor_imagery`
  config; `study` parameter ∈ {stieger2021 (default proxy, big),
  tangermann2012 (small, smoke tests), dreyer2023 (official candidate,
  2-class)}. One-hot targets → argmax in `target_transform`.
- **sleep_onset** — `eeg/sleep_onset` config (Kemp2000Analysis): 5-s windows,
  scalar target = seconds to first stable N2, cap 600 s; ranking metric bMAE
  (bins [0, 40, 90, 300, 600], reimplemented numpy-side in
  `compet_core/metrics.py`, mirrors `neuralbench.metrics.BinnedMAE`).
- **image_decoding** — `eeg/image` config (Gifford2022Large); targets are
  DINOv2-giant embeddings via the `HuggingFaceImage` extractor (heavy one-time
  pass — the dataset overrides `target.infra.{cluster:None,folder:...}` to run
  it locally with a cache). Objective: cosine retrieval against the unique
  test-split embeddings, top-5/top-1.
- **emg_pose** — Simulated only. **No neuralfetch study exists for Salter2024
  emg2pose** (only `Sivakumar2024Emg2qwerty`) and no `emg/pose` task config —
  the real loader is upstream/follow-up work (coordinate with the neuralbench
  team). Note the website/neuralbench inconsistency: the site says EMG2Pose,
  the neuralbench example implements emg2qwerty; the user chose EMG2Pose.

## Current state (validated 2026-09-04)

- All 4 tracks: `benchopt run tracks/<t> -d Simulated` green +
  `benchopt test --skip-install` green + flake8 clean.
- End-to-end ingestion → scoring on Simulated for all 4 tracks; expected
  score keys; `create_bundle.py --all` builds the 4 bundles.
- Inference-only gating verified (fit skipped ⇒ untrained model scores).
- Real-data validation of the neuralbench layer: **passed on margaret**
  (`BCI[study=tangermann2012]`, MeanLogReg → bal-acc 0.266 on the 4-class
  subject-level split, chance 0.25 — sane for that baseline). Data lands in
  `tracks/bci_decoding/data/neural_compet/` (NEMAR/BIDS download, ~1.7 GB,
  ~2 h mostly at slow S3 throughput). Sleep-EDF and stieger2021 validation
  jobs submitted (531149/531150). The container is CPU-only/small and
  neuralset needs Python ≥ 3.12 → env at `.venv-margaret/` (torch 2.6 cu124,
  benchopt installed from the `~/workspace/benchopt` checkout — PyPI 1.9.1 is
  too old). On margaret use `sbatch`, never detached tmux (login node reaps
  it).

## TODOs / open issues

- Validate on real data: tangermann2012 done; stieger2021 + sleep_edf jobs
  submitted; things_eeg2 (large + DINOv2 embedding pass) still to run.
- REVE frozen-probe baseline per track (linear_probe.py kept for this) —
  braindecode envs may clash with the neuralbench torch pin.
- emg_pose real data loader (Salter2024) — upstream.
- Hidden-test isolation on Codabench (sealed phase) — still deferred; public
  proxy test splits for now.
- `benchopt_release` CI job disabled until benchopt 1.9.2 hits PyPI.
- `terms.md` is lorem ipsum; competition yaml dates are placeholders.
- The old MOABB-MI dataset + `neuralset_task.py` → drop after nb_task real
  validation.

## Pointers

- Plan: `~/.claude/plans/we-are-organizing-a-linked-tulip.md`.
- neuralbench docs: https://facebookresearch.github.io/neuroai/neuralbench/
  (biosignal_challenge_2026 examples = one per track).
- Previous single-benchmark design: git history before the
  `4-track-restructure` branch (see `RFC restructure as monorepo` commit).
