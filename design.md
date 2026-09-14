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
- **Training is opt-in through the objective's `training` parameter.** A
  plain `benchopt run tracks/<t>` is inference-only — exactly what the
  platform does — and the solvers' optional `fit(model, train_loader)` only
  runs when `-o "<objective>[training=True]"` is selected (how baselines are
  trained and how participants train through the starting kit; the flag also
  lands in the results as `objective_training`). A training phase on the
  platform is just a phase config whose native `objective:` key selects the
  training variant — no env vars, no custom keys. The objectives'
  `test_config = {"training": True}` makes `benchopt test` exercise the full
  contract. NB: cluster validation scripts must add the `-o` flag to fit the
  baselines.
- **A submission is a trained model, evaluated inference-only.** The user
  decided (2026-09-04): no training and no fine-tuning on the platform. The
  contract (`compet_core/base_solver.py::CompetSolver`) is
  `load_model(meta) -> model with predict(X)`; the optional
  `fit(model, train_loader)` only runs when the objective's `training`
  parameter is selected (see the dedicated bullet below). Weights ship
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
- **CPU/GPU via benchopt's requirements dict.** Objectives declare
  `requirements = {"cpu": ["scikit-learn", "pytorch-cpu"], "gpu": [...,
  "pytorch-gpu"]}` (conda-forge metapackages), selected by `benchopt install
  [--gpu]`; the default cpu variant is what CI's test env installs.
  `CompetSolver.requirements` is empty — torch/sklearn come from the
  objective env; submissions declare only their extras.
- **CI = benchopt's reusable workflow, tamed via `test_config.py` hooks.**
  The hook mechanism resolves `check_<any test function>` — beyond the two
  run-time hooks, each track defines `check_test_dataset_install` (real
  datasets: their pip stack pins CUDA `torch==2.6`, exceeding runner disk)
  and `check_test_solver_install` for the braindecode solvers (pip pulls a
  CUDA torchaudio that cannot load against CPU torch). Neither hook nor the
  cpu/gpu dict is documented in the `using-benchopt` skill — see
  `~/workspace/benchopt/note_skill_update_test_config.md` for the planned
  upstream skill addition.
- **`compet_core` needs no install.** Each track ships a
  `benchmark_utils/__init__.py` that walks up from the benchmark dir to the
  first parent holding `compet_core/` (repo root in a checkout, bundle root
  on Codabench) and puts it on `sys.path`; every benchmark module does
  `import benchmark_utils` before `from compet_core import ...`. This is what
  lets CI work on the **private** repo (a `pip::git+...` requirement cannot
  be installed there — that was the first CI failure). The root
  `pyproject.toml` remains for optional `pip install -e .` convenience.
- **Ingestion = a thin wrapper around `benchopt run` (2026-09-09).** The
  phase's Codabench `input_data` holds a `config.yaml` that is a *native*
  `benchopt run --config` file (`dataset`, `seed`, `no_timeout`, ...) plus
  two competition-only keys ingestion strips before the run (benchopt
  rejects unknown config options): `data_home` (-> `$BENCHOPT_DATA_HOME`)
  and `scoring.columns` (leaderboard key -> dataframe column, forwarded to
  and parsed by scoring — missing/NaN columns fail loudly). Ingestion copies
  the benchmark to a writable workdir (with `compet_core` as sibling for the
  `benchmark_utils` walk), drops in `input_data/datasets/*.py`
  (sealed-phase splits — the hidden-test mechanism: upload as a private
  Codabench dataset, never in the public repo) and the submission solver,
  then execs `benchopt run ... --no-cache` (`BENCHOPT_DEBUG=true` so a
  solver error exits nonzero with the traceback). Dev-phase configs live in
  `codabench/phases/dev/<track>/`. Planned benchopt 1.10 features (file
  paths for `-s`/`-d`/`--output`, see
  `~/workspace/benchopt/note_feature_path_selectors.md`) will delete the
  copy steps.
- **One Docker recipe, 4 images.** `tools/Dockerfile` takes
  `--build-arg TRACK=<t>` and bakes `tracks/<t>` at `/compet/benchmark`
  (`$COMPET_BENCHMARK_DIR`) + `compet_core` + the programs; deps from
  `requirements.txt` are the submissions' dependency contract (no install at
  submission time; benchopt from the main tarball until 1.10 is on PyPI).
  Base = `python:3.12-slim` + pip `torch==2.6.0` (cu124 wheels bundle the
  CUDA runtime): neuralset needs Python ≥ 3.12 while `pytorch/pytorch`
  images ship 3.11 — unpinned, pip silently resolved the ancient py311
  neuralset 0.0.2, hence the pins in requirements.txt. Data is
  downloaded on the docker host with the same image
  (`docker run -v <host>:/data <img> benchopt prepare $COMPET_BENCHMARK_DIR
  -d <ds>`; `ENV BENCHOPT_DATA_HOME=/data`) and bind-mounted as `/data` for
  scoring runs — **to verify**: the self-hosted Codabench compute worker
  must support the extra volume mount (fallback: private derived image with
  `COPY data /data`). `tools/run_docker.py --track <t>` is the local test.
- **Bundles**: `tools/create_bundle.py --track <t>` ships the track's
  benchmark under the canonical `benchmark/` name + `compet_core/` (both
  kept as no-docker fallback) + shared ingestion/scoring +
  `codabench/competition_<t>.yaml` (as `competition.yaml`) +
  `solution/<t>/` + `codabench/phases/dev/<t>/` as
  `dev_phase/input_data/`. `data/` dirs are always skipped.
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
- **image_decoding** — `datasets/image_studies.py` wraps the `eeg/image`
  config; `study` parameter ∈ {gifford2022large (task default),
  grootswagers2022human, xu2024alljoined, **xu2025alljoined — the chosen
  warm-up proxy** (same 32-ch Emotiv hardware as the hidden eval cohort;
  overlay needs neuralbench ≥ 0.3)}. Targets are DINOv2-giant embeddings via
  the `HuggingFaceImage` extractor (heavy one-time pass — the dataset
  overrides `target.infra.{cluster:None,folder:...}` to run it locally with a
  cache). Objective: cosine retrieval against the unique test-split
  embeddings, top-5/top-1.
- **emg_pose** — Simulated only for now, but **unblocked upstream**: since
  neuralbench/neuralfetch 0.3.x, `Salter2024Emg2pose` (corpus NM000281) and
  the `emg/pose` task config exist — implementing the real dataset needs the
  pinned stack bumped from 0.2.3 to 0.3.x (re-validate the other tracks with
  it, `exca` pin included).

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

- Validate on real data: tangermann2012 done (bal-acc 0.266 — identical on the 0.2.3 and 0.3.1 stacks, revalidated 2026-09-11); sleep_edf done
  (Median → bMAE 165.5 s vs official EEGNet-sleep 143.3 s — sane floor);
  **stieger2021 hit the 24 h SLURM limit mid-download** (2026-09-05; NEMAR S3
  often throttles to ~100-250 kB/s and the study is tens of GB) — finished
  files persist, so resubmit `~/workspace/tmp/sbatch_bci_stieger.sh` with a
  longer `--time` to resume; the image studies (large + DINOv2 embedding pass)
  still to run. NB: editing the NFS working tree while a cluster run is live
  trips benchopt's "class changed between pickle and unpickle" cache guard —
  `benchopt clean tracks/<t>` and rerun.
- REVE frozen-probe baseline per track (linear_probe.py kept for this) —
  braindecode envs may clash with the neuralbench torch pin.
- emg_pose real data loader — bump the neuro stack to 0.3.x and wrap the new `emg/pose` task config.
- Hidden-test isolation on Codabench: mechanism in place (final phase =
  private Codabench `input_data` dataset with `config.yaml` +
  `datasets/*.py` sealed split); the sealed dataset files themselves remain
  to be written. Public proxy test splits for now.
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
