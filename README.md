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

# real data (one-time download; large for some tracks)
python tools/setup_data.py --track bci_decoding

# end-to-end ingestion + scoring on Simulated (mirrors Codabench)
python codabench/ingestion_program/ingestion.py \
    --submission-dir solution/bci_decoding --output-dir ingestion_res \
    --benchmark-dir tracks/bci_decoding --datasets Simulated
python codabench/scoring_program/scoring.py \
    --prediction-dir ingestion_res --output-dir scoring_res
```

## Build & CI

- `python tools/create_bundle.py --all` produces one `bundle_<track>.zip` per
  track, ready to upload to Codabench.
- CI runs `benchopt test` on the 4 tracks plus lint, and an end-to-end
  Docker test of the ingestion/scoring programs.
