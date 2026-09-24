# Track 03 baselines

Four references, each a benchopt solver *and* a valid submission:

| Solver | Name | What it shows |
|---|---|---|
| `median_baseline.py` | `Median` | the contract with no weights and no training — the leaderboard floor |
| `mean_ridge.py` | `Mean-Ridge` | a scikit-learn model; `save_model` writes a joblib dump |
| `torch_linear.py` | `Torch-Linear` | the same model in PyTorch, with its own Adam loop in `fit` |
| `eegnet_reg.py` | `EEGNet` | braindecode EEGNet regressor, trained end-to-end |

Run them like any solver, or compare against yours:

```bash
benchopt run tracks/sleep_onset -d Simulated -s Median -s Mean-Ridge
benchopt run tracks/sleep_onset --config tracks/sleep_onset/starter.yml -s MyModel
```

Selectors are case-insensitive globs, so `-s "*linear*"` or `-s "eegnet*"`
also work.

Training through benchopt writes each trained submission to its own folder
`outputs/<Solver.name>/` (`submission.py` + weights) — re-run the solver to
test it, or zip that folder to upload. `tools/make_starting_kit.py` packages
each solver as an example ZIP, pulling weights from that folder when present
and shipping it untrained otherwise.
