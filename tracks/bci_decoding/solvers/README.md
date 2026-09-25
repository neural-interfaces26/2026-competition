# Track 02 baselines

Four references, each a benchopt solver *and* a valid submission:

| Solver | Name | What it shows |
|---|---|---|
| `constant.py` | `Constant` | the contract with no weights and no training — the leaderboard floor (chance) |
| `mean_logreg.py` | `MeanLogReg` | a scikit-learn model: mean-over-time features + logistic regression |
| `torch_linear.py` | `Torch-Linear` | the same model in PyTorch, with its own Adam loop in `fit` |
| `eegnet.py` | `EEGNet` | braindecode EEGNet, trained end-to-end |

Run them like any solver from `tracks/bci_decoding` folder, or compare against yours:

```bash
# Inference only on simulated
benchopt run --config starter.yml -s Constant -s MeanLogReg

# Training a model with name MyModel
benchopt run --config training.yml -s MyModel
```

Selectors are case-insensitive globs, so `-s "*linear*"` or `-s "eegnet*"`
also work, as well as paths to a submission compatible python file.

Training through benchopt writes each trained submission to its own folder
`outputs/<Solver.name>/` (`submission.py` + weights) — re-run the solver to
test it, or zip that folder to upload. `tools/make_starting_kit.py` packages
each solver as an example ZIP, pulling weights from that folder when present
and shipping it untrained otherwise.
