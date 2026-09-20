# Track 03 baselines

Four references, each a benchopt solver *and* a valid submission:

| Solver | Name | What it shows |
|---|---|---|
| `median_baseline.py` | `Median` | the contract with no weights and no training — the leaderboard floor |
| `mean_ridge.py` | `Mean-Ridge` | a scikit-learn model; `save_model` writes a joblib dump |
| `torch_linear.py` | `Torch-Linear` | the same model in PyTorch, with its own Adam loop in `fit` |
| `eegnet_reg.py` | `EEGNet` | braindecode EEGNet; `eegnet_reg.pt` holds the NeuralBench start-kit weights |

Run them like any solver, or compare against yours:

```bash
benchopt run tracks/sleep_onset -d Simulated -s Median -s Mean-Ridge
benchopt run tracks/sleep_onset -s MyModel -o "Sleep-onset[training=True]"
```

Selectors are case-insensitive globs, so `-s "*linear*"` or `-s "eegnet*"`
also work.

Each one is packaged as an upload-ready ZIP in the starting kit
(`tools/make_starting_kit.py` builds it; a solver's sibling `<name>.<ext>`
travels as its `weights` file). Training through benchopt writes the same
kind of archive to `outputs/submission_<name>.zip`.
