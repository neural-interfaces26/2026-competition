# Track 02 solver examples

These runnable solvers show how to develop a Track 02 model against the same
Benchopt contract used by Codabench:

- `mean_logreg.py` is the scikit-learn example. Its `MeanLogReg` model, defined
  in `benchmark_utils/baselines.py`, uses per-channel mean features, a
  `StandardScaler`, and logistic regression. It is a compact reference floor.
- `eegnet.py` is the trainable neural-network example. It uses a Braindecode
  EEGNet and implements `save_model`, so training exports an upload-ready ZIP
  containing `submission.py` and `weights.pt`.

## Produce a test submission

From the repository root:

```bash
benchopt install tracks/bci_decoding
benchopt run tracks/bci_decoding -d Simulated -s EEGNet \
  -o "BCI-decoding[training=True]"
```

The ZIP is written under `tracks/bci_decoding/outputs/`. The simulated run
checks training, weight export, reloading, and prediction. Select a permitted
development dataset instead of `Simulated` to train a meaningful model.
