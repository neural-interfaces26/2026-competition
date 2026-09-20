# Track 03 solver examples

This directory provides three complementary examples for understanding a
Track 03 submission:

- `dummy_submission/` contains a complete, zero-training upload with
  `submission.py`, dummy `weights.pt`, and an optional `config.json`. Zip those
  three files to test the full Codabench path immediately.
- `median_baseline.py` is the minimal reference floor and demonstrates the
  regression contract.
- `eegnet_reg.py` is the trainable example. It uses a Braindecode EEGNet and
  implements `save_model`, so training exports an upload-ready ZIP containing
  `submission.py` and learned `weights.pt`.

## Produce a trained test submission

From the repository root:

```bash
benchopt install tracks/sleep_onset
benchopt run tracks/sleep_onset -d Simulated -s EEGNet \
  -o "Sleep-onset[training=True]"
```

The ZIP is written under `tracks/sleep_onset/outputs/`. The simulated run
checks training, weight export, reloading, and prediction. Select a permitted
development dataset instead of `Simulated` to train a meaningful model.
