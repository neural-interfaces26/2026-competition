# Dummy Track 03 submission

This deliberately untrained example represents what a complete Track 03
submission can look like:

```text
dummy_submission.py
dummy_weights.pt
dummy_config.json
```

It includes custom PyTorch model code, shipped weights, and an optional
configuration file. Its predictions are meaningless. Its only purpose is to
validate ZIP ingestion, model reconstruction, weight loading, inference,
scoring, and leaderboard publication without requiring training.

## Zip and upload it

From this directory, run:

```bash
zip -j dummy-sleep-cnn.zip \
  dummy_submission.py dummy_weights.pt dummy_config.json
```

The files must be at the ZIP root. Upload `dummy-sleep-cnn.zip` through the
Track 03 **My Submissions** tab. Codabench discovers the `Solver` in any
root-level Python file, although `submission.py` remains the standard filename
for a real submission.

## Next step

- Adapt this structure to your model, rename `dummy_submission.py` to
  `submission.py`, and replace the dummy weights.
- Or return to the parent directory and train the real worked example in
  `eegnet_reg.py`, which exports an upload-ready ZIP with learned weights.
