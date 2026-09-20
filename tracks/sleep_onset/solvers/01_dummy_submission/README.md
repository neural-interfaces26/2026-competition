# Dummy Track 03 submission

This deliberately untrained example represents what a final Track 03
submission can look like:

```text
dummy_submission.py
dummy_weights.pt
dummy_config.json
```

It includes a tiny linear PyTorch model, shipped dummy weights, and an optional
configuration file. Its predictions are meaningless. It exists only to test
the complete upload, inference, scoring, and leaderboard workflow without
training anything.

## Zip and upload it

From this directory, run:

```bash
python3 make_zip.py
```

The script creates `dummy-sleep-linear.zip` with the required
`submission.py` at its root. Upload that ZIP through the Track 03
**My Submissions** tab.

The generated ZIP is also committed in this directory so it can be downloaded
and submitted directly.

## Next step

- Adapt this structure to your model and replace the dummy weights.
- Or continue with `../02_train_and_package_with_benchopt/`, which demonstrates
  the full local training and automatic packaging pathway.
- `../03_eegnet_startkit_submission/` will later receive compatible trained
  weights for the realistic EEGNet start-kit submission.
