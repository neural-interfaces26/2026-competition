# Dummy Track 03 submission

This zero-training example is the shortest path from the repository to a
complete Codabench upload for Track 03, Sleep Onset:

```text
submission.py
weights.pt
config.json
```

`submission.py` defines a compact temporal CNN and the required `Solver`.
`weights.pt` is a deterministic, intentionally untrained PyTorch state dict.
`config.json` is a minimal example of an additional non-Python file read from
`meta["submission_dir"]` during model loading.
It produces data-dependent mock predictions around 250 seconds and validates
model reconstruction, checkpoint loading, device placement, inference,
scoring, and leaderboard publication. It is not a baseline to beat.

No training is required. This example exists only to make the complete upload
contract tangible before you train a real model.

## Create the upload

From this directory, run:

```bash
zip -j dummy-sleep-cnn.zip submission.py weights.pt config.json
```

All three files must be at the archive root. Do not place them inside an
additional directory. Upload the resulting ZIP through the Track 03
**My Submissions** tab.

## Adapt it

At minimum, keep the `Solver.load_model` and `predict` contract visible in
`submission.py`, replace `MinimalSleepCNN` with your model, and replace
`weights.pt` with the state dict produced by your local training pipeline.
Keep, adapt, or remove `config.json` according to your model's needs.
The server runs inference only and treats the uploaded directory as read-only.
For a trainable reference that can export learned weights, return to the parent
directory and use `eegnet_reg.py`.
