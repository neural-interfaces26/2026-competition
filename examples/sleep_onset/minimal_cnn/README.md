# Minimal Track 03 PyTorch submission

This worked example shows a complete neural-network submission for
the Sleep Onset track:

```text
submission.py
weights.pt
```

`submission.py` defines a compact temporal CNN and the required `Solver`.
`weights.pt` is a deterministic, intentionally untrained PyTorch state dict.
It produces data-dependent mock predictions around 250 seconds and validates
model reconstruction, checkpoint loading, device placement, inference,
scoring, and leaderboard publication. It is not a baseline to beat.

## Create the upload

From this directory, run:

```bash
zip -j minimal-sleep-cnn.zip submission.py weights.pt
```

Both files must be at the archive root. Do not place them inside an additional
directory. And try submit it on Codabench!

## Adapt it

At minimum, keep the `Solver.load_model` and `predict` contract visible in
`submission.py`, replace `MinimalSleepCNN` with your model, and replace
`weights.pt` with the state dict produced by your local training pipeline.
The server runs inference only and treats the uploaded directory as read-only.
