# Track 04 baselines

Four references, each a benchopt solver *and* a valid submission:

| Solver | Name | What it shows |
|---|---|---|
| `mean_pose.py` | `MeanPose` | the contract with no weights and no training — the train-mean-pose floor |
| `ridge_pose.py` | `Ridge` | a scikit-learn model; `save_model` writes a joblib dump |
| `torch_linear.py` | `Torch-Linear` | the same model in PyTorch, with its own Adam loop in `fit` |
| `eegnet_pose.py` | `EEGNet` | braindecode EEGNet with a dense readout, trained end-to-end |

The two linear baselines are *per-time-step* readouts: the Simulated EMG is an
instantaneous linear mixture of the joint angles, so a linear map at each time
step already recovers the pose above the noise floor.

Run them like any solver, or compare against yours:

```bash
benchopt run tracks/emg_pose -d Simulated -s MeanPose -s Ridge
benchopt run tracks/emg_pose -s MyModel -o "EMG-pose[training=True]"
```

Selectors are case-insensitive globs, so `-s "*linear*"` or `-s "eegnet*"`
also work.

Training through benchopt writes each trained submission to its own folder
`outputs/<Solver.name>/` (`submission.py` + weights) — re-run the solver to
test it, or zip that folder to upload. `tools/make_starting_kit.py` packages
each solver as an example ZIP, pulling weights from that folder when present
and shipping it untrained otherwise.
