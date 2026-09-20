# Track 04 solver examples

`mean_pose.py` is the current reference floor for Track 04. It demonstrates
the `Solver` and `fit` contract and returns joint-angle trajectories with the
required `(B, n_joints, T)` shape.

Run it from the repository root with:

```bash
benchopt install tracks/emg_pose
benchopt run tracks/emg_pose -d Simulated -s MeanPose \
  -o "EMG-pose[training=True]"
```

This validates the task and output contract, but `MeanPose` does not export a
trained submission ZIP. To turn your own trainable model into a submission,
implement `load_model`, `fit`, and `save_model` as shown by the EEGNet solvers
in the other tracks. A training run will then write the upload-ready ZIP under
`tracks/emg_pose/outputs/`.
