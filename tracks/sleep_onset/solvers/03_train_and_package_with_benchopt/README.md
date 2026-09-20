# Minimal trainable submission

This is the smallest complete example of training and packaging a submission
with Benchopt. It uses an intentionally simple PyTorch linear model so the
workflow remains visible.

Benchopt discovers solver files placed directly in the track's `solvers/`
directory. From the repository root, first copy this example there:

```bash
cp tracks/sleep_onset/solvers/03_train_and_package_with_benchopt/linear_reg.py \
  tracks/sleep_onset/solvers/linear_reg.py
```

Then install and run the benchmark:

```bash
benchopt install tracks/sleep_onset
benchopt run tracks/sleep_onset -d Simulated -s Linear \
  -o "Sleep-onset[training=True]"
```

During this local run, Benchopt:

1. creates the model with `load_model`
2. trains it with `fit`
3. writes `weights.pt` with `save_model`
4. creates `tracks/sleep_onset/outputs/submission_Linear.zip`

The simulated dataset only validates the workflow. To train a meaningful
model, select a permitted development dataset instead. Codabench never calls
`fit` or `save_model`. It only loads the exported weights and runs inference.

The copied `tracks/sleep_onset/solvers/linear_reg.py` is only a local Benchopt
entry point and can be deleted after the example.
