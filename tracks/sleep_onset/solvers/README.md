# Track 03 submission examples

The examples form a three-step progression:

1. **`01_dummy_submission/`** contains an immediately uploadable,
   deliberately meaningless model. Use it only to verify the Codabench
   workflow.
2. **`02_train_and_package_with_benchopt/`** is the complete minimal training
   pathway. It shows how Benchopt calls `fit` locally and how `save_model`
   produces a ZIP containing `submission.py` and trained `weights.pt`.
3. **`03_eegnet_startkit_submission/`** is the complete, trained EEGNet
   start-kit example for the public Sleep-EDF warm-up. It includes an
   inference-only solver, its trained `weights.pt`, and an upload-ready ZIP.

`median_baseline.py` remains the active constant reference floor. The three
numbered directories are intentionally separate examples rather than active
Benchopt solvers.

Start with each directory's README.
