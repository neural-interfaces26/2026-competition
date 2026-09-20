# Track 03 submission examples

The examples form a three-step progression:

1. **`01_dummy_submission/`** — an immediately uploadable, deliberately
   meaningless model. Use it only to verify the Codabench workflow.
2. **`03_train_and_package_with_benchopt/`** — the minimal training
   pathway: benchopt calls `fit` locally and `save_model` produces a ZIP
   holding `submission.py` and the trained `weights.pt`.
3. **`02_eegnet_startkit_submission/`** — the trained EEGNet start-kit
   example for the public Sleep-EDF warm-up, an inference-only solver plus
   its `weights.pt`, for reproducing a published baseline.

Archives are not committed: `python tools/make_examples.py --track
sleep_onset` builds them from the folder's sources, so they cannot drift.

`median_baseline.py` and `eegnet_reg.py` are the track's active benchopt
solvers. The numbered directories are examples, not solvers — benchopt
only discovers `solvers/*.py`, so nothing inside them is run by
`benchopt run`.

Start with each directory's README.
