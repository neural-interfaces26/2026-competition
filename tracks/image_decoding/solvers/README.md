# Track 01 solver examples

These runnable solvers show how to develop a Track 01 model against the same
Benchopt contract used by Codabench:

- `mean_embedding.py` is the minimal reference floor. It demonstrates
  `Solver`, `load_model`, `fit`, and the required embedding output.
- `eegnet_clip.py` is the trainable example. It uses a Braindecode EEGNet with
  a CLIP-style retrieval loss and implements `save_model`, so training exports
  an upload-ready ZIP containing `submission.py` and `weights.pt`.

## Produce a test submission

From the repository root:

```bash
benchopt install tracks/image_decoding
benchopt run tracks/image_decoding -d Simulated -s EEGNet-CLIP \
  -o "Image-decoding[training=True]"
```

The ZIP is written under `tracks/image_decoding/outputs/`. The simulated run
checks training, weight export, reloading, and prediction. Select a permitted
development dataset instead of `Simulated` to train a meaningful model.
