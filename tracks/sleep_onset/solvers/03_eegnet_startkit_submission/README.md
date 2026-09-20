# EEGNet start-kit submission

This directory contains the inference code for the realistic EEGNet submission
derived from the Track 03 start kit. Unlike the minimal linear example, it
intentionally omits `fit` and `save_model`. Codabench only needs to reconstruct
the architecture, load the trained weights, and run `predict`.

The directory is not yet a complete submission. A compatible `weights.pt`
exported from the official start kit must be added first. Those weights must
contain the state dictionary expected by `EEGNetRegressor.net`.

Once the weights are available, the upload archive will contain:

```text
submission.py
weights.pt
```

The source `eegnet_reg.py` will be copied into the archive as `submission.py`.
