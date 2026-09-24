# Track 01 baselines

Four references, each a benchopt solver *and* a valid submission:

| Solver | Name | What it shows |
|---|---|---|
| `mean_embedding.py` | `MeanEmbedding` | the contract with no weights and no training — the retrieval floor |
| `mean_ridge.py` | `Mean-Ridge` | a scikit-learn model; `save_model` writes a joblib dump |
| `torch_linear.py` | `Torch-Linear` | the same model in PyTorch, with its own Adam loop in `fit` |
| `eegnet_clip.py` | `EEGNet-CLIP` | braindecode EEGNet trained with a CLIP retrieval loss, end-to-end |

Run them like any solver, or compare against yours:

```bash
benchopt run tracks/image_decoding -d Simulated -s MeanEmbedding -s Mean-Ridge
benchopt run tracks/image_decoding --config tracks/image_decoding/starter.yml -s MyModel
```

Selectors are case-insensitive globs, so `-s "*linear*"` or `-s "eegnet*"`
also work.

Training through benchopt writes each trained submission to its own folder
`outputs/<Solver.name>/` (`submission.py` + weights) — re-run the solver to
test it, or zip that folder to upload. `tools/make_starting_kit.py` packages
each solver as an example ZIP, pulling weights from that folder when present
and shipping it untrained otherwise.
