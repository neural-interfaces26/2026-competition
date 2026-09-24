# Packaging a NeuralBench-trained EEGNet as a submission — DRAFT

> **Status: draft, not yet linked from the participant guide.** The
> checkpoint-key mapping in step 3 is **not yet verified against a real
> NeuralBench run** — inspect your own checkpoint's keys before trusting it.
> This page covers **EEGNet only**. Foundation models (REVE) trained through
> NeuralBench's `DownstreamWrapper` (channel adapter + probe/LoRA) are **out of
> scope** — they need a wrapper solver that does not exist in the repo yet.

## The idea

Each track already ships an **EEGNet solver that *is* the inference wrapper**.
In `load_model` it rebuilds `braindecode.models.EEGNet(...)` and loads a plain
PyTorch `state_dict` from `weights.pt`:

| Track | Solver file | `Solver.name` | Note |
|---|---|---|---|
| 01 image_decoding | `solvers/eegnet_clip.py` | `EEGNet-CLIP` | EEGNet **+ a CLIP retrieval head** — see caveat below |
| 02 bci_decoding | `solvers/eegnet.py` | `EEGNet` | plain EEGNet classifier |
| 03 sleep_onset | `solvers/eegnet_reg.py` | `EEGNet` | plain EEGNet regressor |
| 04 emg_pose | `solvers/eegnet_pose.py` | `EEGNet` | EEGNet + dense readout |

So packaging a NeuralBench-trained EEGNet reduces to: **produce that
`state_dict`, drop it next to the solver as `weights.pt`, and zip.**

## Steps

1. **Train EEGNet in NeuralBench** with the track's start kit, e.g.
   ```bash
   neuralbench eeg motor_imagery --dataset dreyer2023 -m eegnet   # drop --debug for a full run
   ```
   and note where NeuralBench writes the checkpoint.

2. **Match the architecture.** The competition solver builds
   `EEGNet(n_chans=…, n_outputs=…, n_times=…)` from `meta`, with braindecode's
   default `F1`/`D`/`kernel_length`/… If NeuralBench's `configs/<track>/eegnet.yaml`
   overrides any of those, mirror them in the solver's `load_model`, or
   `load_state_dict` will fail on shape mismatch.

3. **Extract the braindecode `state_dict`** and save it as `weights.pt`. A
   NeuralBench checkpoint wraps the module (NeuralTrain config / trainer), so
   the EEGNet tensors sit under a prefix — strip it so the keys match
   `braindecode.models.EEGNet().state_dict()`:
   ```python
   import torch
   ckpt = torch.load("neuralbench_eegnet.ckpt", map_location="cpu")
   sd = ckpt.get("state_dict", ckpt)

   # ⚠️ VERIFY this prefix against your checkpoint: print(list(sd)[:10])
   PREFIX = "model."
   net = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}
   torch.save(net, "weights.pt")
   ```

4. **Assemble the submission.** Copy the track's EEGNet solver to
   `submission.py`, put `weights.pt` next to it, and confirm `load_model`
   reads `meta["submission_dir"] / "weights.pt"`.

5. **Test locally before uploading** — a fixed-size checkpoint trained on the
   real dimensions will *not* fit `Simulated`, so validate on the prepared
   track data:
   ```bash
   benchopt prepare tracks/<track> --config tracks/<track>/starter.yml
   COMPET_SUBMISSION_DIR="$PWD/my_submission" \
       benchopt run tracks/<track> --config tracks/<track>/starter.yml -s EEGNet
   ```

6. **Zip and upload** `submission.py` + `weights.pt` at the root of the ZIP,
   through **My Submissions**.

## Caveats / open items

- **Key mapping (step 3) is unverified.** Confirm the prefix and that every
  `braindecode.models.EEGNet` parameter is present after stripping.
- **Track 01 is special:** `EEGNet-CLIP` wraps EEGNet with a retrieval head, so
  its `state_dict` is *not* a bare EEGNet — a NeuralBench retrieval EEGNet must
  match `eegnet_clip.py`'s architecture, not plain EEGNet.
- **Hyperparameters must match** NeuralBench's `configs/<track>/eegnet.yaml`.
- **REVE / foundation models are not covered** — they carry a channel adapter +
  probe (+ optional LoRA) from the `DownstreamWrapper`, which the current
  solvers do not reconstruct. That is a separate "bridge solver" task.

Once the EEGNet path is verified end-to-end, fold a trimmed version into
[`participate.md`](participate.md) (Practice 3).
