"""Sample submission for the EEG competition — foundation-model track.

A submission is a **valid benchopt solver** subclassing ``CompetEEGSolver``.
You only implement how your *frozen* foundation model embeds windows:

- ``load_model(self, meta)``      -> the frozen model (loaded once). ``meta``
                                     carries ``sfreq, ch_names, chs_info,
                                     n_chans, n_times, n_classes, task``.
- ``time_embed(self, model, X)``  -> ``(B, T', D)`` temporal embedding for a
                                     batch of windows ``X: (B, C, T)`` (torch).
- ``embed(self, model, X)``       -> ``(B, D)`` window embedding. Optional;
                                     defaults to mean-pooling ``time_embed``.

The competition fits the scikit-learn linear head for you (on-device, via the
array API) and scores both the epoched and dense tasks — you ship only the
encoder. This sample wraps braindecode's **REVE**. braindecode is imported at
module top (no try/except fallback): if it is missing, benchopt reports the
solver as *not installed* rather than silently substituting another encoder
and masking the missing dependency.

Test it locally (from the bundle root):

    cp solution/submission.py benchmark/solvers/_submission.py
    benchopt run benchmark/ -d Simulated -s REVE
    rm benchmark/solvers/_submission.py
"""

import torch
from braindecode.models import REVE
from braindecode.models.reve import RevePositionBank

from benchmark_utils.base_solver import CompetEEGSolver

REVE_SFREQ = 200  # REVE is pretrained at 200 Hz


def _reve_channel(name):
    """Map a dataset channel name to REVE's single-electrode vocabulary.

    REVE positions channels by a *single* electrode name (standard
    10-20/10-10/10-05), so strip a leading ``"EEG "`` and reduce a bipolar
    derivation to its anode (the first electrode):
    ``"EEG Fpz-Cz" -> "Fpz"``, ``"Pz-Oz" -> "Pz"``. This is an approximation
    for bipolar montages (e.g. Sleep-EDF), which have no single position.
    """
    return name.removeprefix("EEG ").strip().split("-")[0]


def _reve_chs_info(ch_names):
    """Build REVE ``chs_info`` from dataset channel names (anode-mapped).

    Validates the mapped names against REVE's position bank up front and
    raises a clear error listing any that don't resolve — REVE itself would
    otherwise fail deep inside its position lookup with a cryptic
    ``IndexError`` when *no* channel matches.
    """
    reve_names = [_reve_channel(n) for n in ch_names]
    known = set(RevePositionBank().mapping)
    missing = sorted({n for n in reve_names if n not in known})
    if missing:
        raise ValueError(
            f"REVE has no electrode position for {missing} (mapped from "
            f"channel names {list(ch_names)}). REVE places channels by single "
            "10-20/10-10/10-05 electrode name; provide resolvable channel "
            "names or use a different montage."
        )
    return [{"ch_name": n} for n in reve_names]


class Solver(CompetEEGSolver):

    name = "REVE"

    requirements = [
        "pip::braindecode", "pip::huggingface_hub", "pip::safetensors"
    ]

    def load_model(self, meta):
        self._sfreq = float(meta.get("sfreq", REVE_SFREQ))

        # REVE places channels by single-electrode name; map the dataset's
        # channel names into its vocabulary (anode for bipolar montages).
        ch_names = meta.get("ch_names")
        chs_info = (
            _reve_chs_info(ch_names) if ch_names is not None
            else meta["chs_info"]
        )

        try:
            model = REVE.from_pretrained(
                "brain-bzh/reve-base",
                n_outputs=meta["n_classes"],
                n_times=meta["n_times"],
                chs_info=chs_info,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to load REVE from HuggingFace Hub. "
                "REVE requires to be logged in the HF interface and to accept "
                "the terms of use. Visit https://huggingface.co/"
                "brain-bzh/reve-base to accept the terms, and run "
                "`hf auth login` to log in."
            ) from e
        model.eval()
        model.requires_grad_(False)
        return model.to(self.device)

    def _resample(self, X):
        if int(self._sfreq) == REVE_SFREQ:
            return X
        n_out = int(round(X.shape[-1] * REVE_SFREQ / self._sfreq))
        return torch.nn.functional.interpolate(
            X, size=n_out, mode="linear", align_corners=False
        )

    def time_embed(self, model, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        X = self._resample(X)
        with torch.inference_mode():
            out = model(X, return_features=True)
        feats = out["features"] if isinstance(out, dict) else out
        # REVE features are (B, C, n_patches, D): collapse the channel axis
        # (mean) to get a temporal embedding (B, T'=n_patches, D).
        if feats.ndim == 4:
            feats = feats.mean(dim=1)
        elif feats.ndim == 2:  # (B, D) -> add a singleton time axis
            feats = feats[:, None, :]
        return feats  # (B, T', D)
