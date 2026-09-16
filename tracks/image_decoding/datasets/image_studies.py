"""Image-decoding studies through the official neuralbench task pipeline.

Wraps the neuralbench ``eeg/image`` task config — see ``compet_core.nb_task``:
1.2-s epochs around each ``Image`` stimulus (−0.2 → 1.0 s), targets =
DINOv2-giant embeddings of the viewed images (``HuggingFaceImage``
extractor, computed once and cached), predefined timeline-based split.
The ``study`` parameter picks the dataset overlay:

- ``gifford2022large``     : THINGS-EEG2, the task default.
- ``grootswagers2022human``: THINGS-EEG1.
- ``xu2024alljoined``      : Alljoined-1.
- ``xu2025alljoined``      : Alljoined-1.6M — the closest warm-up proxy (the
                             hidden eval cohort uses the same 32-channel
                             Emotiv hardware); overlay ships with
                             neuralbench >= 0.3.

Requires a one-time download of the study **and** a one-time embedding pass
over the stimulus images (GPU strongly recommended — run ``benchopt
prepare`` on a compute node). The zero-dependency ``Simulated`` dataset
covers no-network smoke testing.
"""

from benchopt import BaseDataset
from benchopt.config import get_data_path

# Hard requirements of the real-data path, imported at module level so
# benchopt reports the dataset as not-installed when they are missing.
import neuralbench  # noqa: F401
import transformers  # noqa: F401

import benchmark_utils  # noqa: F401 — locates compet_core
from compet_core.data import get_device
from compet_core.nb_task import download_study, load_task

# Overlay yaml in the task's ``datasets/`` folder (None = the task default).
_OVERLAYS = {
    "gifford2022large": None,
    "grootswagers2022human": "grootswagers2022human",
    "xu2024alljoined": "xu2024alljoined",
    "xu2025alljoined": "xu2025alljoined",
}


class Dataset(BaseDataset):

    name = "Image"

    requirements = [
        "pip::neuralset", "pip::neuralfetch", "pip::neuralbench",
        "pip::mne", "pip::transformers", "scikit-learn", "pip::torch",
    ]

    parameters = {
        "study": ["gifford2022large"],
        "batch_size": [64],
        # Dataloader workers; 0 extracts windows in-process, which is what
        # shared CI/platform runners want. Raise it from the phase config
        # (``Image[num_workers=4]``) on a worker with spare cores.
        "num_workers": [0],
        # "test" restricts the study to its test split (worker staging /
        # evaluation-only runs; incompatible with the objective's
        # training=True) — see compet_core.nb_task.
        "subset": ["full"],
    }

    # Ignore loader config for prepare cache key.
    prepare_cache_ignore = ("batch_size", "num_workers")

    def prepare(self):
        # Download the study, then run the pipeline once: the extraction —
        # including the one-time DINOv2 embedding pass — caches next to the
        # data, so runs only touch warm caches. Both steps are idempotent.
        download_study(
            "eeg", "image", self._data_dir(),
            dataset=_OVERLAYS[self.study],
        )
        self._load()

    def _data_dir(self):
        path = get_data_path("neural_compet")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self, device="cpu"):
        return load_task(
            "eeg", "image",
            data_dir=self._data_dir(),
            dataset=_OVERLAYS[self.study],
            device=device,
            batch_size=self.batch_size,
            seed=self.get_seed(),
            num_workers=self.num_workers,
            subset=self.subset,
            # Run the image-embedding extractor locally (no exca cluster) and
            # cache it next to the data.
            overrides={
                "target.infra.cluster": None,
                "target.infra.folder": str(self._data_dir() / "cache"),
                "target.infra.permissions": None,
            },
        )

    def get_data(self):
        self.prepare()  # idempotent — so plain ``benchopt run`` also works
        loaders, meta = self._load(device=get_device())
        return dict(
            train_loader=loaders["train"],
            subset=self.subset,
            test_loader=loaders["test"],
            n_outputs=int(meta.pop("target_shape")[-1]),
            **{k: v for k, v in meta.items() if k != "raw_target_shape"},
        )
