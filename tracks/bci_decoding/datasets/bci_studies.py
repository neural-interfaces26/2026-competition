"""BCI-decoding studies through the official neuralbench task pipeline.

Wraps the neuralbench ``eeg/motor_imagery`` task config (study, subject-level
split, 4-s stimulus windows, one-hot labels) — see ``compet_core.nb_task``.
The ``study`` parameter picks the dataset overlay:

- ``stieger2021``    : Stieger2021Continuous — the competition's public proxy
                       (62 subjects, 4-class; large download).
- ``tangermann2012`` : BNCI2014_001 — small, handy for real-data smoke tests.
- ``dreyer2023``     : Dreyer2023Large — official-candidate dataset
                       (2-class, predefined train/test subjects).

Requires a one-time download (``benchopt prepare``); the zero-dependency
``Simulated`` dataset covers no-network smoke testing.
"""

from benchopt import BaseDataset
from benchopt.config import get_data_path

# Hard requirements of the real-data path, imported at module level so
# benchopt reports the dataset as not-installed when they are missing.
import moabb  # noqa: F401
import neuralbench  # noqa: F401

import benchmark_utils  # noqa: F401 — locates compet_core
from compet_core.data import get_device
from compet_core.nb_task import download_study, load_task

# Overlay yaml in the task's ``datasets/`` folder (None = the task default).
_OVERLAYS = {
    "stieger2021": None,
    "tangermann2012": "tangermann2012",
    "dreyer2023": "dreyer2023",
}


class Dataset(BaseDataset):

    name = "BCI"

    requirements = [
        "pip::neuralset", "pip::neuralfetch", "pip::neuralbench",
        "pip::moabb", "pip::mne", "scikit-learn", "pip::torch",
    ]

    parameters = {
        "study": ["stieger2021"],
        "batch_size": [64],
        # "test" restricts the study to its test split (worker staging /
        # evaluation-only runs) — see compet_core.nb_task.
        "subset": ["all"],
    }

    test_parameters = {
        "study": ["tangermann2012"],
        "batch_size": [32],
    }

    def prepare(self):
        # Idempotent one-time download of the selected study.
        download_study(
            "eeg", "motor_imagery", self._data_dir(),
            dataset=_OVERLAYS[self.study],
        )

    def _data_dir(self):
        path = get_data_path("neural_compet")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_data(self):
        self.prepare()  # idempotent — so plain ``benchopt run`` also works
        device = get_device()
        loaders, meta = load_task(
            "eeg", "motor_imagery",
            data_dir=self._data_dir(),
            dataset=_OVERLAYS[self.study],
            device=device,
            batch_size=self.batch_size,
            seed=self.get_seed(),
            subset=self.subset,
            # One-hot ``(K,)`` -> integer class label.
            target_transform=lambda y: y.argmax(-1),
        )
        return dict(
            train_loader=loaders["train"],
            test_loader=loaders["test"],
            n_classes=int(meta.pop("raw_target_shape")[-1]),
            **{k: v for k, v in meta.items() if k != "target_shape"},
        )
