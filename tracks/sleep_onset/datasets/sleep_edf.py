"""Sleep-onset task on Sleep-EDF (Kemp2000Analysis), the public proxy.

Wraps the official neuralbench ``eeg/sleep_onset`` task config — see
``compet_core.nb_task``: non-overlapping 5-s windows over the pre-N2 part of
each night, target = seconds to the first stable N2 epoch (capped at 600 s,
``SleepOnsetTargetExtractor``), subject-level train/val/test split, and a
``RegressionBinSampler`` balancing the train batches across latency bins.

Requires a one-time full-study download (~78 subjects — large; prefer running
``benchopt prepare`` on a compute node). The zero-dependency ``Simulated``
dataset covers no-network smoke testing.
"""

from benchopt import BaseDataset
from benchopt.config import get_data_path

# Hard requirement of the real-data path, imported at module level so
# benchopt reports the dataset as not-installed when it is missing.
import neuralbench  # noqa: F401

import benchmark_utils  # noqa: F401 — locates compet_core
from compet_core.data import get_device
from compet_core.nb_task import download_study, load_task


class Dataset(BaseDataset):

    name = "Sleep-EDF"

    requirements = [
        "pip::neuralset", "pip::neuralfetch", "pip::neuralbench",
        "pip::mne", "scikit-learn", "pip::torch",
    ]

    parameters = {
        "batch_size": [64],
        # "test" restricts the study to its test split (worker staging /
        # evaluation-only runs) — see compet_core.nb_task.
        "subset": ["all"],
    }

    def prepare(self):
        # Idempotent one-time download of the whole study (large).
        download_study("eeg", "sleep_onset", self._data_dir())

    def _data_dir(self):
        path = get_data_path("neural_compet")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_data(self):
        self.prepare()  # idempotent — so plain ``benchopt run`` also works
        device = get_device()
        loaders, meta = load_task(
            "eeg", "sleep_onset",
            data_dir=self._data_dir(),
            device=device,
            batch_size=self.batch_size,
            seed=self.get_seed(),
            subset=self.subset,
        )
        return dict(
            train_loader=loaders["train"],
            test_loader=loaders["test"],
            **{k: v for k, v in meta.items()
               if k not in ("target_shape", "raw_target_shape")},
        )
