"""BCI-decoding studies through the official neuralbench task pipeline.

Wraps the neuralbench ``eeg/motor_imagery`` task config (study, subject-level
split, 4-s stimulus windows, one-hot labels) — see ``benchmark_utils.nb_task``.
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

from benchmark_utils.data import get_device
from benchmark_utils.nb_task import download_study, load_task

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
        # Dataloader workers; 0 extracts windows in-process, which is what
        # shared CI/platform runners want. Raise it from the phase config
        # (``BCI[num_workers=4]``) on a worker with spare cores.
        "num_workers": [0],
        # "test" restricts the study to its test split (worker staging /
        # evaluation-only runs; incompatible with the objective's
        # training=True) — see benchmark_utils.nb_task.
        "subset": ["full"],
    }

    test_parameters = {
        "study": ["tangermann2012"],
        "batch_size": [32],
    }

    # Ignore loader config for prepare cache key.
    prepare_cache_ignore = ("batch_size", "num_workers")

    def prepare(self):
        # Download the study, then run the pipeline once: the extraction
        # (filtering, segmenting, targets) caches next to the data, so runs
        # only touch warm caches. Both steps are idempotent.
        download_study(
            "eeg", "motor_imagery", self._data_dir(),
            dataset=_OVERLAYS[self.study],
        )
        self._load()

    def _data_dir(self):
        path = get_data_path("neural_compet")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self, device="cpu"):
        return load_task(
            "eeg", "motor_imagery",
            data_dir=self._data_dir(),
            dataset=_OVERLAYS[self.study],
            device=device,
            batch_size=self.batch_size,
            seed=self.get_seed(),
            num_workers=self.num_workers,
            subset=self.subset,
            # One-hot ``(K,)`` -> integer class label.
            target_transform=lambda y: y.argmax(-1),
        )

    def get_data(self):
        self.prepare()  # idempotent — so plain ``benchopt run`` also works
        loaders, meta = self._load(device=get_device())
        return dict(
            # subset="test" stages the evaluation split only: no train
            # split to hand over, which is how a run learns it cannot train.
            train_loader=None if self.subset == "test" else loaders["train"],
            test_loader=loaders["test"],
            n_classes=int(meta.pop("raw_target_shape")[-1]),
            **{k: v for k, v in meta.items() if k != "target_shape"},
        )
