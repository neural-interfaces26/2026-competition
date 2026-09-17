"""Sleep-onset task on Sleep-EDF (Kemp2000Analysis), the public proxy.

Wraps the official neuralbench ``eeg/sleep_onset`` task config — see
``benchmark_utils.nb_task``: non-overlapping 5-s windows over the pre-N2 part
of
each night, target = seconds to the first stable N2 epoch (capped at 600 s,
``SleepOnsetTargetExtractor``), subject-level train/val/test split, and a
``RegressionBinSampler`` balancing the train batches across latency bins.

Requires a one-time full-study download (~78 subjects — large; prefer running
``benchopt prepare`` on a compute node). The zero-dependency ``Simulated``
dataset covers no-network smoke testing.
"""

import shutil
import subprocess

from benchopt import BaseDataset
from benchopt.config import get_data_path

# Hard requirement of the real-data path, imported at module level so
# benchopt reports the dataset as not-installed when it is missing.
import neuralbench  # noqa: F401

from benchmark_utils.data import get_device
from benchmark_utils.nb_task import download_study, load_task


class Dataset(BaseDataset):

    name = "Sleep-EDF"

    requirements = [
        "pip::neuralset", "pip::neuralfetch", "pip::neuralbench",
        "pip::mne", "scikit-learn", "pip::torch",
    ]

    parameters = {
        "batch_size": [64],
        # Dataloader workers; 0 extracts windows in-process, which is what
        # shared CI/platform runners want. Raise it from the phase config
        # (``Sleep-EDF[num_workers=4]``) on a worker with spare cores.
        "num_workers": [0],
        # "test" restricts the study to its test split (worker staging /
        # evaluation-only runs; incompatible with the objective's
        # training=True) — see benchmark_utils.nb_task.
        "subset": ["full"],
    }

    # Ignore loader config for prepare cache key.
    prepare_cache_ignore = ("batch_size", "num_workers")

    def prepare(self):
        # Seed from S3 when possible, then download (validates the seeded
        # files, fetches anything missing) and run the pipeline once: the
        # extraction (filtering, segmenting, targets) caches next to the
        # data, so runs only touch warm caches. All steps are idempotent.
        self._seed_from_s3()
        download_study("eeg", "sleep_onset", self._data_dir())
        self._load()

    def _seed_from_s3(self):
        """Fast path: PhysioNet's public S3 mirror — minutes, where the
        origin's throttled HTTP takes hours. No credentials needed
        (``--no-sign-request``); fills both layouts the neuralfetch study
        checks (MNE's flat files + wget's mirror). Best-effort: skipped
        without the aws CLI, and any failure falls back to the regular
        download."""
        study = self._data_dir() / "Kemp2000Analysis" / "physionet-sleep-data"
        src = "s3://physionet-open/sleep-edfx/1.0.0/sleep-cassette/"
        mirror = study / "physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
        # The mirror is what a complete seed creates last: an interrupted one
        # leaves it absent and re-syncs (``aws s3 sync`` is incremental).
        if mirror.is_dir() or shutil.which("aws") is None:
            return
        for dst in (study, mirror):
            subprocess.run(
                ["aws", "s3", "sync", "--no-sign-request",
                 "--only-show-errors", src, str(dst)], check=False)

    def _data_dir(self):
        path = get_data_path("neural_compet")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self, device="cpu"):
        return load_task(
            "eeg", "sleep_onset",
            data_dir=self._data_dir(),
            device=device,
            batch_size=self.batch_size,
            seed=self.get_seed(),
            num_workers=self.num_workers,
            subset=self.subset,
        )

    def get_data(self):
        self.prepare()  # idempotent — so plain ``benchopt run`` also works
        loaders, meta = self._load(device=get_device())
        return dict(
            # subset="test" stages the evaluation split only: no train
            # split to hand over, which is how a run learns it cannot train.
            train_loader=None if self.subset == "test" else loaders["train"],
            test_loader=loaders["test"],
            **{k: v for k, v in meta.items()
               if k not in ("target_shape", "raw_target_shape")},
        )
