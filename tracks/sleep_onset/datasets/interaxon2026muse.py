"""Sleep-onset task on the public Muse data (Interaxon2026Muse), the warm-up.

Wraps neuralbench's streamed Track 3 task, ``eeg/_sleep_onset_stream``,
whose default dataset this is — see ``benchmark_utils.nb_task``: the Track 3
training release (NEMAR nm000287 v1.0.0: 540 at-home recordings of four Muse
channels at 128 Hz) cut into non-overlapping 5-s windows from the start of
each recording up to its first N2 epoch, the signal at its native rate,
unfiltered, in microvolts, target = seconds to the first N2 epoch (capped at
600 s, ``SleepOnsetTargetExtractor``), and a ``RegressionBinSampler``
balancing the train batches across latency bins. The supplied session split
holds 500 training and 40 test recordings, every test participant also seen
in training; 20% of the training participants are held out for validation.
Each test recording is one stream (``data.stream_by: [timeline]``). Its 40
test recordings are the Codabench warm-up evaluation set.

Requires a one-time download (``benchopt prepare``, ~1.1 GB). The
zero-dependency ``Simulated`` dataset covers no-network smoke testing.
"""

from benchopt import BaseDataset
from benchopt.config import get_data_path

# Hard requirement of the real-data path, imported at module level so
# benchopt reports the dataset as not-installed when it is missing.
import neuralbench  # noqa: F401

from benchmark_utils.data import get_device
from benchmark_utils.nb_task import download_study, load_task, require_prepared

TASK = "_sleep_onset_stream"


class Dataset(BaseDataset):

    name = "Interaxon2026Muse"

    requirements = [
        # neuroai stack from neuroai pull request 301 (see requirements.txt);
        # all four are git-pinned so pip does not mix released sub-deps.
        "pip::neuralset @ git+https://github.com/facebookresearch/neuroai.git@refs/pull/301/head#subdirectory=neuralset-repo",  # noqa: E501
        "pip::neuralfetch @ git+https://github.com/facebookresearch/neuroai.git@refs/pull/301/head#subdirectory=neuralfetch-repo",  # noqa: E501
        "pip::neuraltrain @ git+https://github.com/facebookresearch/neuroai.git@refs/pull/301/head#subdirectory=neuraltrain-repo",  # noqa: E501
        "pip::neuralbench @ git+https://github.com/facebookresearch/neuroai.git@refs/pull/301/head#subdirectory=neuralbench-repo",  # noqa: E501
        # NeuralFetch downloads the Muse release from NEMAR through it.
        "pip::nemar-py>=0.3.1", "pip::mne", "scikit-learn",
    ]

    parameters = {
        "batch_size": [64],
        # Dataloader workers; 0 extracts windows in-process, which is what
        # shared CI/platform runners want. Raise it from the phase config
        # (``Interaxon2026Muse[num_workers=4]``) on a worker with spare
        # cores.
        "num_workers": [0],
        # "test" restricts the study to its test split (worker staging /
        # evaluation-only runs; incompatible with the objective's
        # training=True) — see benchmark_utils.nb_task.
        "subset": ["full"],
    }

    # Ignore loader config for prepare cache key.
    prepare_cache_ignore = ("batch_size", "num_workers")

    def prepare(self):
        # Download the study, then run the pipeline once: the extraction
        # (segmenting, targets) caches next to the data, so runs only touch
        # warm caches. Both steps are idempotent.
        download_study("eeg", TASK, self._data_dir())
        self._load()

    def _data_dir(self):
        path = get_data_path("neural_compet")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self, device="cpu"):
        return load_task(
            "eeg", TASK,
            data_dir=self._data_dir(),
            device=device,
            batch_size=self.batch_size,
            seed=self.get_seed(),
            num_workers=self.num_workers,
            subset=self.subset,
        )

    def get_data(self):
        # Load already-prepared data only; downloading + extracting is the
        # explicit ``prepare`` step (``benchopt prepare`` / ``--prepare``).
        require_prepared("eeg", TASK, self._data_dir())
        loaders, meta = self._load(device=get_device())
        return dict(
            # subset="test" stages the evaluation split only: no train
            # split to hand over, which is how a run learns it cannot train.
            train_loader=None if self.subset == "test" else loaders["train"],
            test_loader=loaders["test"],
            **{k: v for k, v in meta.items()
               if k not in ("target_shape", "raw_target_shape")},
        )
