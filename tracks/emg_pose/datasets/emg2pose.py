"""EMG-to-pose task on Salter2024Emg2pose, the official corpus.

Wraps the neuralbench ``emg/pose`` task config — see
``benchmark_utils.nb_task``: 5-s windows of 16-channel wrist surface EMG at
2 kHz, target = 20 hand joint-angle trajectories (``EmgExtractor`` MISC
picks), with the corpus' predefined train/val/test split.

Requires a one-time full-study download (``benchopt prepare`` — large; prefer
a compute node). The zero-dependency ``Simulated`` dataset covers no-network
smoke testing.
"""

from benchopt import BaseDataset
from benchopt.config import get_data_path

# Hard requirement of the real-data path, imported at module level so
# benchopt reports the dataset as not-installed when it is missing.
import neuralbench  # noqa: F401

from benchmark_utils.data import get_device
from benchmark_utils.nb_task import download_study, load_task


class Dataset(BaseDataset):

    name = "Salter2024Emg2pose"

    requirements = [
        "pip::neuralset", "pip::neuralfetch", "pip::neuralbench",
        "pip::mne", "scikit-learn", "pip::torch",
    ]

    parameters = {
        "batch_size": [64],
        # Dataloader workers; 0 extracts windows in-process, which is what
        # shared CI/platform runners want. Raise it from the phase config
        # (``Salter2024Emg2pose[num_workers=4]``) on a worker with spare cores.
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
        # (filtering, segmenting, targets) caches next to the data, so runs
        # only touch warm caches. Both steps are idempotent.
        download_study("emg", "pose", self._data_dir())
        self._load()

    def _data_dir(self):
        path = get_data_path("neural_compet")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self, device="cpu"):
        return load_task(
            "emg", "pose",
            data_dir=self._data_dir(),
            device=device,
            batch_size=self.batch_size,
            seed=self.get_seed(),
            num_workers=self.num_workers,
            subset=self.subset,
            # Cache the pose-target extraction next to the data; without a
            # folder its exca cache defaults to the container-local /tmp, which
            # --rm drops, so the replay (and every submission) rebuilds it.
            overrides={
                "target.infra.cluster": None,
                "target.infra.folder": str(self._data_dir() / "cache"),
                "target.infra.permissions": None,
                # Serialize the timeline build: concurrent
                # workers append to the TimelineLoader cachedict while others
                # read it, making exca's "non-last line" jsonl guard fail.
                # Switch to inline Cached backend.
                "study.source.timelines.infra.backend": "Cached",
                "study.source.timelines.infra.folder": str(
                    self._data_dir() / "cache"
                ),
            },
        )

    def get_data(self):
        self.prepare()  # idempotent — so plain ``benchopt run`` also works
        loaders, meta = self._load(device=get_device())
        return dict(
            # subset="test" stages the evaluation split only: no train
            # split to hand over, which is how a run learns it cannot train.
            train_loader=None if self.subset == "test" else loaders["train"],
            test_loader=loaders["test"],
            # Target windows are (n_joints, T); the objective needs n_joints.
            n_joints=int(meta["target_shape"][0]),
            **{k: v for k, v in meta.items()
               if k not in ("target_shape", "raw_target_shape")},
        )
