"""BCI-decoding studies through NeuralBench's streamed Track 2 task.

Wraps the neuralbench ``eeg/_motor_imagery_stream`` task config — see
``benchmark_utils.nb_task``: one 4-s window per cue (from 1 s after the cue
on PROTEUS, from the cue on the two proxies), one-hot labels, the signal
notch-filtered at 50 and 60 Hz and band-pass filtered at 0.1-75 Hz at its
recorded rate, then resampled to 120 Hz, in microvolts with no scaling or
clamping, and a test split streamed one session of one participant at a
time (``data.stream_by: [subject, session]``), its runs in recording order.
The ``study`` parameter picks the dataset variant:

- ``dreyer2026proteus`` : Dreyer2026Proteus — the PROTEUS training release
                          (NEMAR nm000290 v1.0.0: 41 EEG channels, three
                          cued commands, Graz and BrainHero interfaces), the
                          task default and the warm-up evaluation study, on
                          the task's subject-level split; the default.
- ``dreyer2023``        : Dreyer2023Large — a two-class public proxy
                          (predefined train/test subjects).
- ``tangermann2012``    : BNCI2014_001 — small, handy for real-data smoke
                          tests.

On PROTEUS, each window's ``info`` also gives its context, the interface of
its run (Graz or BrainHero), which the objective scores separately within
each session.

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
from benchmark_utils.nb_task import download_study, load_task, require_prepared

TASK = "_motor_imagery_stream"

# Dataset variant: an overlay yaml in the task's ``datasets/`` folder
# (None = the task default).
_OVERLAYS = {
    "dreyer2026proteus": None,
    "dreyer2023": "dreyer2023",
    "tangermann2012": "tangermann2012",
}

# PROTEUS runs by BIDS task -> the interface they used, the context of the
# subject x session x context cells. Baseline runs hold no cue, so no window.
_PROTEUS_INTERFACES = {
    "AcquisitionBH": "BrainHero",
    "OnlineRawBH": "BrainHero",
    "AcquisitionGraz": "Graz",
    "OnlineRawGraz": "Graz",
}


def _proteus_interface(trigger):
    """Interface (Graz or BrainHero) of the run a PROTEUS window comes from."""
    task = trigger.get("task")
    if task not in _PROTEUS_INTERFACES:
        raise ValueError(f"PROTEUS run with unknown task {task!r}: its "
                         "interface, the window's context, is unknown.")
    return _PROTEUS_INTERFACES[task]


# Context of each window, for the studies that define one.
_CONTEXTS = {"dreyer2026proteus": _proteus_interface}


class Dataset(BaseDataset):

    name = "BCI"

    requirements = [
        # neuroai stack from neuroai pull request 301 (see requirements.txt);
        # all four are git-pinned so pip does not mix released sub-deps.
        "pip::neuralset @ git+https://github.com/facebookresearch/neuroai.git@refs/pull/301/head#subdirectory=neuralset-repo",  # noqa: E501
        "pip::neuralfetch @ git+https://github.com/facebookresearch/neuroai.git@refs/pull/301/head#subdirectory=neuralfetch-repo",  # noqa: E501
        "pip::neuraltrain @ git+https://github.com/facebookresearch/neuroai.git@refs/pull/301/head#subdirectory=neuraltrain-repo",  # noqa: E501
        "pip::neuralbench @ git+https://github.com/facebookresearch/neuroai.git@refs/pull/301/head#subdirectory=neuralbench-repo",  # noqa: E501
        # The PROTEUS release downloads from NEMAR through NeuralFetch; the
        # two proxies through MOABB.
        "pip::nemar-py>=0.3.1", "pip::moabb", "pip::mne", "scikit-learn",
    ]

    parameters = {
        "study": ["dreyer2026proteus"],
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
            "eeg", TASK, self._data_dir(),
            dataset=_OVERLAYS[self.study],
        )
        self._load()

    def _data_dir(self):
        path = get_data_path("neural_compet")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self, device="cpu"):
        return load_task(
            "eeg", TASK,
            data_dir=self._data_dir(),
            dataset=_OVERLAYS[self.study],
            device=device,
            batch_size=self.batch_size,
            seed=self.get_seed(),
            num_workers=self.num_workers,
            subset=self.subset,
            # One-hot ``(K,)`` -> integer class label.
            target_transform=lambda y: y.argmax(-1),
            context_of=_CONTEXTS.get(self.study),
        )

    def get_data(self):
        # Load already-prepared data only; downloading + extracting is the
        # explicit ``prepare`` step (``benchopt prepare`` / ``--prepare``).
        require_prepared("eeg", TASK, self._data_dir(),
                         dataset=_OVERLAYS[self.study])
        loaders, meta = self._load(device=get_device())
        return dict(
            # subset="test" stages the evaluation split only: no train
            # split to hand over, which is how a run learns it cannot train.
            train_loader=None if self.subset == "test" else loaders["train"],
            test_loader=loaders["test"],
            n_classes=int(meta.pop("raw_target_shape")[-1]),
            **{k: v for k, v in meta.items() if k != "target_shape"},
        )
