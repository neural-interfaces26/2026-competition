"""Data loading through neuralbench task configs (the official pipelines).

Built on :func:`neuralbench.experiment_config.merge_task_config` for config
composition (defaults <- task <- dataset overlay) and
:class:`neuralbench.data.Data` for the loaders::

    loaders, meta = load_task(
        "eeg", "motor_imagery", data_dir=..., dataset="tangermann2012",
    )

Differences with running neuralbench itself:

- everything runs locally with an explicit ``data_dir`` (the study path and
  the exca infra are overridden — the ``~/.neuralbench`` config resolves in
  the background but none of its paths are used);
- the returned loaders follow the competition contract (``(X, y, info)``
  torch batches moved onto ``device``, see ``compet_core.data``), so nothing
  downstream is tied to neuralset/neuralbench types;
- ``subset="test"`` restricts the pipeline to the test split (see
  :func:`build_test_only_filter`) so workers can stage evaluation data only.

Like the rest of the neuro stack, this module is import-heavy; import it
only from ``datasets/`` modules (never from solvers).
"""

from pathlib import Path

import numpy as np
import torch

from compet_core.data import to_numpy


def _task_data_config(modality, task, dataset, data_dir, overrides):
    """The task's merged ``data`` config, localized: everything runs from
    ``data_dir`` with no exca cluster, caller overrides applied last."""
    from neuralbench.experiment_config import merge_task_config

    data_dir = Path(data_dir)
    cfg = merge_task_config(modality, task, dataset)["data"]
    cfg.update({
        "study.source.path": str(data_dir),
        # keep every cache next to the data (the studies' timeline loaders
        # need a folder for their exca Cached backend).
        "study.source.infra.folder": str(data_dir / "cache"),
        "neuro.infra.cluster": None,
        "neuro.infra.folder": str(data_dir / "cache"),
    })
    cfg.update(overrides or {})
    return cfg


def build_test_only_filter(data_cfg):
    """``filter_stimuli`` override restricting a study to its test split.

    Composes a :class:`neuralset.events.transforms.QueryEvents` query that
    keeps non-trigger events untouched and selects trigger events with the
    split's ``test_split_query`` when the config defines one, or on the
    ``split`` column otherwise (predefined splits shipped with the source
    data). Any ``filter_stimuli`` already present in the config is preserved
    (queries are and-composed).

    Only expressible for ``PredefinedSplit`` — runtime splits (e.g.
    ``SklearnSplit`` by subject hash) are not a stimuli query; asking for
    ``subset="test"`` on such a study raises.
    """
    split = dict(data_cfg.get("study", {}).get("split") or {})
    if split.get("name") != "PredefinedSplit":
        raise ValueError(
            "subset='test' requires a PredefinedSplit (an explicit "
            f"test_split_query or split column); got {split.get('name')!r}."
        )
    selector = split.get("test_split_query")
    if not selector:
        selector = f"{split.get('col_name', 'split')} == 'test'"

    trigger = data_cfg["trigger_event_type"]
    triggers = [trigger] if isinstance(trigger, str) else list(trigger)
    query = f"type not in {triggers!r} or ({selector})"

    existing = dict(data_cfg.get("study", {}).get("filter_stimuli") or {})
    if existing.get("query"):
        query = f"({existing['query']}) and ({query})"
    return {"name": "QueryEvents", "query": query}


def _channel_names(seg_ds):
    """Ordered channel names of a prepared ``SegmentDataset``'s extractor.

    The extracted signal is a bare tensor, but the neuro extractor keeps a
    name -> column-index map assigned during ``prepare()``. There is no
    public accessor (neuralbench's ``Data.prepare`` reads the same private
    attribute); returns ``None`` when unavailable so datasets degrade
    gracefully (``chs_info`` stays ``None``).
    """
    chans = getattr(seg_ds.extractors["neuro"], "_channels", None)
    if not chans:
        return None
    return [name for name, _ in sorted(chans.items(), key=lambda kv: kv[1])]


def download_study(modality, task, data_dir, dataset=None):
    """One-time download of a task's study data (``Dataset.prepare``).

    Skipped when ``data_dir`` is not writable: on the competition workers
    the staged data is mounted read-only, and even a fully-cached
    ``Study.download()`` ends with a ``chmod`` that would crash there.
    """
    import os

    if not os.access(data_dir, os.W_OK):
        print(f"[compet] {data_dir} is read-only — skipping download "
              "(data assumed staged).")
        return

    import neuralset as ns

    cfg = _task_data_config(modality, task, dataset, data_dir, None)
    study = dict(cfg["study"]["source"])
    study["path"] = Path(study["path"]) / study["name"]
    ns.Study(**study).download()


class _NBWindows(torch.utils.data.Dataset):
    """Adapt a prepared neuralset ``SegmentDataset`` split to ``(X, y, info)``.

    ``seg_ds[i]`` yields a neuralset ``Batch`` whose ``data`` holds the
    ``neuro`` signal, the ``target`` and the ``subject_id``; this wrapper
    converts them to the competition contract (plain torch tensors).
    """

    def __init__(self, seg_ds, target_transform=None):
        self.seg_ds = seg_ds
        self.target_transform = target_transform

    def __len__(self):
        return len(self.seg_ds)

    def __getitem__(self, i):
        data = self.seg_ds[i].data
        X = torch.as_tensor(to_numpy(data["neuro"]), dtype=torch.float32)
        if X.ndim == 3:                 # (1, C, T) -> (C, T)
            X = X[0]
        y = torch.as_tensor(to_numpy(data["target"]))
        if y.ndim and y.shape[0] == 1:  # (1, ...) -> (...)
            y = y[0]
        if self.target_transform is not None:
            y = self.target_transform(y)
        y = y.to(torch.float32 if y.is_floating_point() else torch.long)
        subject = data.get("subject_id")
        info = {"subject_id": int(np.asarray(to_numpy(subject)).reshape(-1)[0])
                if subject is not None else -1}
        return X, y, info


def _make_loaders(loaders, device, target_transform):
    """Rebuild the competition loaders from the neuralbench ones.

    Keeps neuralbench's per-split datasets and train sampler (e.g. the
    ``RegressionBinSampler``) but swaps the collate for the competition one
    (plain tensors moved onto ``device``).
    """
    from torch.utils.data import DataLoader, default_collate

    def collate(batch):
        X, y, info = default_collate(batch)
        return X.to(device), y.to(device), info

    out = {}
    for split, loader in loaders.items():
        ds = _NBWindows(loader.dataset, target_transform)
        out[split] = DataLoader(
            ds, batch_size=loader.batch_size, collate_fn=collate,
            sampler=getattr(loader, "sampler", None)
            if split == "train" else None,
            shuffle=(split == "train"
                     and getattr(loader, "sampler", None) is None),
        )
    return out


def load_task(modality, task, *, data_dir, dataset=None, device="cpu",
              batch_size=64, seed=0, overrides=None, target_transform=None,
              subset="full"):
    """Build the competition loaders + meta from a neuralbench task config.

    Parameters
    ----------
    modality : str
        neuralbench task family (``"eeg"`` or ``"emg"``).
    task : str
        Task name (e.g. ``"motor_imagery"``, ``"sleep_onset"``, ``"image"``).
    data_dir : path-like
        Root directory holding the downloaded studies (benchopt data path).
    dataset : str or None
        Optional dataset overlay from the task's ``datasets/`` folder (e.g.
        ``"tangermann2012"``); ``None`` uses the task's default study.
    device : str
        Device the batches are moved onto (see ``compet_core.data``).
    batch_size, seed : int
        Dataloader settings.
    overrides : dict or None
        Extra ``data:``-section overrides, as dotted keys (e.g.
        ``{"study.source.query": ...}``).
    target_transform : callable or None
        Applied to each window's target (e.g. one-hot -> class index).
    subset : {"full", "test"}
        ``"test"`` restricts the study to its test split via a
        ``filter_stimuli`` override (:func:`build_test_only_filter`) —
        e.g. to stage only the evaluation data on a worker. The
        train/val loaders are then (near-)empty; only use the test one.

    Returns
    -------
    loaders : dict
        ``{"train", "val", "test"}`` -> competition-contract DataLoaders.
    meta : dict
        ``sfreq, ch_names, chs_info, n_chans, n_times, device`` (+ target
        shape info under ``target_shape`` / ``raw_target_shape``).
    """
    from neuralbench.data import Data

    from compet_core.data import chs_info_from_names

    cfg = _task_data_config(modality, task, dataset, data_dir, overrides)
    # single-process loaders: the defaults inject num_workers=N_CPUS, which
    # over-subscribes platform/CI runners (our rewrapped loaders extract
    # windows lazily in-process anyway).
    cfg.update({"batch_size": batch_size, "seed": seed, "num_workers": 0,
                "pin_memory": False, "persistent_workers": False})
    if subset == "test":
        cfg["study.filter_stimuli"] = build_test_only_filter(cfg)
    elif subset != "full":
        raise ValueError(f"subset must be 'full' or 'test', got {subset!r}")

    nb_loaders = Data(**cfg).prepare()
    loaders = _make_loaders(nb_loaders, device, target_transform)

    # Peek one window (lazy) for shapes; channel names come from the prepared
    # neuro extractor's channel map.
    X0, y0, _ = loaders["test"].dataset[0]
    raw_y0 = to_numpy(nb_loaders["test"].dataset[0].data["target"])
    if raw_y0.ndim and raw_y0.shape[0] == 1:
        raw_y0 = raw_y0[0]
    ch_names = _channel_names(nb_loaders["test"].dataset)
    neuro = nb_loaders["test"].dataset.extractors["neuro"]
    meta = dict(
        sfreq=float(neuro.frequency),
        ch_names=ch_names,
        chs_info=chs_info_from_names(ch_names),
        n_chans=int(X0.shape[-2]),
        n_times=int(X0.shape[-1]),
        target_shape=tuple(y0.shape),
        # Shape before ``target_transform`` (e.g. one-hot width -> n_classes).
        raw_target_shape=tuple(raw_y0.shape),
        device=device,
    )
    return loaders, meta
