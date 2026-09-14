"""Data loading through neuralbench task configs (the official pipelines).

Builds :class:`neuralbench.data.Data` from the task configs shipped in the
``neuralbench`` wheel (``neuralbench/tasks/<modality>/<task>/config.yaml``),
so each track's benchopt dataset reuses the *exact* official data pipeline
(study, split, segmenter, target extractor, sampler) in a few lines::

    loaders, meta = load_task(
        "eeg", "motor_imagery", data_dir=..., dataset="tangermann2012",
    )

Differences with running neuralbench itself:

- no ``~/.neuralbench/config.json`` and no exca cluster infra — everything
  runs locally with explicit ``data_dir``;
- the returned loaders follow the competition contract (``(X, y, info)``
  torch batches moved onto ``device``, see ``compet_core.data``), so nothing
  downstream is tied to neuralset/neuralbench types.

Like the rest of the neuro stack, this module is import-heavy; import it
only from ``datasets/`` modules (never from solvers).
"""

from pathlib import Path

import numpy as np
import torch

from compet_core.data import to_numpy

# Mirrors ``neuralbench/defaults/config.yaml``'s ``data:`` section, minus the
# exca infra blocks (cluster caching) and the user-config paths, which we
# replace with explicit arguments.
_NEURO_DEFAULTS = {
    "name": "EegExtractor",
    "picks": ["eeg"],
    "frequency": 120.0,
    "filter": [0.1, 75.0],
    "notch_filter": [50.0, 60.0],
    "baseline": None,
    "scaler": "RobustScaler",
    "clamp": 20.0,
}


def _strip_markers(cfg):
    """Drop leftover ``=replace=`` markers (kept verbatim by ``ConfDict``
    when the updated key did not pre-exist) before pydantic validation."""
    for key in list(cfg):
        if key == "=replace=":
            del cfg[key]
        elif isinstance(cfg[key], dict):
            _strip_markers(cfg[key])


def _data_config(modality, task, dataset, data_dir, overrides):
    """Compose the ``data:`` config: defaults <- task <- dataset <- overrides.

    Reproduces ``neuralbench.experiment_config.prepare_task_configs``'s
    overlay chain for the ``data`` section only (ConfDict handles the
    ``=replace=`` markers used by the task configs).
    """
    from exca import ConfDict
    from neuralbench.registry import _resolve_task_dir, load_yaml_config

    task_dir = _resolve_task_dir(modality, task)

    cfg = ConfDict({
        "study": {"source": {
            "path": str(data_dir),
            # neuralset >= 0.3 requires a folder for the studies' cached
            # timeline loaders (exca ``Cached`` backend).
            "infra": {"backend": "Cached",
                      "folder": str(Path(data_dir) / "cache")},
        }},
        "neuro": dict(_NEURO_DEFAULTS),
        "channel_positions": {"n_spatial_dims": 3},
    })
    # ``safe=True`` skips the ``!!python/...`` tags (metrics/config-manager
    # lookups) that live outside the ``data:`` section anyway.
    task_cfg = load_yaml_config(task_dir / "config.yaml", safe=True)
    cfg.update(task_cfg["data"])
    if dataset is not None:
        ds_cfg = load_yaml_config(
            task_dir / "datasets" / f"{dataset}.yaml", safe=True
        )
        source = dict(cfg["study"]["source"])
        cfg.update(ds_cfg["data"])
        # ``=replace=`` may wipe the source; restore the defaults (path).
        for key, value in source.items():
            cfg["study"]["source"].setdefault(key, value)
    cfg.update(overrides or {})
    _strip_markers(cfg)
    return cfg


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

    cfg = _data_config(modality, task, dataset, data_dir, None)
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


def _make_loaders(data, loaders, device, target_transform):
    """Rebuild the competition loaders from ``Data.prepare()``'s output.

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
              batch_size=64, seed=0, overrides=None, target_transform=None):
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
        Extra ``data:``-section overrides (ConfDict dotted keys work, e.g.
        ``{"study.source.query": ...}``).
    target_transform : callable or None
        Applied to each window's target (e.g. one-hot -> class index).

    Returns
    -------
    loaders : dict
        ``{"train", "val", "test"}`` -> competition-contract DataLoaders.
    meta : dict
        ``sfreq, ch_names, chs_info, n_chans, n_times, device`` (+ target
        shape info under ``target_shape``).
    """
    from neuralbench.data import Data

    from compet_core.data import chs_info_from_names
    from compet_core.neuralset_task import channel_names

    cfg = _data_config(modality, task, dataset, data_dir, overrides)
    cfg.update({"batch_size": batch_size, "seed": seed,
                "pin_memory": False, "persistent_workers": False})
    data = Data(**cfg)
    nb_loaders = data.prepare()
    loaders = _make_loaders(data, nb_loaders, device, target_transform)

    # Peek one window (lazy) for shapes; channel names come from the prepared
    # neuro extractor's channel map.
    X0, y0, _ = loaders["test"].dataset[0]
    raw_y0 = to_numpy(nb_loaders["test"].dataset[0].data["target"])
    if raw_y0.ndim and raw_y0.shape[0] == 1:
        raw_y0 = raw_y0[0]
    ch_names = channel_names(nb_loaders["test"].dataset, key="neuro")
    meta = dict(
        sfreq=float(cfg["neuro"]["frequency"]),
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
