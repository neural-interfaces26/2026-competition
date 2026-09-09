"""Shared data utilities for the competition track benchmarks.

The tracks pass **lazy PyTorch dataloaders** between components (not
materialized arrays) because EEG/EMG recordings can be large. Every loader
yields ``(X, y, info)`` batches:

- ``X``    : float32 tensor ``(B, C, T)`` — windows of ``C`` channels.
- ``y``    : targets, track-specific: class label ``(B,)``, scalar
             regression target ``(B,)``, embedding ``(B, D)``, or per-step
             sequence ``(B, ..., T)``. Integer targets are ``long``, float
             targets ``float32``.
- ``info`` : dict with ``record_id`` and ``onset`` (sample index of the
             window start in its source recording) so metrics can trace a
             window back to its recording.

Tensors stay as **torch tensors** end-to-end (so torch models receive tensors
directly); conversion to numpy happens only at the scikit-learn boundaries
(linear heads and metrics), via :func:`to_numpy`.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate


def to_numpy(x):
    """Convert a torch tensor (or array-like) to a numpy array.

    Used only at the scikit-learn boundaries — keep torch elsewhere.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_device():
    """Device the benchmark runs on (``cuda`` when available, else ``cpu``).

    Auto-detects a GPU; override with ``COMPET_DEVICE`` (e.g. ``cpu`` to
    force CPU even on a GPU box, handy for debugging). Datasets call this once
    and (a) move every batch onto the device at loading time and (b) advertise
    it through ``meta["device"]`` so a model can place itself there in
    ``load_model``.
    """
    forced = os.environ.get("COMPET_DEVICE")
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


def chs_info_from_names(ch_names):
    """Minimal MNE-style ``chs_info`` — one ``{"ch_name": ...}`` per channel.

    Foundation models that place channels *by name* (e.g. braindecode REVE,
    which maps standard 10-20/10-10/10-05 electrode names to 3D coordinates)
    read ``ch["ch_name"]`` from this list to look up positions. ``None`` in →
    ``None`` out, so datasets without channel names stay explicit.
    """
    if ch_names is None:
        return None
    return [{"ch_name": str(n)} for n in ch_names]


def as_target(y):
    """Convert targets to a torch tensor with a metric-friendly dtype.

    Integer targets (class labels) become ``long``; floating targets
    (regression values, embeddings) become ``float32``.
    """
    y = torch.as_tensor(to_numpy(y))
    if y.is_floating_point():
        return y.to(torch.float32)
    return y.to(torch.long)


def _move_collate(device):
    """Default collate, then move the signal/labels onto ``device``.

    Done at *batch-loading* time so the encoder/model and the linear head all
    see ``X`` and ``y`` on the same device (no per-solver ``.to(device)`` and
    no silent device-mismatch fallback in the probe). ``info`` stays on CPU —
    it is only used by the numpy metric code.
    """
    def collate(batch):
        X, y, info = default_collate(batch)
        return X.to(device), y.to(device), info
    return collate


def resample_labels(seq, new_len):
    """Nearest-neighbour resample an int label sequence along the last axis.

    Aligns a per-time-step target/prediction to a different temporal
    resolution (e.g. the encoder's ``T'`` vs the raw ``T``). Type-preserving:
    torch in → torch out, numpy in → numpy out.

    Parameters
    ----------
    seq : array-like or tensor, shape ``(..., L)``
    new_len : int
    """
    length = seq.shape[-1]
    if length == new_len:
        return seq
    if isinstance(seq, torch.Tensor):
        idx = torch.floor(
            torch.arange(new_len, device=seq.device) * (length / new_len)
        ).long().clamp(0, length - 1)
        return seq[..., idx]
    seq = np.asarray(seq)
    idx = np.floor(np.arange(new_len) * (length / new_len)).astype(int)
    idx = np.clip(idx, 0, length - 1)
    return seq[..., idx]


class ArrayWindows(Dataset):
    """In-memory windows dataset yielding ``(X, y, info)`` torch tensors.

    Parameters
    ----------
    X : array-like or tensor, shape ``(N, C, T)``
    y : array-like or tensor, ``(N, ...)`` — track-specific target per window
        (int targets stored as ``long``, float targets as ``float32``).
    record_id : array-like ``(N,)`` or None
        Source-recording id per window (defaults to all zeros).
    onset : array-like ``(N,)`` or None
        Window start sample in its recording (defaults to the index).
    """

    def __init__(self, X, y, record_id=None, onset=None):
        self.X = torch.as_tensor(to_numpy(X), dtype=torch.float32)
        self.y = as_target(y)
        n = len(self.X)
        self.record_id = (
            np.zeros(n, dtype=np.int64) if record_id is None
            else np.asarray(record_id, dtype=np.int64)
        )
        self.onset = (
            np.arange(n, dtype=np.int64) if onset is None
            else np.asarray(onset, dtype=np.int64)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        info = {
            "record_id": int(self.record_id[i]),
            "onset": int(self.onset[i]),
        }
        return self.X[i], self.y[i], info


def make_loader(X, y, batch_size=32, shuffle=False, record_id=None,
                onset=None, device="cpu"):
    """Wrap arrays/tensors in ``ArrayWindows`` + a torch ``DataLoader``.

    The collate batches ``X``/``y`` into tensors (moved onto ``device``) and
    turns the ``info`` dict into a dict of tensors (kept on CPU).
    """
    dataset = ArrayWindows(X, y, record_id=record_id, onset=onset)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=_move_collate(device),
    )


class SegmentWindows(Dataset):
    """Lazy windows over a neuralset ``SegmentDataset``, yielding ``(X, y,
    info)``.

    Wraps a *prepared* neuralset ``SegmentDataset`` so the EEG signal is
    extracted one window at a time (kept off-memory) rather than materialized
    up front, while still conforming to the benchmark loader contract. Labels
    and per-window ``info`` are small and precomputed at construction.

    Parameters
    ----------
    seg_ds : neuralset SegmentDataset
        Prepared dataset whose ``seg_ds[i]`` yields a ``Batch`` with the EEG
        signal under ``eeg_key`` (shape ``(1, C, T)`` or ``(C, T)``).
    y : array-like, ``(N, ...)`` — track-specific target per window
    record_id, onset : array-like ``(N,)``
        Source-recording id and window start sample, aligned with ``seg_ds``.
    """

    def __init__(self, seg_ds, y, record_id, onset, eeg_key="eeg"):
        self.seg_ds = seg_ds
        self.eeg_key = eeg_key
        self.y = as_target(y)
        self.record_id = np.asarray(record_id, dtype=np.int64)
        self.onset = np.asarray(onset, dtype=np.int64)

    def __len__(self):
        return len(self.seg_ds)

    def __getitem__(self, i):
        x = torch.as_tensor(
            to_numpy(self.seg_ds[i].data[self.eeg_key]), dtype=torch.float32
        )
        if x.ndim == 3:                 # (1, C, T) -> (C, T)
            x = x[0]
        info = {
            "record_id": int(self.record_id[i]),
            "onset": int(self.onset[i]),
        }
        return x, self.y[i], info


def make_segment_loader(seg_ds, y, record_id, onset, batch_size=32,
                        shuffle=False, device="cpu"):
    """Wrap a neuralset ``SegmentDataset`` in ``SegmentWindows`` + a torch
    ``DataLoader`` (lazy signal extraction; batches moved onto ``device``)."""
    dataset = SegmentWindows(seg_ds, y, record_id, onset)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=_move_collate(device),
    )


def group_split(groups, test_size, seed):
    """Train/test split that keeps each source recording (group) intact.

    Splits at the *recording* level (``groups``, e.g. ``record_id`` derived
    from the study timeline) rather than randomly over windows, so windows from
    the same recording/session never leak across the train/test boundary —
    important when one subject/session yields many correlated windows.

    Returns ``(train_idx, test_idx)`` integer arrays.
    """
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.asarray(groups)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(groups, groups=groups))
    return train_idx, test_idx
