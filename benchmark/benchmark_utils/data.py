"""Shared data utilities for the EEG competition benchmark.

The benchmark passes **lazy PyTorch dataloaders** between components (not
materialized arrays) because EEG/PSG recordings can be large. Every loader
yields ``(X, y, info)`` batches:

- ``X``    : float32 tensor ``(B, C, T)`` — windows of ``C`` channels.
- ``y``    : labels. *epoched* tasks → ``(B,)`` one label per window;
             *dense* tasks → ``(B, T)`` one label per time-step.
- ``info`` : dict with ``record_id`` and ``onset`` (sample index of the
             window start in its source recording) so dense metrics can
             reassemble a recording's prediction sequence.

Tensors stay as **torch tensors** end-to-end (so braindecode foundation
models receive tensors directly); conversion to numpy happens only at the
scikit-learn boundaries (the linear head and the metrics), via
:func:`to_numpy`.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def to_numpy(x):
    """Convert a torch tensor (or array-like) to a numpy array.

    Used only at the scikit-learn boundaries — keep torch elsewhere.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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
            torch.arange(new_len) * (length / new_len)
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
    y : array-like or tensor, ``(N,)`` (epoched) or ``(N, T)`` (dense)
    record_id : array-like ``(N,)`` or None
        Source-recording id per window (defaults to all zeros).
    onset : array-like ``(N,)`` or None
        Window start sample in its recording (defaults to the index).
    """

    def __init__(self, X, y, record_id=None, onset=None):
        self.X = torch.as_tensor(to_numpy(X), dtype=torch.float32)
        self.y = torch.as_tensor(to_numpy(y), dtype=torch.long)
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
                onset=None):
    """Wrap arrays/tensors in ``ArrayWindows`` + a torch ``DataLoader``.

    The default collate batches ``X``/``y`` into tensors and turns the
    ``info`` dict into a dict of tensors.
    """
    dataset = ArrayWindows(X, y, record_id=record_id, onset=onset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
