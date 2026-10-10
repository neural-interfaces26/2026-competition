"""Streamed evaluation, shared by Tracks 02 and 03.

Mirrors NeuralBench's stream tasks (``data.stream_by`` with the
``ResetPerStream`` callback). The test windows of a stream (Track 02: one
session of one participant; Track 03: one recording) reach the model one at
a time (``B = 1``), in recording order. Each stream starts from a fresh copy
of the model as it was when evaluation began, whose optional
``reset_state()`` is called first. A model can adapt within a stream, but
nothing it changes while predicting (weights, buffers, attributes) carries
over to the next one.

The test loader gives every window's ``stream_id``, ``record_id`` and
``onset`` in its ``info`` (see ``benchmark_utils.data``): each stream's
windows must come contiguously, each recording's in time order, and this is
checked as the windows are read.
"""

import copy

import numpy as np

from benchmark_utils.data import to_numpy

STREAM_KEYS = ("stream_id", "record_id", "onset")


def fresh_model(model):
    """Copy ``model`` with ``copy.deepcopy`` and call its ``reset_state()``.

    ``reset_state`` is optional; models without one are only copied.
    """
    try:
        clone = copy.deepcopy(model)
    except Exception as err:
        raise TypeError(
            "Streamed evaluation runs each stream on a copy of the model "
            "returned by load_model, made with copy.deepcopy, which failed "
            f"on this model: {err!r}"
        ) from err
    reset_state = getattr(clone, "reset_state", None)
    if callable(reset_state):
        reset_state()
    return clone


def _columns(info, n):
    """The ``info`` entries of a batch of ``n`` windows, as 1-D arrays."""
    columns = {}
    for key, value in info.items():
        value = np.asarray(to_numpy(value)).reshape(-1)
        if len(value) == n:
            columns[key] = value
    missing = [key for key in STREAM_KEYS if key not in columns]
    if missing:
        raise KeyError(
            f"Streamed evaluation needs info{missing} for every window; got "
            f"keys {sorted(info)}."
        )
    return columns


def predict_streams(model, loader):
    """Predict every window of ``loader``, stream by stream, one at a time.

    Parameters
    ----------
    model : object
        The model returned by ``load_model``, exposing ``predict(X)``. It is
        never called itself: every stream runs on a fresh copy
        (:func:`fresh_model`).
    loader : iterable of ``(X, y, info)`` batches
        The test windows, each stream's contiguous and each recording's in
        time order. Batches of any size are split into single windows.

    Returns
    -------
    y_true, y_pred : arrays, shape ``(N, ...)``
        One row per window, in loader order; ``y_pred[i]`` is what
        ``predict`` returned for window ``i`` (its batch axis dropped).
    info : dict of arrays, shape ``(N,)``
        The windows' ``info`` (``stream_id``, ``record_id``, ``onset`` and
        any other per-window entry).
    """
    y_true, y_pred, infos = [], [], []
    current = None                  # the copy predicting the current stream
    stream = record = None
    finished_streams, finished_records = set(), set()
    last_onset = -np.inf
    for X, y, info in loader:
        columns = _columns(info, len(X))
        streams = columns["stream_id"]
        records = columns["record_id"]
        onsets = columns["onset"].astype(np.float64)
        for i in range(len(X)):
            if streams[i] != stream:
                if streams[i] in finished_streams:
                    raise ValueError(
                        f"Stream {streams[i]} resumes after another stream "
                        "started: each stream's windows must be contiguous."
                    )
                if stream is not None:
                    finished_streams.add(stream)
                stream = streams[i]
                current = None      # free the previous copy before the next
                current = fresh_model(model)
            if records[i] != record:
                if records[i] in finished_records:
                    raise ValueError(
                        f"Recording {records[i]} resumes after another "
                        "recording started: windows must come in time order."
                    )
                if record is not None:
                    finished_records.add(record)
                record, last_onset = records[i], -np.inf
            if onsets[i] < last_onset:
                raise ValueError(
                    f"Recording {record}: a window starting at {onsets[i]} "
                    f"comes after one starting at {last_onset}; windows "
                    "must come in time order."
                )
            last_onset = onsets[i]

            pred = np.asarray(to_numpy(current.predict(X[i:i + 1])))
            if pred.ndim == 0 or pred.shape[0] != 1:
                raise ValueError(
                    "predict(X) must return one prediction per window; got "
                    f"shape {pred.shape} for a batch of 1."
                )
            y_pred.append(pred[0])
            y_true.append(to_numpy(y[i]))
        infos.append(columns)

    if not y_true:
        raise ValueError("The test loader yielded no window.")
    info = {key: np.concatenate([c[key] for c in infos]) for key in infos[0]}
    return np.stack(y_true), np.stack(y_pred), info
