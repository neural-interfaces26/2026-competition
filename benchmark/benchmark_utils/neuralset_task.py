"""Data-loading helpers built on neuralset / neuralfetch.

**Scope of the neuralset dependency.** neuralset/neuralfetch are used here
*only* for data loading — turning an external EEG study into windowed
``(X, y)`` data. The signal ``X`` is kept as a torch tensor (so braindecode
models get tensors); labels are numpy. Do not import neuralset outside this
module and the ``datasets/`` that call it, so the rest of the benchmark stays
framework-agnostic.

Pipeline (see brainai design docs ``studies.md`` / ``dataloader-pipeline``)::

    Study(path, query).run()      -> events DataFrame
    Segmenter(...).apply(events)  -> SegmentDataset
    ds.prepare(); ds.load_all()   -> Batch(data={"eeg": (N, C, T), ...})

- **epoched** tasks trigger on trial events and read one label per window
  with a ``LabelEncoder`` extractor.
- **dense** tasks trigger sliding windows over the continuous recording; the
  per-step target is rebuilt from the annotation events (kept independent of
  neuralset's labelling extractors, to stay loosely coupled).
"""

import numpy as np
import torch
from neuralset import dataloader as dl


def _eeg_extractor(frequency, picks):
    return {"name": "EegExtractor", "frequency": float(frequency),
            "picks": list(picks)}


def _segment_meta(segments, frequency):
    """Per-window ``record_id`` (timeline) and ``onset`` (sample index)."""
    rec_ids, onsets = [], []
    timeline_to_int = {}
    for seg in segments:
        tl = getattr(seg, "timeline", None)
        if tl is None:
            tl = getattr(getattr(seg, "trigger", None), "timeline", 0)
        rid = timeline_to_int.setdefault(tl, len(timeline_to_int))
        rec_ids.append(rid)
        start = float(getattr(seg, "start", 0.0))
        onsets.append(int(round(start * frequency)))
    return (np.asarray(rec_ids, dtype=np.int64),
            np.asarray(onsets, dtype=np.int64))


def load_epoched(study, *, trigger_query, trial_event_type, duration,
                 frequency, label_field="description", picks=("eeg",)):
    """Return ``(X (N,C,T) tensor, y (N,), record_id, onset)`` (epoched)."""
    events = study.run()
    segmenter = dl.Segmenter(
        trigger_query=trigger_query,
        start=0.0,
        duration=float(duration),
        extractors={
            "eeg": _eeg_extractor(frequency, picks),
            "label": {
                "name": "LabelEncoder",
                "event_types": trial_event_type,
                "event_field": label_field,
                "aggregation": "trigger",
            },
        },
        drop_incomplete=True,
        drop_unused_events=True,
    )
    ds = segmenter.apply(events)
    ds.prepare()
    batch = ds.load_all()

    X = torch.as_tensor(batch.data["eeg"], dtype=torch.float32)  # (N, C, T)
    y = batch.data["label"]
    y = (y.detach().cpu().numpy() if hasattr(y, "detach")
         else np.asarray(y)).reshape(-1).astype(np.int64)
    record_id, onset = _segment_meta(batch.segments, frequency)
    return X, y, record_id, onset


def _dense_targets(events, segments, frequency, window_samples, *,
                   event_type, field, mapping, background):
    """Rebuild per-time-step labels for dense windows from events.

    Every sample is labelled with the class of the annotation event (e.g. a
    sleep stage) covering that time, or ``background`` if none. Kept
    independent of neuralset's labelling extractors on purpose.
    """
    ev = events[events["type"] == event_type].copy()
    ev["cls"] = ev[field].map(mapping)
    ev = ev.dropna(subset=["cls"])

    y = np.full((len(segments), window_samples), background, dtype=np.int64)
    for i, seg in enumerate(segments):
        tl = getattr(seg, "timeline", None)
        w_start = float(getattr(seg, "start", 0.0))
        rows = ev[ev["timeline"] == tl] if tl is not None else ev
        for r in rows.itertuples():
            a = int(round((r.start - w_start) * frequency))
            b = int(round((r.start + r.duration - w_start) * frequency))
            a, b = max(a, 0), min(b, window_samples)
            if b > a:
                y[i, a:b] = int(r.cls)
    return y


def load_dense(study, *, signal_event_type, window_s, frequency, event_type,
               field, mapping, background, picks=("eeg",)):
    """Return ``(X (N,C,T) tensor, y (N,T), record_id, onset)`` (dense).

    Non-overlapping sliding windows of ``window_s`` seconds are cut over the
    continuous recording; ``y`` is the per-sample class sequence.
    """
    events = study.run()
    segmenter = dl.Segmenter(
        trigger_query=f"type=='{signal_event_type}'",
        start=0.0,
        duration=float(window_s),
        stride=float(window_s),
        extractors={"eeg": _eeg_extractor(frequency, picks)},
        drop_incomplete=True,
        drop_unused_events=False,
    )
    ds = segmenter.apply(events)
    ds.prepare()
    batch = ds.load_all()

    X = torch.as_tensor(batch.data["eeg"], dtype=torch.float32)  # (N, C, T)
    record_id, onset = _segment_meta(batch.segments, frequency)
    y = _dense_targets(
        events, batch.segments, frequency, X.shape[-1],
        event_type=event_type, field=field, mapping=mapping,
        background=background,
    )
    return X, y, record_id, onset
