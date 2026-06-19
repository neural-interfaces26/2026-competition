"""Data-loading helpers built on neuralset / neuralfetch.

**Scope of the neuralset dependency.** neuralset/neuralfetch are used here
*only* for data loading — turning an external EEG study into windowed
``(X, y)`` data. The signal ``X`` is kept as a torch tensor (so braindecode
models get tensors); labels are numpy. Do not import neuralset outside this
module and the ``datasets/`` that call it, so the rest of the benchmark stays
framework-agnostic.

Pipeline (see brainai design docs ``studies.md`` / ``dataloader-pipeline``)::

    Study(path, query).run()      -> events DataFrame
    Segmenter(...).apply(events)  -> SegmentDataset (prepared, not loaded)

The ``SegmentDataset`` is itself a torch ``Dataset`` (``ds[i]`` lazily
extracts one window's signal), so it is returned **un-materialized** and
wrapped by :class:`~benchmark_utils.data.SegmentWindows`; the EEG signal is
never loaded into memory all at once. Each builder also returns the small
per-window metadata (labels, ``record_id``, ``onset``) needed to split the data
by recording (see :func:`~benchmark_utils.data.group_split`) and to reassemble
dense predictions.

- **epoched** tasks trigger on trial events; labels come from a cheap
  label-only ``LabelEncoder`` pass (no signal), aligned 1:1 with the signal
  segments.
- **dense** tasks trigger sliding windows over the continuous recording; the
  per-step target is rebuilt from the annotation events (kept independent of
  neuralset's labelling extractors, to stay loosely coupled).
"""

import numpy as np
from neuralset import dataloader as dl


def _eeg_extractor(frequency, picks):
    return {"name": "EegExtractor", "frequency": float(frequency),
            "picks": list(picks)}


def channel_names(ds, key="eeg"):
    """Ordered channel names of a *prepared* ``SegmentDataset``'s extractor.

    The EEG signal in ``ds[i].data[key]`` is already a bare tensor (the MNE
    header is dropped), but the extractor keeps a name->column-index map
    (``_channels``) assigned during ``prepare()``. Invert it to recover the
    channel names in tensor-row order — used to build ``chs_info`` for
    name-based foundation models (e.g. REVE).

    Returns ``None`` if the map is unavailable, so datasets degrade gracefully
    (``chs_info`` stays ``None`` rather than raising).
    """
    chans = getattr(ds.extractors[key], "_channels", None)
    if not chans:
        return None
    return [name for name, _ in sorted(chans.items(), key=lambda kv: kv[1])]


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


def build_epoched(study, *, trigger_query, trial_event_type, duration,
                  frequency, label_field="description", picks=("eeg",)):
    """Build a lazy epoched ``SegmentDataset`` + per-window labels/metadata.

    Returns ``(ds, y, record_id, onset)`` where ``ds`` is a *prepared*
    ``SegmentDataset`` that lazily extracts EEG per window, ``y`` the
    integer trial labels (one per window), and ``record_id`` / ``onset`` the
    source-recording id and window-start sample. The signal is **not**
    materialized here.
    """
    events = study.run()
    common = dict(
        trigger_query=trigger_query, start=0.0, duration=float(duration),
        drop_incomplete=True, drop_unused_events=True,
    )
    # Lazy signal dataset — extracted one window at a time by the dataloader.
    ds = dl.Segmenter(
        extractors={"eeg": _eeg_extractor(frequency, picks)}, **common
    ).apply(events)
    ds.prepare()

    # Cheap label-only pass (no signal), aligned 1:1 with ``ds`` segments.
    lds = dl.Segmenter(
        extractors={"label": {
            "name": "LabelEncoder",
            "event_types": trial_event_type,
            "event_field": label_field,
            "aggregation": "trigger",
        }},
        **common,
    ).apply(events)
    lds.prepare()
    y = lds.load_all().data["label"]
    y = (y.detach().cpu().numpy() if hasattr(y, "detach")
         else np.asarray(y)).reshape(-1)

    record_id, onset = _segment_meta(ds.segments, frequency)
    return ds, y, record_id, onset


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


def build_dense(study, *, signal_event_type, window_s, frequency, event_type,
                field, mapping, background, picks=("eeg",)):
    """Build a lazy dense ``SegmentDataset`` + per-step labels/metadata.

    Non-overlapping sliding windows of ``window_s`` seconds are cut over the
    continuous recording. Returns ``(ds, y, record_id, onset)`` with ``ds`` a
    *prepared* ``SegmentDataset`` (lazy signal) and ``y`` the ``(N, T)``
    per-sample class sequence rasterized from the annotation events. The signal
    is **not** materialized here.
    """
    events = study.run()
    ds = dl.Segmenter(
        trigger_query=f"type=='{signal_event_type}'",
        start=0.0,
        duration=float(window_s),
        stride=float(window_s),
        extractors={"eeg": _eeg_extractor(frequency, picks)},
        drop_incomplete=True,
        drop_unused_events=False,
    ).apply(events)
    ds.prepare()

    # Window length in samples — peek one window (cheap) for the exact T.
    window_samples = int(np.asarray(ds[0].data["eeg"]).shape[-1])
    record_id, onset = _segment_meta(ds.segments, frequency)
    y = _dense_targets(
        events, ds.segments, frequency, window_samples,
        event_type=event_type, field=field, mapping=mapping,
        background=background,
    )
    return ds, y, record_id, onset
