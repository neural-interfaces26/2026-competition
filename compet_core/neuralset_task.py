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
wrapped by :class:`~compet_core.data.SegmentWindows`; the EEG signal is
never loaded into memory all at once. The builder also returns the small
per-window metadata (labels, ``record_id``, ``onset``) needed to split the
data by recording (see :func:`~compet_core.data.group_split`).

Epoched windows trigger on trial events; labels come from a cheap label-only
``LabelEncoder`` pass (no signal), aligned 1:1 with the signal segments.

Note: new datasets should prefer the neuralbench-config path in
``compet_core.data`` (``neuralbench.data.Data``); this module remains for
studies addressed directly through neuralset/neuralfetch.
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
