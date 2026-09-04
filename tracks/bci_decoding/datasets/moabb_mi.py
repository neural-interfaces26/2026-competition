"""Motor-imagery task from MOABB BNCI2014_001, via neuralfetch.

One label per trial window. Data loading goes through the neuralfetch
``Tangermann2012Review`` study (alias ``BNCI2014_001``) and the neuralset
segmenter, behind ``compet_core.neuralset_task`` so nothing downstream is
tied to neuralset. Small single-subject dataset, handy for real-data smoke
tests next to the main ``Stieger2021`` proxy.

Requires a one-time download (``benchopt prepare`` / ``tools/setup_data.py``);
the zero-dependency ``Simulated`` dataset covers no-network smoke testing.
"""

import numpy as np
from benchopt import BaseDataset
from benchopt.config import get_data_path
from neuralfetch.studies.moabb2025 import Tangermann2012Review
from sklearn.preprocessing import LabelEncoder

from compet_core.data import (
    chs_info_from_names, get_device, group_split, make_segment_loader,
)
from compet_core.neuralset_task import build_epoched, channel_names


class Dataset(BaseDataset):

    name = "MOABB-MI"

    requirements = [
        "pip::neuralset", "pip::neuralfetch", "pip::moabb", "pip::mne",
        "scikit-learn", "pip::torch",
    ]

    parameters = {
        # Keep the POC small: a single subject, split into train/test trials.
        "subject": [1],
        "frequency": [200.0],   # REVE's 200 Hz; encoder resamples otherwise
        "trial_s": [4.0],
        "test_size": [0.4],
    }

    test_parameters = {
        "subject": [1],
        "frequency": [100.0],
        "trial_s": [2.0],
        "test_size": [0.5],
    }

    def _study(self):
        path = get_data_path("compet_eeg")
        path.mkdir(parents=True, exist_ok=True)
        query = f'subject == "Tangermann2012Review/{self.subject}"'
        return Tangermann2012Review(path=str(path), query=query)

    def _ensure_prepared(self):
        # Idempotent: mne/moabb skip already-downloaded files.
        self._study().download()

    def prepare(self):
        self._ensure_prepared()

    def get_data(self):
        self._ensure_prepared()

        ds, y, record_id, onset = build_epoched(
            self._study(),
            trigger_query="type=='Stimulus'",
            trial_event_type="Stimulus",
            duration=self.trial_s,
            frequency=self.frequency,
            label_field="description",
        )

        # Encode class names to consecutive ints.
        y = LabelEncoder().fit_transform(y)
        n_classes = int(len(np.unique(y)))

        # Recording-level split (no run/session leaks across train/test).
        tr, te = group_split(record_id, self.test_size, self.get_seed())

        # Peek one window (lazy) for the channel/time dimensions; channel
        # names come from the extractor's channel map (see channel_names).
        sample = np.asarray(ds[0].data["eeg"])  # (1, C, T)
        ch_names = channel_names(ds)
        device = get_device()

        return dict(
            train_loader=make_segment_loader(
                ds.select(tr), y[tr], record_id[tr], onset[tr], shuffle=True,
                device=device,
            ),
            test_loader=make_segment_loader(
                ds.select(te), y[te], record_id[te], onset[te], device=device,
            ),
            n_classes=n_classes,
            sfreq=self.frequency,
            ch_names=ch_names,
            chs_info=chs_info_from_names(ch_names),
            n_chans=sample.shape[-2],
            n_times=sample.shape[-1],
            device=device,
        )
