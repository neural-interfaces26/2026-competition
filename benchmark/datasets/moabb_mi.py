"""Motor-imagery (epoched) task from MOABB BNCI2014_001, via neuralfetch.

One label per trial window — the *epoched* regime. Data loading goes through
the neuralfetch ``Tangermann2012Review`` study (alias ``BNCI2014_001``) and
the neuralset segmenter, behind ``benchmark_utils.neuralset_task`` so nothing
downstream is tied to neuralset.

Requires a one-time download (``benchopt prepare`` / ``tools/setup_data.py``);
the zero-dependency ``Simulated`` dataset covers no-network smoke testing.
"""

import numpy as np
from benchopt import BaseDataset
from benchopt.config import get_data_path
from neuralfetch.studies.moabb2025 import Tangermann2012Review
from sklearn.preprocessing import LabelEncoder

from benchmark_utils.data import make_loader
from benchmark_utils.neuralset_task import load_epoched


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
        query = f'subject == "Tangermann2012Review/{self.subject}"'
        return Tangermann2012Review(path=str(path), query=query)

    def _ensure_prepared(self):
        # Idempotent: mne/moabb skip already-downloaded files.
        self._study().download()

    def prepare(self):
        self._ensure_prepared()

    def get_data(self):
        self._ensure_prepared()

        X, y, record_id, onset = load_epoched(
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

        # Trial-level train/test split (reproducible).
        rng = np.random.default_rng(self.get_seed())
        idx = rng.permutation(len(X))
        n_test = int(round(self.test_size * len(X)))
        te, tr = idx[:n_test], idx[n_test:]

        return dict(
            train_loader=make_loader(
                X[tr], y[tr], shuffle=True,
                record_id=record_id[tr], onset=onset[tr],
            ),
            test_loader=make_loader(
                X[te], y[te], record_id=record_id[te], onset=onset[te],
            ),
            task="mi",
            task_kind="epoched",
            metrics=["accuracy", "balanced_accuracy"],
            n_classes=n_classes,
            sfreq=self.frequency,
            ch_names=None,
            chs_info=None,
            n_chans=X.shape[1],
            n_times=X.shape[2],
        )
