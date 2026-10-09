"""Zero-(EEG-)dependency simulated dataset — the always-available smoke test.

Stands in for the real BCI-decoding task so any model can be tested without
downloads: epoched windows whose channel means are shifted by a per-class
template, one label per window.

Only depends on numpy + torch (the benchmark's base stack), so it needs no
EEG packages and powers ``benchopt run tracks/bci_decoding -d Simulated``
and ``benchopt test``.
"""

import numpy as np
from benchopt import BaseDataset

from benchmark_utils.data import (
    STANDARD_EEG_CHANNELS,
    chs_info_from_names,
    get_device,
    make_loader,
)


class Dataset(BaseDataset):

    name = "Simulated"

    requirements = []

    parameters = {
        "n_chans, n_times": [(8, 200)],
        "n_classes": [3],
        "n_train, n_test": [(120, 60)],
        "sfreq": [100.0],
    }

    test_parameters = {
        "n_chans, n_times": [(4, 80)],
        "n_classes": [3],
        "n_train, n_test": [(20, 10)],
        "sfreq": [100.0],
    }

    def _make_windows(self, rng, n, templates):
        X = np.empty((n, self.n_chans, self.n_times), dtype=np.float32)
        y = rng.integers(0, self.n_classes, size=n)
        for i, k in enumerate(y):
            mean = templates[k][:, None]
            X[i] = mean + rng.standard_normal((self.n_chans, self.n_times))
        return X, y.astype(np.int64)

    def prepare(self):
        # No downloads, so nothing to prepare.
        pass

    def get_data(self):
        rng = np.random.default_rng(self.get_seed())
        # One channel-mean template per class; scaled so a linear model can
        # separate the classes above the noise floor.
        templates = rng.standard_normal((self.n_classes, self.n_chans)) * 2.0
        device = get_device()
        if self.n_chans > len(STANDARD_EEG_CHANNELS):
            raise ValueError("simulated EEG supports at most 63 channels")
        ch_names = list(STANDARD_EEG_CHANNELS[:self.n_chans])

        X_tr, y_tr = self._make_windows(rng, self.n_train, templates)
        X_te, y_te = self._make_windows(rng, self.n_test, templates)
        # Test windows in streams as the real task scores them: a few
        # sessions, each of two runs in recording order (see
        # benchmark_utils/streaming.py).
        n_runs = min(6, self.n_test)
        record_id = np.arange(self.n_test) * n_runs // self.n_test
        first = np.searchsorted(record_id, record_id)
        onset = np.arange(self.n_test) - first

        return dict(
            train_loader=make_loader(X_tr, y_tr, shuffle=True, device=device),
            test_loader=make_loader(X_te, y_te, record_id=record_id,
                                    onset=onset, stream_id=record_id // 2,
                                    device=device),
            n_classes=self.n_classes,
            sfreq=self.sfreq,
            ch_names=ch_names,
            chs_info=chs_info_from_names(ch_names),
            n_chans=self.n_chans,
            n_times=self.n_times,
            device=device,
        )
