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

from benchmark_utils.data import chs_info_from_names, get_device, make_loader

# Standard 10-20 electrode names, so name-based models (e.g. REVE) can
# resolve channel positions even on the synthetic dataset. The first
# ``n_chans`` are used.
STANDARD_1020 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]


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
        ch_names = STANDARD_1020[:self.n_chans]
        ch_names += [
            f"EEG{i:03d}" for i in range(len(ch_names), self.n_chans)
        ]

        X_tr, y_tr = self._make_windows(rng, self.n_train, templates)
        X_te, y_te = self._make_windows(rng, self.n_test, templates)

        return dict(
            train_loader=make_loader(X_tr, y_tr, shuffle=True, device=device),
            test_loader=make_loader(X_te, y_te, device=device),
            n_classes=self.n_classes,
            sfreq=self.sfreq,
            ch_names=ch_names,
            chs_info=chs_info_from_names(ch_names),
            n_chans=self.n_chans,
            n_times=self.n_times,
            device=device,
        )
