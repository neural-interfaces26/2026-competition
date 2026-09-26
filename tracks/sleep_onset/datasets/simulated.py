"""Zero-(EEG-)dependency simulated dataset — the always-available smoke test.

Stands in for the real sleep-onset task so any model can be tested without
downloads: each window's target is a latency ``y`` in ``[0, 600]`` s, and the
signal encodes it learnably — the channel means drift linearly with ``y``
(mimicking slowing activity as sleep approaches), so a linear readout on the
window means can regress the latency above the noise floor.

Only depends on numpy + torch (the benchmark's base stack), so it needs no
EEG packages and powers ``benchopt run tracks/sleep_onset -d Simulated``
and ``benchopt test``.
"""

import numpy as np
from benchopt import BaseDataset

from benchmark_utils.data import chs_info_from_names, get_device, make_loader

CAP_S = 600.0

# Standard 10-20 electrode names (see bci_decoding's Simulated).
STANDARD_1020 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]


class Dataset(BaseDataset):

    name = "Simulated"

    requirements = []

    parameters = {
        "n_chans, n_times": [(4, 500)],
        "n_train, n_test": [(300, 150)],
        "sfreq": [100.0],
    }

    test_parameters = {
        "n_chans, n_times": [(2, 100)],
        "n_train, n_test": [(20, 10)],
        "sfreq": [100.0],
    }

    def _make_windows(self, rng, n, slope):
        # Latencies skewed towards late onsets (like real recordings), with
        # the cap value present.
        y = np.minimum(rng.exponential(scale=250.0, size=n), CAP_S)
        X = rng.standard_normal(
            (n, self.n_chans, self.n_times)
        ).astype(np.float32)
        X += (slope[:, None] * (y[:, None, None] / CAP_S)).astype(np.float32)
        return X, y.astype(np.float32)

    def prepare(self):
        # No downloads, so nothing to prepare.
        pass

    def get_data(self):
        rng = np.random.default_rng(self.get_seed())
        # Per-channel drift direction; scaled above the noise floor.
        slope = rng.standard_normal(self.n_chans) * 4.0
        device = get_device()
        ch_names = STANDARD_1020[:self.n_chans]

        X_tr, y_tr = self._make_windows(rng, self.n_train, slope)
        X_te, y_te = self._make_windows(rng, self.n_test, slope)

        return dict(
            train_loader=make_loader(X_tr, y_tr, shuffle=True, device=device),
            test_loader=make_loader(X_te, y_te, device=device),
            sfreq=self.sfreq,
            ch_names=ch_names,
            chs_info=chs_info_from_names(ch_names),
            n_chans=self.n_chans,
            n_times=self.n_times,
            device=device,
        )
