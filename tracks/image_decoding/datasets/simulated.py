"""Zero-(EEG-)dependency simulated dataset — the always-available smoke test.

Stands in for the real image-decoding task so any model can be tested without
downloads: ``n_images`` synthetic "images" each get a random embedding
``(D,)`` and a signal template; a window is its image's template + noise and
its target is the image's embedding — so a linear readout on the window means
can regress the embedding and retrieval is learnable.

Only depends on numpy + torch (the benchmark's base stack), so it needs no
EEG packages and powers ``benchopt run tracks/image_decoding -d Simulated``
and ``benchopt test``.
"""

import numpy as np
from benchopt import BaseDataset

from benchmark_utils.data import chs_info_from_names, get_device, make_loader

# Standard 10-20 electrode names (see bci_decoding's Simulated).
STANDARD_1020 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]


class Dataset(BaseDataset):

    name = "Simulated"

    requirements = []

    parameters = {
        "n_chans, n_times": [(8, 120)],
        "n_images": [20],
        "n_outputs": [32],
        "n_train, n_test": [(300, 120)],
        "sfreq": [100.0],
    }

    test_parameters = {
        "n_chans, n_times": [(4, 60)],
        "n_images": [8],
        "n_outputs": [16],
        "n_train, n_test": [(20, 10)],
        "sfreq": [100.0],
    }

    def _make_windows(self, rng, n, templates, embeddings):
        idx = rng.integers(0, self.n_images, size=n)
        X = (templates[idx]
             + rng.standard_normal((n, self.n_chans, self.n_times))
             ).astype(np.float32)
        return X, embeddings[idx].astype(np.float32)

    def prepare(self):
        # No downloads, so nothing to prepare.
        pass

    def get_data(self):
        rng = np.random.default_rng(self.get_seed())
        # Per-image signal template (constant over time) and embedding.
        templates = np.repeat(
            rng.standard_normal((self.n_images, self.n_chans, 1)) * 2.0,
            self.n_times, axis=-1,
        )
        embeddings = rng.standard_normal((self.n_images, self.n_outputs))
        device = get_device()
        ch_names = STANDARD_1020[:self.n_chans]
        ch_names += [
            f"EEG{i:03d}" for i in range(len(ch_names), self.n_chans)
        ]

        X_tr, y_tr = self._make_windows(
            rng, self.n_train, templates, embeddings
        )
        X_te, y_te = self._make_windows(
            rng, self.n_test, templates, embeddings
        )

        return dict(
            train_loader=make_loader(X_tr, y_tr, shuffle=True, device=device),
            test_loader=make_loader(X_te, y_te, device=device),
            n_outputs=self.n_outputs,
            sfreq=self.sfreq,
            ch_names=ch_names,
            chs_info=chs_info_from_names(ch_names),
            n_chans=self.n_chans,
            n_times=self.n_times,
            device=device,
        )
