"""Zero-(EEG-)dependency simulated dataset — the always-available smoke test.

Stands in for the **real tasks** so any model (including task-specific
specialists) can be tested without downloads. The ``task`` parameter selects
which task to mimic, each mapped to its regime:

- ``mi``    : motor imagery, *epoched* — one class per window (channel means
  shifted by a per-class template), label ``y: (N,)``.
- ``sleep`` : sleep staging, *dense* — each window is a sequence of class
  segments (the channel means shift segment-by-segment), label ``y: (N, T)``
  per time-step. The last class acts as the background/null used by the
  ``onset_f1`` metric.

Only depends on numpy + torch (the benchmark's base stack), so it needs no
EEG packages and powers ``benchopt run benchmark/ -d Simulated`` and
``benchopt test``.
"""

import numpy as np
from benchopt import BaseDataset

from benchmark_utils.data import (
    chs_info_from_names, get_device, make_loader,
)

# Each simulated task mimics one real task's regime.
TASK_KIND = {"mi": "epoched", "sleep": "dense"}

# Standard 10-20 electrode names, so name-based foundation models (e.g. REVE)
# can resolve channel positions even on the synthetic dataset. The first
# ``n_chans`` are used.
STANDARD_1020 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]


class Dataset(BaseDataset):

    name = "Simulated"

    requirements = []

    parameters = {
        "task": ["mi", "sleep"],
        "n_chans, n_times": [(8, 200)],
        "n_classes": [3],
    }

    test_parameters = {
        "task": ["mi", "sleep"],
        "n_chans, n_times": [(4, 80)],
        "n_classes": [3],
    }

    def _class_templates(self, rng):
        # One channel-mean template per class; scaled so a linear model can
        # separate the classes above the noise floor.
        return rng.standard_normal((self.n_classes, self.n_chans)) * 2.0

    def _make_epoched(self, rng, n, templates):
        X = np.empty((n, self.n_chans, self.n_times), dtype=np.float32)
        y = rng.integers(0, self.n_classes, size=n)
        for i, k in enumerate(y):
            mean = templates[k][:, None]
            X[i] = mean + rng.standard_normal((self.n_chans, self.n_times))
        return X, y.astype(np.int64)

    def _make_dense(self, rng, n, templates):
        X = np.empty((n, self.n_chans, self.n_times), dtype=np.float32)
        y = np.empty((n, self.n_times), dtype=np.int64)
        n_seg = 4
        bounds = np.linspace(0, self.n_times, n_seg + 1).astype(int)
        for i in range(n):
            for s in range(n_seg):
                a, b = bounds[s], bounds[s + 1]
                k = rng.integers(0, self.n_classes)
                X[i, :, a:b] = (
                    templates[k][:, None]
                    + rng.standard_normal((self.n_chans, b - a))
                )
                y[i, a:b] = k
        return X, y

    def prepare(self):
        # No downloads, so nothing to prepare.
        pass

    def get_data(self):
        rng = np.random.default_rng(self.get_seed())
        templates = self._class_templates(rng)
        task_kind = TASK_KIND[self.task]
        device = get_device()
        ch_names = STANDARD_1020[:self.n_chans]

        n_train, n_test = 120, 60
        if task_kind == "epoched":
            X_tr, y_tr = self._make_epoched(rng, n_train, templates)
            X_te, y_te = self._make_epoched(rng, n_test, templates)
            metrics = ["accuracy", "balanced_accuracy"]
        else:
            X_tr, y_tr = self._make_dense(rng, n_train, templates)
            X_te, y_te = self._make_dense(rng, n_test, templates)
            metrics = ["staging_balanced_accuracy", "onset_f1"]

        return dict(
            train_loader=make_loader(X_tr, y_tr, shuffle=True, device=device),
            test_loader=make_loader(X_te, y_te, device=device),
            task=self.task,
            task_kind=task_kind,
            metrics=metrics,
            n_classes=self.n_classes,
            sfreq=100.0,
            ch_names=ch_names,
            chs_info=chs_info_from_names(ch_names),
            n_chans=self.n_chans,
            n_times=self.n_times,
            device=device,
        )
