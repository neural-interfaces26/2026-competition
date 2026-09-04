"""Zero-dependency simulated dataset — the always-available smoke test.

Stands in for the real EMG-to-pose task (Salter2024 emg2pose, **no
neuralfetch study / neuralbench task config yet** — this simulated dataset is
currently the track's only one): joint-angle trajectories are smooth random
curves in a plausible range, and the EMG channels are an instantaneous linear
mixture of the joint angles + noise, so a per-time-step linear readout can
recover the pose above the noise floor.

Only depends on numpy + torch (the benchmark's base stack), so it powers
``benchopt run tracks/emg_pose -d Simulated`` and ``benchopt test``.
"""

import numpy as np
from benchopt import BaseDataset

from compet_core.data import get_device, make_loader


class Dataset(BaseDataset):

    name = "Simulated"

    requirements = []

    parameters = {
        # 16-ch wrist EMG -> 20 joint angles, as in the real task.
        "n_chans, n_joints, n_times": [(16, 20, 400)],
    }

    test_parameters = {
        "n_chans, n_joints, n_times": [(4, 5, 80)],
    }

    def _make_windows(self, rng, n, mixing):
        # Smooth trajectories: cumulative sum of noise, low-pass by windowed
        # averaging, scaled to a plausible joint-angle range (degrees).
        y = np.cumsum(rng.standard_normal((n, self.n_joints, self.n_times)),
                      axis=-1)
        kernel = np.ones(9) / 9.0
        y = np.apply_along_axis(
            lambda s: np.convolve(s, kernel, mode="same"), -1, y
        )
        y = 30.0 * y / np.abs(y).max(axis=-1, keepdims=True) + 20.0
        # Instantaneous linear mixture + noise.
        X = np.einsum("cj,njt->nct", mixing, y)
        X += 2.0 * rng.standard_normal(X.shape)
        return X.astype(np.float32), y.astype(np.float32)

    def prepare(self):
        # No downloads, so nothing to prepare.
        pass

    def get_data(self):
        rng = np.random.default_rng(self.get_seed())
        mixing = rng.standard_normal((self.n_chans, self.n_joints)) * 0.3
        device = get_device()

        X_tr, y_tr = self._make_windows(rng, 100, mixing)
        X_te, y_te = self._make_windows(rng, 50, mixing)

        return dict(
            train_loader=make_loader(X_tr, y_tr, shuffle=True, device=device),
            test_loader=make_loader(X_te, y_te, device=device),
            n_joints=self.n_joints,
            sfreq=200.0,
            ch_names=None,
            chs_info=None,
            n_chans=self.n_chans,
            n_times=self.n_times,
            device=device,
        )
