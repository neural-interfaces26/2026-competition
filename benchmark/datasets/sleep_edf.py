"""Sleep staging / onset detection (dense) from Sleep-EDF, via neuralfetch.

A per-time-step target sequence over a long continuous window — the *dense*
regime. Data loading goes through the neuralfetch ``Kemp2000Analysis`` study
(Sleep-EDF / Sleep Cassette) and the neuralset segmenter, behind
``benchmark_utils.neuralset_task`` so nothing downstream is tied to neuralset.

The 5 sleep stages (W, N1, N2, N3, R) are classes 0..4; class 5 is the
background/null used by the ``onset_f1`` metric. Long sliding windows are cut
over the continuous recording and the model must emit a stage *sequence* —
this is what makes the task genuinely continuous rather than pre-epoched 30 s
windows.

Requires a one-time download (``benchopt prepare`` / ``tools/setup_data.py``);
the zero-dependency ``Simulated`` dataset covers no-network smoke testing.
"""

import numpy as np
from benchopt import BaseDataset
from benchopt.config import get_data_path
from neuralfetch.studies.kemp2000analysis import Kemp2000Analysis

from benchmark_utils.data import make_loader
from benchmark_utils.neuralset_task import load_dense

# Sleep stages -> class index; background/null is the last class.
STAGE_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
N_CLASSES = len(STAGE_MAP) + 1
BACKGROUND = N_CLASSES - 1


class Dataset(BaseDataset):

    name = "Sleep-EDF"

    requirements = [
        "pip::neuralset", "pip::neuralfetch", "pip::mne",
        "scikit-learn", "pip::torch",
    ]

    parameters = {
        # Keep the POC small: one subject; long windows for a dense sequence.
        "subject": ["00"],
        "frequency": [100.0],   # Sleep-EDF native rate
        "window_s": [300.0],    # 5-minute windows -> per-step stage sequence
        "test_size": [0.4],
    }

    test_parameters = {
        "subject": ["00"],
        "frequency": [100.0],
        "window_s": [60.0],
        "test_size": [0.5],
    }

    def _study(self):
        path = get_data_path("compet_eeg")
        query = f'subject == "Kemp2000Analysis/{self.subject}"'
        return Kemp2000Analysis(path=str(path), query=query)

    def _ensure_prepared(self):
        # Idempotent: mne skips already-downloaded files.
        self._study().download()

    def prepare(self):
        self._ensure_prepared()

    def get_data(self):
        self._ensure_prepared()

        X, y, record_id, onset = load_dense(
            self._study(),
            signal_event_type="Eeg",
            window_s=self.window_s,
            frequency=self.frequency,
            event_type="SleepStage",
            field="stage",
            mapping=STAGE_MAP,
            background=BACKGROUND,
        )

        # Window-level train/test split (reproducible).
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
            task="sleep",
            task_kind="dense",
            metrics=["staging_balanced_accuracy", "onset_f1"],
            n_classes=N_CLASSES,
            sfreq=self.frequency,
            ch_names=None,
            chs_info=None,
            n_chans=X.shape[1],
            n_times=X.shape[2],
        )
