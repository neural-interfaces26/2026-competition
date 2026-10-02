import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge

from benchmark_utils.data import make_loader
from benchmark_utils.linear_probe import LinearProbe, RandomProjectionEncoder


def test_classifier_probe_returns_one_label_per_window():
    generator = torch.Generator().manual_seed(0)
    X = torch.randn(12, 2, 40, generator=generator)
    y = torch.tensor([0, 1] * 6)
    loader = make_loader(X, y, batch_size=4)

    probe = LinearProbe(RandomProjectionEncoder(2, patch_len=10, d=8))
    probe.fit(loader)

    prediction = probe.predict(X[:3])
    assert prediction.shape == (3,)
    assert set(np.unique(prediction)).issubset({0, 1})


def test_multioutput_probe_joblib_roundtrip(tmp_path):
    generator = torch.Generator().manual_seed(1)
    X = torch.randn(10, 2, 40, generator=generator)
    y = torch.randn(10, 5, generator=generator)
    loader = make_loader(X, y, batch_size=5)

    encoder = RandomProjectionEncoder(2, patch_len=10, d=8, seed=3)
    probe = LinearProbe(encoder, Ridge()).fit(loader)
    expected = probe.predict(X[:2])

    path = tmp_path / "weights.joblib"
    joblib.dump(probe.head, path)
    restored = LinearProbe(
        RandomProjectionEncoder(2, patch_len=10, d=8, seed=3), Ridge()
    )
    restored.head = joblib.load(path)

    np.testing.assert_allclose(restored.predict(X[:2]), expected)
