"""Unit tests for the metric helpers (numpy only).

Run with ``pytest benchmark_utils``.
"""

import numpy as np
import pytest

from benchmark_utils.data import subject_ids
from benchmark_utils.metrics import (
    SequenceRegressionScores, balanced_accuracy, binned_mae, group_means,
    group_scores, topk_accuracy,
)


def test_topk_accuracy_matches_argsort_without_ties():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((50, 20))
    target = rng.integers(0, 20, size=50)
    for k in (1, 5):
        top = np.argsort(-scores, axis=1)[:, :k]
        expected = (top == target[:, None]).any(axis=1).mean()
        assert topk_accuracy(scores, target, k=k) == expected


def test_topk_accuracy_ties_share_their_average_rank():
    # Constant scores: every candidate ties at rank (M - 1) / 2.
    assert topk_accuracy(np.zeros((3, 12)), [0, 1, 2], k=5) == 0.0
    assert topk_accuracy(np.zeros((3, 6)), [0, 1, 2], k=5) == 1.0
    # One tie with another candidate: rank 0.5, a top-1 hit.
    assert topk_accuracy([[0.5, 0.5, 0.1]], [0], k=1) == 1.0
    # Two other candidates tie above the target: rank 2, a top-2 miss.
    assert topk_accuracy([[0.9, 0.9, 0.1]], [2], k=2) == 0.0


def test_topk_accuracy_nan_true_score_is_a_miss():
    assert topk_accuracy([[np.nan, 0.1, 0.2]], [0], k=5) == 0.0


def test_group_means():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 0.0], [5.0, 5.0]])
    keys = np.array([[1, 0], [1, 0], [0, 2], [1, 1]])
    unique_keys, means = group_means(values, keys)
    np.testing.assert_array_equal(unique_keys, [[0, 2], [1, 0], [1, 1]])
    np.testing.assert_allclose(means, [[10.0, 0.0], [2.0, 3.0], [5.0, 5.0]])
    unique_keys, means = group_means(np.arange(4.0), [3, 3, 1, 3])
    np.testing.assert_array_equal(unique_keys[:, 0], [1, 3])
    np.testing.assert_allclose(means, [2.0, 4.0 / 3.0])


def _reference_scores(y_pred, y_true, groups):
    """Direct (non-streaming) computation on (B, C, T) arrays."""
    steps_pred = y_pred.transpose(0, 2, 1).reshape(-1, y_pred.shape[1])
    steps_true = y_true.transpose(0, 2, 1).reshape(-1, y_true.shape[1])
    steps_group = np.repeat(groups, y_true.shape[2])
    keep = ~np.isnan(steps_true).any(axis=1)
    p, t, g = steps_pred[keep], steps_true[keep], steps_group[keep]
    err = p - t
    group_mae = np.array([np.abs(err[g == s]).mean() for s in np.unique(g)])
    ss_res = (err ** 2).sum(axis=0)
    ss_tot = ((t - t.mean(axis=0)) ** 2).sum(axis=0)
    return dict(
        mae=np.abs(err).mean(),
        mae_group_mean=group_mae.mean(),
        mae_group_std=group_mae.std(ddof=1),
        n_groups=len(group_mae),
        rmse=np.sqrt((err ** 2).mean()),
        r2=(1 - ss_res / ss_tot).mean(),
    )


def _random_batch(rng, n=12, n_chans=5, n_times=30, n_groups=3):
    y_true = rng.standard_normal((n, n_chans, n_times)) + np.arange(n_chans)[
        :, None]
    y_pred = y_true + 0.5 * rng.standard_normal(y_true.shape)
    groups = rng.integers(0, n_groups, size=n)
    return y_pred, y_true, groups


def test_sequence_scores_match_reference_and_stream():
    rng = np.random.default_rng(0)
    y_pred, y_true, groups = _random_batch(rng)
    y_true[2, 1, 7] = np.nan     # an unlabelled step, dropped for all channels
    expected = _reference_scores(y_pred, y_true, groups)

    for batch_size in (12, 5, 1):
        scores = SequenceRegressionScores()
        for start in range(0, len(y_true), batch_size):
            sl = slice(start, start + batch_size)
            scores.update(y_pred[sl], y_true[sl], groups[sl])
        out = scores.compute()
        assert out["n_groups"] == expected["n_groups"]
        for key, value in expected.items():
            assert out[key] == pytest.approx(value, rel=1e-9), key


def test_sequence_scores_single_group_has_no_spread():
    rng = np.random.default_rng(1)
    y_pred, y_true, _ = _random_batch(rng)
    scores = SequenceRegressionScores()
    scores.update(y_pred, y_true, np.zeros(len(y_true), dtype=int))
    out = scores.compute()
    assert out["n_groups"] == 1
    assert out["mae_group_mean"] == pytest.approx(out["mae"])
    assert np.isnan(out["mae_group_std"])


def test_sequence_scores_nan_prediction_propagates():
    rng = np.random.default_rng(2)
    y_pred, y_true, groups = _random_batch(rng)
    y_pred[0, 0, 0] = np.nan
    scores = SequenceRegressionScores()
    scores.update(y_pred, y_true, groups)
    assert np.isnan(scores.compute()["mae"])


def test_subject_ids():
    assert list(subject_ids({"subject_id": np.array([4, 7])}, 2)) == [4, 7]
    assert list(subject_ids({"record_id": np.array([1, 1]),
                             "onset": np.array([0, 5])}, 2)) == [1, 1]
    assert list(subject_ids({}, 3)) == [0, 0, 0]


def test_binned_mae_weights():
    edges = (0.0, 40.0, 90.0, 300.0, 600.0)
    # One target per bin, errors 1, 2, 3, 4 s; the cap value is in range.
    y_true = np.array([10.0, 50.0, 100.0, 600.0])
    y_pred = y_true + np.array([1.0, -2.0, 3.0, -4.0])
    assert binned_mae(y_true, y_pred, edges) == pytest.approx(2.5)
    w = (10.0, 5.0, 3.0, 1.0)
    expected = (10 * 1 + 5 * 2 + 3 * 3 + 1 * 4) / 19
    assert binned_mae(y_true, y_pred, edges, w) == pytest.approx(expected)
    # Only the non-empty bins weigh: the first two here.
    assert binned_mae(y_true[:2], y_pred[:2], edges, w) == pytest.approx(
        (10 * 1 + 5 * 2) / 15)
    # Out-of-range targets are ignored; nothing in range gives NaN.
    assert np.isnan(binned_mae([700.0], [0.0], edges, w))
    with pytest.raises(ValueError, match="bin_weights"):
        binned_mae(y_true, y_pred, edges, (1.0, 1.0))
    with pytest.raises(ValueError, match="bin_weights"):
        binned_mae(y_true, y_pred, edges, (1.0, 0.0, 1.0, 1.0))


def test_balanced_accuracy():
    # Every class present: the mean of the per-class recalls.
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 2, 0])
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(
        (0.5 + 1.0 + 0.5) / 3)
    # A class predicted but absent from the targets counts as recall 0
    # (torchmetrics), where scikit-learn would leave it out.
    assert balanced_accuracy([0, 0, 1, 1], [0, 2, 1, 1]) == pytest.approx(
        (0.5 + 1.0 + 0.0) / 3)
    # A class absent from both is skipped.
    assert balanced_accuracy([0, 1], [0, 1]) == 1.0


def test_group_scores():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 1, 1])
    groups = np.array([[7, 0], [7, 0], [3, 0], [3, 0], [3, 1], [3, 1]])
    scores = group_scores(balanced_accuracy, y_true, y_pred, groups)
    # Sorted keys: (3, 0), (3, 1), (7, 0).
    np.testing.assert_allclose(scores, [0.5, 0.5, 1.0])
    # One-dimensional keys, and every group counts equally.
    scores = group_scores(lambda t, p: float(np.mean(t == p)),
                          y_true, y_pred, [1, 1, 1, 1, 1, 2])
    np.testing.assert_allclose(scores, [0.8, 0.0])
    assert len(group_scores(balanced_accuracy, [], [], [])) == 0
