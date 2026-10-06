"""Unit tests for the metric helpers (numpy only).

Run with ``pytest benchmark_utils``.
"""

import numpy as np
import pytest

from benchmark_utils.data import subject_ids
from benchmark_utils.metrics import (
    SequenceRegressionScores, group_means, topk_accuracy,
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
