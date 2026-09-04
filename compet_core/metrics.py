"""Metric helpers shared by the track objectives (numpy, sklearn-style).

Kept dependency-light (numpy only) so objectives never need the neuralbench
stack; semantics mirror the official neuralbench metrics.
"""

import numpy as np


def binned_mae(y_true, y_pred, bin_edges=(0.0, 40.0, 90.0, 300.0, 600.0)):
    """Mean absolute error binned by ground-truth value (neuralbench bMAE).

    Targets are partitioned into ``len(bin_edges) - 1`` bins; the MAE is
    computed inside each bin and the reported value is the unweighted mean
    over non-empty bins. Bins follow ``[lo, hi)`` except the last one, which
    includes its upper boundary (so a cap value lands in the last bin);
    out-of-range targets are ignored.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    edges = np.asarray(bin_edges, dtype=np.float64)

    err = np.abs(y_pred - y_true)
    # right-open bins, with the last bin including its upper edge
    idx = np.digitize(y_true, edges[1:-1], right=False)
    in_range = (y_true >= edges[0]) & (y_true <= edges[-1])

    maes = [err[in_range & (idx == b)].mean()
            for b in range(len(edges) - 1)
            if np.any(in_range & (idx == b))]
    return float(np.mean(maes)) if maes else float("nan")


def topk_accuracy(scores, target_idx, k=5):
    """Top-``k`` retrieval accuracy from a score matrix.

    Parameters
    ----------
    scores : array, shape ``(N, M)``
        Similarity of each of the ``N`` queries to the ``M`` candidates.
    target_idx : array, shape ``(N,)``
        Index of each query's true candidate.
    """
    scores = np.asarray(scores)
    target_idx = np.asarray(target_idx)
    order = np.argsort(-scores, axis=1)[:, :k]
    return float((order == target_idx[:, None]).any(axis=1).mean())
