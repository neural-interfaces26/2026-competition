"""Metric helpers shared by the track objectives (numpy, sklearn-style).

Kept dependency-light (numpy only) so objectives never need the neuralbench
stack; semantics mirror the official neuralbench metrics.
"""

import numpy as np


def binned_mae(y_true, y_pred, bin_edges=(0.0, 40.0, 90.0, 300.0, 600.0),
               bin_weights=None):
    """Mean absolute error binned by ground-truth value (neuralbench bMAE).

    Targets are partitioned into ``len(bin_edges) - 1`` bins and the MAE is
    computed inside each bin. The reported value is the mean of the per-bin
    MAEs over the non-empty bins, weighted by ``bin_weights`` when given
    (W-bMAE): ``sum(w * mae) / sum(w)`` over the non-empty bins, as
    neuralbench's ``BinnedMAE``. Bins follow ``[lo, hi)`` except the last
    one, which includes its upper boundary (so a cap value lands in the last
    bin); out-of-range targets are ignored.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    edges = np.asarray(bin_edges, dtype=np.float64)
    n_bins = len(edges) - 1
    weights = (np.ones(n_bins) if bin_weights is None
               else np.asarray(bin_weights, dtype=np.float64))
    if weights.shape != (n_bins,) or np.any(weights <= 0):
        raise ValueError(f"bin_weights must hold {n_bins} positive values, "
                         f"one per bin; got {bin_weights!r}")

    err = np.abs(y_pred - y_true)
    # right-open bins, with the last bin including its upper edge
    idx = np.digitize(y_true, edges[1:-1], right=False)
    in_range = (y_true >= edges[0]) & (y_true <= edges[-1])

    nonempty = [b for b in range(n_bins) if np.any(in_range & (idx == b))]
    if not nonempty:
        return float("nan")
    maes = np.array([err[in_range & (idx == b)].mean() for b in nonempty])
    w = weights[nonempty]
    return float((w * maes).sum() / w.sum())


def balanced_accuracy(y_true, y_pred):
    """Balanced accuracy as neuralbench logs it: mean recall over classes.

    The classes are those present in ``y_true`` or ``y_pred``, as in
    torchmetrics' multiclass ``Accuracy(average="macro")``: a class that is
    predicted but absent from the targets counts with a recall of 0, where
    scikit-learn's ``balanced_accuracy_score`` leaves it out. The two agree
    whenever every predicted class occurs in the targets.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.union1d(y_true, y_pred)
    if not len(classes):
        return float("nan")
    recalls = [np.mean(y_pred[y_true == c] == c) if np.any(y_true == c)
               else 0.0 for c in classes]
    return float(np.mean(recalls))


def group_scores(metric, y_true, y_pred, groups):
    """Score each group of rows on its own (neuralbench's ``GroupedMetric``).

    Parameters
    ----------
    metric : callable
        ``metric(y_true, y_pred) -> float``, applied to each group's rows.
    y_true, y_pred : arrays, shape ``(N, ...)``
    groups : array, shape ``(N,)`` or ``(N, K)``
        Group key, or tuple of keys, of each row.

    Returns
    -------
    scores : array, shape ``(G,)``
        One score per distinct key, in sorted key order. Their mean is
        ``GroupedMetric(reduction="mean")``: every group counts equally,
        however many rows it holds.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if not len(y_true):
        return np.array([])
    keys = np.asarray(groups).reshape(len(y_true), -1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return np.array([metric(y_true[inverse == g], y_pred[inverse == g])
                     for g in range(inverse.max() + 1)])


def topk_accuracy(scores, target_idx, k=5):
    """Top-``k`` retrieval accuracy from a score matrix.

    A query is a hit when the zero-based rank of its true candidate is below
    ``k``. Tied scores share their average rank, as in neuraltrain's
    ``Rank``/``TopkAcc``, so tied candidates are not ranked by their index;
    a query whose true score is NaN is a miss.

    Parameters
    ----------
    scores : array, shape ``(N, M)``
        Similarity of each of the ``N`` queries to the ``M`` candidates.
    target_idx : array, shape ``(N,)``
        Index of each query's true candidate.
    """
    scores = np.asarray(scores)
    target_idx = np.asarray(target_idx).reshape(-1)
    true = np.take_along_axis(scores, target_idx[:, None], axis=1)
    rank = ((scores > true).sum(axis=1)
            + (scores >= true).sum(axis=1) - 1) / 2
    rank[np.isnan(true[:, 0])] = np.inf
    return float((rank < k).mean())


def group_means(values, keys):
    """Average the rows of ``values`` that share a key.

    Parameters
    ----------
    values : array, shape ``(N, ...)``
    keys : array, shape ``(N,)`` or ``(N, K)``
        Integer key, or tuple of keys, of each row.

    Returns
    -------
    unique_keys : array, shape ``(G, K)``
        The distinct keys, sorted.
    means : array, shape ``(G, ...)``
        Float64 mean of the rows of each key.
    """
    values = np.asarray(values)
    keys = np.asarray(keys).reshape(len(values), -1)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(unique_keys),) + values.shape[1:])
    np.add.at(sums, inverse, values)
    counts = np.bincount(inverse, minlength=len(unique_keys))
    return unique_keys, sums / counts.reshape((-1,) + (1,) * (values.ndim - 1))


class SequenceRegressionScores:
    """Streaming scores of channel-major sequence targets ``(B, C, T)``.

    Follows neuralbench's dense-target evaluation (``emg/pose``): every time
    step is a row and every channel an output, and a step whose target has a
    NaN in any channel is left out. Call :meth:`update` once per batch, then
    :meth:`compute`:

    - ``mae``: mean absolute error over every (step, channel), pooled;
    - ``mae_group_mean`` / ``mae_group_std``: mean and sample standard
      deviation (``ddof=1``, NaN below two groups) of the per-group MAE,
      neuralbench's ``GroupedMetric`` with ``reduction="mean"``/``"std"``;
    - ``n_groups``: number of groups with at least one scored step;
    - ``rmse``: root mean squared error, pooled;
    - ``r2``: coefficient of determination of each channel across steps,
      averaged over channels (torchmetrics ``R2Score``, uniform average).
    """

    def __init__(self):
        self._groups, self._abs_sums, self._counts = [], [], []
        self._sq_err = self._y_sum = self._y_sq_sum = 0.0
        self._n_steps = 0

    def update(self, y_pred, y_true, groups):
        """Add a batch: ``y_pred`` and ``y_true`` of shape ``(B, C, T)``,
        ``groups`` of shape ``(B,)`` (one group per window)."""
        y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
        err = y_pred - y_true
        labelled = ~np.isnan(y_true).any(axis=1)          # (B, T)
        if not labelled.all():
            unlabelled = ~labelled[:, None, :]
            err = np.where(unlabelled, 0.0, err)
            y_true = np.where(unlabelled, 0.0, y_true)

        self._groups.append(np.asarray(groups).reshape(-1))
        self._abs_sums.append(np.abs(err).sum(axis=(1, 2), dtype=np.float64))
        self._counts.append(labelled.sum(axis=1) * y_true.shape[1])
        # Per-channel sums for the RMSE and the R2.
        self._sq_err = self._sq_err + np.square(err).sum(
            axis=(0, 2), dtype=np.float64)
        self._y_sum = self._y_sum + y_true.sum(axis=(0, 2), dtype=np.float64)
        self._y_sq_sum = self._y_sq_sum + np.square(y_true).sum(
            axis=(0, 2), dtype=np.float64)
        self._n_steps += int(labelled.sum())

    def compute(self):
        """Return the scores as a dict of floats (``n_groups`` an int)."""
        groups = np.concatenate(self._groups)
        abs_sums = np.concatenate(self._abs_sums)
        counts = np.concatenate(self._counts).astype(np.float64)

        _, inverse = np.unique(groups, return_inverse=True)
        inverse = inverse.reshape(-1)
        group_abs = np.bincount(inverse, weights=abs_sums)
        group_counts = np.bincount(inverse, weights=counts)
        group_mae = (group_abs[group_counts > 0]
                     / group_counts[group_counts > 0])

        n_values = counts.sum()
        mean_y = self._y_sum / max(self._n_steps, 1)
        ss_tot = self._y_sq_sum - self._y_sum * mean_y
        ss_res = self._sq_err
        # torchmetrics' guard for (near-)constant channels.
        res_ok = ~np.isclose(ss_res, 0.0, atol=1e-4)
        tot_ok = ~np.isclose(ss_tot, 0.0, atol=1e-4)
        r2 = np.ones_like(ss_res)
        both = res_ok & tot_ok
        r2[both] = 1.0 - ss_res[both] / ss_tot[both]
        r2[res_ok & ~tot_ok] = 0.0

        nan = float("nan")
        return dict(
            mae=float(abs_sums.sum() / n_values) if n_values else nan,
            mae_group_mean=float(group_mae.mean()) if len(group_mae) else nan,
            mae_group_std=(float(group_mae.std(ddof=1))
                           if len(group_mae) > 1 else nan),
            n_groups=int(len(group_mae)),
            rmse=float(np.sqrt(ss_res.sum() / n_values)) if n_values else nan,
            r2=float(r2.mean()) if self._n_steps > 1 else nan,
        )
