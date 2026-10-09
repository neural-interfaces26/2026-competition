"""Unit tests for the streamed evaluation (numpy + torch only).

Run with ``pytest benchmark_utils``.
"""

import numpy as np
import pytest
import torch

from benchmark_utils.data import make_loader
from benchmark_utils.streaming import predict_streams


class CountingModel:
    """Predicts how many windows it has seen since its last reset."""

    def __init__(self):
        self.seen = 0
        self.resets = 0
        self.batch_sizes = []

    def reset_state(self):
        self.resets += 1
        self.seen = 0

    def predict(self, X):
        self.batch_sizes.append(len(X))
        self.seen += len(X)
        return np.full(len(X), self.seen)


def _loader(stream_id, record_id=None, onset=None, batch_size=4):
    n = len(stream_id)
    return make_loader(
        np.zeros((n, 2, 3)), np.arange(n), batch_size=batch_size,
        record_id=stream_id if record_id is None else record_id,
        onset=onset, stream_id=stream_id,
    )


def test_each_stream_runs_on_a_fresh_reset_copy():
    model = CountingModel()
    stream_id = np.array([5, 5, 5, 2, 2, 9])
    y_true, y_pred, info = predict_streams(model, _loader(stream_id))
    # One window per call, a fresh copy reset at each stream start: the
    # count restarts with every stream.
    np.testing.assert_array_equal(y_pred, [1, 2, 3, 1, 2, 1])
    np.testing.assert_array_equal(y_true, np.arange(6))
    np.testing.assert_array_equal(info["stream_id"], stream_id)
    # The model itself is never called nor reset.
    assert (model.seen, model.resets, model.batch_sizes) == (0, 0, [])


def test_streams_span_recordings():
    # A stream of two recordings (e.g. a session of two runs): one copy for
    # the whole stream, each recording's windows in time order.
    stream_id = np.array([0, 0, 0, 0, 1, 1])
    record_id = np.array([0, 0, 1, 1, 2, 2])
    onset = np.array([0, 10, 0, 10, 0, 10])
    _, y_pred, _ = predict_streams(
        CountingModel(), _loader(stream_id, record_id, onset))
    np.testing.assert_array_equal(y_pred, [1, 2, 3, 4, 1, 2])


def test_models_without_reset_state_are_copied():
    class Stateful(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("seen", torch.zeros(()))

        def predict(self, X):
            self.seen += len(X)
            return self.seen.repeat(len(X))

    model = Stateful()
    _, y_pred, _ = predict_streams(model, _loader(np.array([0, 0, 1])))
    np.testing.assert_array_equal(y_pred, [1, 2, 1])
    assert model.seen.item() == 0


def test_order_is_checked():
    with pytest.raises(ValueError, match="contiguous"):
        predict_streams(CountingModel(), _loader(np.array([0, 1, 0])))
    with pytest.raises(ValueError, match="time order"):
        predict_streams(CountingModel(), _loader(
            np.array([0, 0, 0]), record_id=np.array([0, 1, 0])))
    with pytest.raises(ValueError, match="time order"):
        predict_streams(CountingModel(), _loader(
            np.array([0, 0]), onset=np.array([5, 1])))


def test_contract_errors():
    class Batched:
        def predict(self, X):
            return np.zeros((len(X) + 1,))

    with pytest.raises(ValueError, match="one prediction per window"):
        predict_streams(Batched(), _loader(np.array([0])))

    class NotCopyable(CountingModel):
        def __deepcopy__(self, memo):
            raise RuntimeError("no copy")

    with pytest.raises(TypeError, match="deepcopy"):
        predict_streams(NotCopyable(), _loader(np.array([0])))

    loader = [(torch.zeros(1, 2, 3), torch.zeros(1), {"record_id": [0]})]
    with pytest.raises(KeyError, match="stream_id"):
        predict_streams(CountingModel(), loader)
