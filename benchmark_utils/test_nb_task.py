"""Unit tests for the pure parts of nb_task (run with ``pytest
benchmark_utils``).

The loading pipeline itself needs the neuralbench stack and real data — it is
validated on the cluster; these tests cover the subset-filter composition,
which is plain dict/string logic.
"""

import pytest

from benchmark_utils.nb_task import build_test_only_filter


def test_predefined_split_with_query_and_existing_filter():
    # xu2025alljoined shape: explicit test query + a shipped filter_stimuli.
    cfg = {
        "study": {
            "split": {"name": "PredefinedSplit",
                      "test_split_query": "label == 'stim_test'"},
            "filter_stimuli": {
                "name": "QueryEvents",
                "query": "type != 'Image' or label in "
                         "['stim_train', 'stim_test']",
            },
        },
        "trigger_event_type": "Image",
    }
    out = build_test_only_filter(cfg)
    assert out["name"] == "QueryEvents"
    assert out["query"] == (
        "(type != 'Image' or label in ['stim_train', 'stim_test']) and "
        "(type not in ['Image'] or (label == 'stim_test'))"
    )


def test_predefined_split_on_data_column():
    # gifford shape: the split column ships with the source data.
    cfg = {
        "study": {"split": {"name": "PredefinedSplit",
                            "test_split_query": None, "col_name": "split"}},
        "trigger_event_type": "Image",
    }
    assert build_test_only_filter(cfg)["query"] == (
        "type not in ['Image'] or (split == 'test')"
    )


def test_runtime_split_is_rejected():
    cfg = {
        "study": {"split": {"name": "SklearnSplit"}},
        "trigger_event_type": "Stimulus",
    }
    with pytest.raises(ValueError, match="PredefinedSplit"):
        build_test_only_filter(cfg)


def test_missing_split_is_rejected():
    with pytest.raises(ValueError, match="PredefinedSplit"):
        build_test_only_filter({"study": {}, "trigger_event_type": "Stimulus"})
