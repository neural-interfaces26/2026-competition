import pytest

# Real datasets whose pip stack (neuralset/neuralfetch/neuralbench) pins a
# CUDA torch build — installing it exceeds the CI runners' disk. Their
# install (and slow-download get_data) tests are skipped; use
# ``benchopt install`` locally / on a compute node instead.
HEAVY_DATASETS = ("bci", "moabb-mi")


def check_test_dataset_install(dataset_class):
    """Hook to skip dataset install test cases in benchopt tests."""
    if dataset_class.name.lower() in HEAVY_DATASETS:
        pytest.skip("real-data stack is too heavy for CI runners")


def check_test_dataset_get_data(dataset_class):
    """Hook to skip dataset test cases in benchopt tests."""
    if dataset_class.name.lower() in HEAVY_DATASETS:
        pytest.skip("real-data download is too slow/large for CI")


def check_test_solver_run(solver_class, test_dataset_name):
    """Hook to skip solver test cases in benchopt tests."""
