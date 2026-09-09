import pytest

# The real dataset's pip stack (neuralset/neuralbench) pins a CUDA torch
# build — installing it exceeds the CI runners' disk, and the full Sleep-EDF
# download is too large for CI anyway. Use ``benchopt install`` locally / on
# a compute node instead.
HEAVY_DATASETS = ("sleep-edf",)


def check_test_dataset_install(dataset_class):
    """Hook to skip dataset install test cases in benchopt tests."""
    if dataset_class.name.lower() in HEAVY_DATASETS:
        pytest.skip("real-data stack is too heavy for CI runners")


def check_test_dataset_get_data(dataset_class):
    """Hook to skip dataset test cases in benchopt tests."""
    if dataset_class.name.lower() in HEAVY_DATASETS:
        pytest.skip("Sleep-EDF is too large for full download in CI")


def check_test_solver_install(solver_class):
    """Hook to skip solver install test cases in benchopt tests."""
    if "eegnet" in solver_class.name.lower():
        # pip braindecode pulls a CUDA torchaudio that cannot load against
        # the CPU torch of the CI env; exercised on the cluster instead.
        pytest.skip("braindecode/torchaudio stack unavailable on CI runners")


def check_test_solver_run(solver_class, test_dataset_name):
    """Hook to skip solver test cases in benchopt tests."""
