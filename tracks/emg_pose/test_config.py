import os

import pytest

# The real dataset's pip stack (neuralset/neuralfetch/neuralbench) pins a CUDA
# torch build — installing it exceeds the CI runners' disk, and the emg2pose
# download is too large for CI anyway. Use ``benchopt install`` locally / on a
# compute node instead.
HEAVY_DATASETS = ("salter2024emg2pose",)


def check_test_dataset_install(dataset_class):
    """Hook to skip dataset install test cases in benchopt tests."""
    if dataset_class.name.lower() in HEAVY_DATASETS:
        pytest.skip("real-data stack is too heavy for CI runners")


def check_test_dataset_get_data(dataset_class):
    """Hook to skip dataset test cases in benchopt tests."""
    if dataset_class.name.lower() in HEAVY_DATASETS:
        pytest.skip("emg2pose is too large for full download in CI")


def check_test_solver_run(solver_class, test_dataset_name, tmp_path):
    """Hook to skip solver test cases in benchopt tests."""
    # Test-trained weights go to a throwaway dir, not the real outputs/.
    os.environ["COMPET_SUBMISSION_DIR"] = str(tmp_path)
