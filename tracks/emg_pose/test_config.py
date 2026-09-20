import pytest


def check_test_dataset_get_data(dataset_class):
    """Hook to skip dataset test cases in benchopt tests."""
    # Only the simulated dataset ships for now (no real emg2pose loader yet).


def check_test_solver_install(solver_class):
    """Hook to skip solver install test cases in benchopt tests."""
    if "eegnet" in solver_class.name.lower():
        # pip braindecode pulls a CUDA torchaudio that cannot load against
        # the CPU torch of the CI env; exercised on the cluster instead.
        pytest.skip("braindecode/torchaudio stack unavailable on CI runners")


def check_test_solver_run(solver_class, test_dataset_name):
    """Hook to skip solver test cases in benchopt tests."""
