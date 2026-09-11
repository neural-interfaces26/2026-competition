import os

# The benchmark tests exercise the full contract including training: enable
# the opt-in fit (a plain run is inference-only, like the platform).
os.environ.setdefault("COMPET_TRAINING", "1")


def check_test_dataset_get_data(dataset_class):
    """Hook to skip dataset test cases in benchopt tests."""
    # Only the simulated dataset ships for now (no real emg2pose loader yet).


def check_test_solver_run(solver_class, test_dataset_name):
    """Hook to skip solver test cases in benchopt tests."""
