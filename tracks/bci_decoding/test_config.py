import pytest


def check_test_dataset_get_data(dataset_class):
    """Hook to skip dataset test cases in benchopt tests."""
    if dataset_class.name.lower() == "bci":
        # The NEMAR/BIDS download (~1.7 GB, often throttled) is too slow
        # for CI; MOABB-MI covers the real-data path there.
        pytest.skip("BCI studies download is too slow for CI")


def check_test_solver_run(solver_class, test_dataset_name):
    """Hook to skip solver test cases in benchopt tests."""
