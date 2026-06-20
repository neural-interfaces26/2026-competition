import sys  # noqa: F401

import pytest  # noqa: F401


def check_test_dataset_get_data(dataset_class):
    if dataset_class.name.lower() == "sleep-edf":
        pytest.skip("Sleep-EDF is too large for full download in CI")


def check_test_solver_run(solver_class, test_dataset_name):
    """Hook to skip test case in benchopt tests
    """
    if solver_class.name.lower() == 'reve':
        pytest.skip("REVE requires login to get pretrained weights")
