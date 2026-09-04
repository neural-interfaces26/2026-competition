def check_test_dataset_get_data(dataset_class):
    """Hook to skip dataset test cases in benchopt tests."""
    # All current datasets are small enough for CI (Simulated is synthetic,
    # MOABB-MI downloads a single subject).


def check_test_solver_run(solver_class, test_dataset_name):
    """Hook to skip solver test cases in benchopt tests."""
