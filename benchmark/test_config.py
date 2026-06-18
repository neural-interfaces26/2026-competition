import sys  # noqa: F401

import pytest  # noqa: F401


def check_test_solver_install(solver_class):
    """Hook called in `test_solver_install`.

    If one solver needs to be skipped/xfailed on a particular architecture,
    call pytest.xfail when detecting the situation.
    """
    pass
