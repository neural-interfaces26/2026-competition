"""Shared components for the 4 competition tracks.

This is the benchmarks' ``benchmark_utils`` package: benchopt loads it by
path and registers it under that name, which is why each track links to this
one directory instead of holding a copy. One module per concern (``data``,
``base_solver``, ``baselines``, ``metrics``, ``nb_task``), so importing one
topic does not pull in the (possibly heavy) dependencies of the others.
"""
