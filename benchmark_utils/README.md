# Shared benchmark components

This package holds the code shared by all four executable track benchmarks:

| Module | Responsibility |
|---|---|
| `base_solver.py` | public `CompetSolver` submission contract and export hooks |
| `data.py` | device-aware loaders, tensor conversion, and resampling helpers |
| `nb_task.py` | bridge from track datasets to the NeuralBench task and data layer |
| `metrics.py` | shared metric implementations |
| `baselines.py` | small reusable constant and linear baseline models |
| `linear_probe.py` | reusable frozen-encoder probe machinery |

The `benchmark_utils` entry inside each `tracks/<track>/` directory is a
symbolic link to this package, not a copy. Benchopt expects a benchmark to be
self-contained, while keeping the implementation here prevents four versions
of the contract from drifting. Bundle builders follow and dereference the link
when creating standalone Codabench or participant archives.

Participants normally consume this package through a track's datasets,
objective, and solvers. Changes here affect every track and should be tested
across all four benchmarks.
