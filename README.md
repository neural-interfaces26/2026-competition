# Neural Interfaces 2026 competition benchmarks

This repository is the executable competition layer for the
[EEG/EMG Foundation Challenge 2026](https://neural-interfaces26.github.io/).
It contains the public benchmarks that Codabench runs, the submission
contract, worked solvers, model-packaging bridges, and the tools used to build
the four Codabench competitions.

Use the following sources according
to what you need:

| Source                                                        | Ground truth for                                                                         |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| [Competition website](https://neural-interfaces26.github.io/) | tracks, datasets, schedule, rules, registration, and participant journey                 |
| Each Codabench **Track description** and **Submission Guide** | active phase data and metric, upload rules, limits, and official evaluation              |
| This repository                                               | executable benchmarks, reference implementations, local checks, and submission packaging |

## Find what you need

| Goal                                                     | Go to                                                                                                                                                                      |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Explore one track's data, contract, and worked models    | [Track 01](tracks/image_decoding/README.md), [Track 02](tracks/bci_decoding/README.md), [Track 03](tracks/sleep_onset/README.md), or [Track 04](tracks/emg_pose/README.md) |
| Understand the two development routes in this repository | [`tracks/README.md`](tracks/README.md)                                                                                                                                     |
| Package a NeuralBench checkpoint for Codabench           | [NeuralBench-to-Codabench bridge](tracks/README.md#package-a-neuralbench-checkpoint)                                                                                       |
| Train or package through Benchopt                        | [Benchopt competition workflow](tracks/README.md#develop-and-package-with-benchopt)                                                                                        |
| Implement or validate your own submission                | [`codabench/pages/participate.md`](codabench/pages/participate.md)                                                                                                         |
| Inspect the metric Codabench executes                    | the selected track's `objective.py` and [`codabench/scoring_program/scoring.py`](codabench/scoring_program/scoring.py)                                                     |
| Understand or build the Codabench deployment artifacts   | [`codabench/`](codabench/) and [`tools/`](tools/)                                                                                                                          |

## Repository map

```text
tracks/             four executable Benchopt benchmarks, worked solvers,
                    configs, and NeuralBench checkpoint wrappers
benchmark_utils/    shared task loading, metrics, and submission contract
codabench/          competition definitions, pages, ingestion, and scoring
solution/           minimal sample solutions required by Codabench bundles
tools/              starting-kit, bundle, container, and data utilities
```

Each track is independently runnable. Its `benchmark_utils` entry is a
symbolic link to the single shared directory at the repository root. Bundle
and starting-kit builders dereference that link when producing standalone
archives.

## Optional local Benchopt setup

You need this environment only if you want to run the competition benchmark
locally, use Benchopt to train and package a model, or replay a submission
against the public contract. It is not required to train solely through
NeuralBench, use your own pipeline, or upload an already packaged ZIP.

From the repository root:

```bash
conda create -n neural-interfaces26 python=3.12 pip -y
conda activate neural-interfaces26
python -m pip install -U "benchopt>=1.10"
```

The selected track and config declare their remaining dependencies. For
example, this zero-download Track 02 check installs only what it needs:

```bash
benchopt install tracks/bci_decoding \
    --config tracks/bci_decoding/starter.yml -y
benchopt run tracks/bci_decoding \
    --config tracks/bci_decoding/starter.yml
```

`benchopt prepare` stores real datasets in Benchopt's data directory. Set
`BENCHOPT_DATA_HOME` before preparing a large study if it should live on a
different disk.

Continue with the [shared track workflows](tracks/README.md) or the README for
your selected track.

## Maintainer entry points

- [`codabench/README.md`](codabench/README.md) explains the competition bundle
  sources and the evaluation flow.
- [`tools/README.md`](tools/README.md) documents bundle, starting-kit, Docker,
  and data-preparation commands.
- CI runs the four benchmark test suites, lint checks, and an end-to-end
  ingestion and scoring test.
