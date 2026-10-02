# Codabench competition sources

This directory is the source used to build the four Codabench competitions.
It does not contain the participant training workflow.

| Path | Responsibility |
|---|---|
| `competition_*.yaml` | pages, phases, limits, tasks, and leaderboard columns for one track |
| `pages/` | Markdown rendered under each competition's **Get Started** tab |
| `phases/` | phase-specific Benchopt configuration and benchmark payload |
| `ingestion_program/` | extracts a submitted ZIP and runs its model through Benchopt |
| `scoring_program/` | reads the Benchopt result and publishes leaderboard scores |

At evaluation time, Codabench mounts the phase data and participant ZIP into a
prebuilt worker image. The ingestion program runs the bundled track benchmark
in inference-only mode. No participant model is trained and no dependencies
are installed during scoring.

For the public submission contract, read
[`pages/participate.md`](pages/participate.md). Maintainers build uploadable
competition bundles through [`tools/create_bundle.py`](../tools/create_bundle.py).
