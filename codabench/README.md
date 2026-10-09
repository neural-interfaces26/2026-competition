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

## Updating a task on a live competition

Codabench keeps every score attached to the task it was computed on. To change
a scoring program or dataset after participants have submitted:

- **Edit the task in place** (Tasks page, or `PATCH /api/tasks/{id}/`). The task
  id stays the same and existing scores remain visible. When patching through
  the API, always send all four dataset keys (`ingestion_program`,
  `scoring_program`, `input_data`, `reference_data`): a key left out is reset to
  empty.
- To reuse data already on Codabench, upload a `task.yaml` that references the
  dataset keys through `POST /api/tasks/upload_task/` instead of re-uploading
  the archives.
- If the task must be replaced by a new one, swap it in the phase settings and
  **keep the old task**. Old submissions then show `n/a` on the leaderboard and
  can be re-run or resubmitted on the new task by an organizer.
- **Never delete a task that submissions have run on.** Deletion clears the
  task reference of every such submission: their scores show as `n/a` with no
  record of what they were scored on, and they cannot be re-run.
