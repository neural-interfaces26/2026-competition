# Codabench sample solutions

These four dependency-light submissions are the sample solutions embedded in
the generated Codabench competition bundles. They are smoke tests for upload,
ingestion, inference, scoring, and leaderboard publication.

The directory name and per-track mapping reflect Codabench's `solutions`
bundle field. This is platform plumbing, not the participant example library.
Participants should use the richer worked models under
[`tracks/<track>/solvers/`](../tracks/) and the generated starting-kit
examples instead.

Each sample works without a committed weight file by constructing a
deterministic fallback model in memory. That behaviour is deliberate for a
platform smoke test and should not be copied into a competitive submission.
