"""Scoring program for the EEG competition.

All evaluation is done in the ingestion program, which stores the raw benchopt
results dataframe (``results.parquet``). Scoring only **parses** it (with
benchopt's ``read_results``) into the flat ``scores.json`` the Codabench
leaderboard consumes — no labels, no metric computation here.

Each leaderboard key is ``{track}_{task}_{metric}`` (e.g.
``linear_probe_mi_balanced_accuracy``, ``general_sleep_onset_f1``) so the FM
and specialist tracks never collide — plus an overall ``duration``.
"""

import json
from pathlib import Path

# Metric columns (everything else under ``objective_`` is context).
METRICS = [
    "accuracy", "balanced_accuracy",
    "staging_balanced_accuracy", "onset_f1",
]


def main(prediction_dir, output_dir):
    from benchopt.results import read_results

    df = read_results(prediction_dir / "results.parquet")

    scores = {}
    for _, r in df.iterrows():
        track = r.get("objective_track")
        task = r.get("objective_task")
        for metric in METRICS:
            col = f"objective_{metric}"
            value = r.get(col)
            if value is None or (isinstance(value, float) and value != value):
                continue  # absent or NaN
            scores[f"{track}_{task}_{metric}"] = float(value)

    meta_path = prediction_dir / "metadata.json"
    if meta_path.exists():
        scores.update(json.loads(meta_path.read_text()))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.json").write_text(json.dumps(scores))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG competition scoring")
    parser.add_argument("--prediction-dir", default="/app/input/res")
    parser.add_argument("--output-dir", default="/app/output")
    args = parser.parse_args()

    main(Path(args.prediction_dir), Path(args.output_dir))
