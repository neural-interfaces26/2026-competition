"""Scoring program, shared by the 4 track competitions.

All evaluation is done in the ingestion program, which stores the raw
benchopt results dataframe (``results.parquet``). Scoring only **parses** it
(with benchopt's ``read_results``) into the flat ``scores.json`` the
Codabench leaderboard consumes — no labels, no metric computation here.

Each competition holds a single task, so the leaderboard keys are simply the
objective's metric names: every *float-valued* ``objective_<metric>`` column
becomes a score (integer columns are context — e.g. ``n_classes`` — and are
skipped), plus the ingestion ``duration``.
"""

import json
import math
from pathlib import Path

import numpy as np

# Context columns that are never scores, whatever their dtype.
CONTEXT = {"objective_name", "objective_value"}


def main(prediction_dir, output_dir):
    from benchopt.results import read_results

    df = read_results(prediction_dir / "results.parquet")

    scores = {}
    for col in df.columns:
        if not col.startswith("objective_") or col in CONTEXT:
            continue
        if not np.issubdtype(df[col].dtype, np.floating):
            continue
        value = float(df[col].iloc[0])
        if not math.isnan(value):
            scores[col.removeprefix("objective_")] = value

    meta_path = prediction_dir / "metadata.json"
    if meta_path.exists():
        scores.update(json.loads(meta_path.read_text()))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.json").write_text(json.dumps(scores))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Competition scoring")
    parser.add_argument("--prediction-dir", default="/app/input/res")
    parser.add_argument("--output-dir", default="/app/output")
    args = parser.parse_args()

    main(Path(args.prediction_dir), Path(args.output_dir))
