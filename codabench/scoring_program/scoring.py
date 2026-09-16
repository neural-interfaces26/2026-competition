"""Scoring program, shared by the 4 track competitions.

All evaluation happens in the ingestion run; this only parses the raw
benchopt results (``results.parquet``) into the flat ``scores.json`` the
leaderboard consumes. The forwarded phase config's ``scoring.columns`` maps
leaderboard keys to dataframe columns; a listed column that is missing or
NaN fails loudly. Without a config, every float-valued ``objective_<m>``
column becomes a score (integer columns are context, e.g. ``n_classes``).
"""

import json
import math
from pathlib import Path

import numpy as np
import yaml

# Context columns that are never scores, whatever their dtype.
CONTEXT = {"objective_name", "objective_value"}


def main(prediction_dir, output_dir):
    from benchopt.results import read_results

    df = read_results(prediction_dir / "results.parquet")
    config_file = prediction_dir / "config.yaml"
    config = (yaml.safe_load(config_file.read_text())
              if config_file.exists() else {}) or {}

    scores = {}
    columns = (config.get("scoring") or {}).get("columns")
    if columns:
        for key, col in columns.items():
            value = float(df[col].iloc[0]) if col in df.columns else np.nan
            if math.isnan(value):
                raise SystemExit(f"[scoring] column {col!r} (leaderboard "
                                 f"key {key!r}) is missing or NaN.")
            scores[key] = value
    else:  # no config: auto-detect the float metric columns
        for col in df.columns:
            if (col.startswith("objective_") and col not in CONTEXT
                    and np.issubdtype(df[col].dtype, np.floating)
                    and not math.isnan(value := float(df[col].iloc[0]))):
                scores[col.removeprefix("objective_")] = value

    meta_file = prediction_dir / "metadata.json"
    if meta_file.exists():
        scores["duration"] = json.loads(meta_file.read_text())["duration"]

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
