"""Ingestion program for the EEG competition.

Runs the bundled benchopt benchmark on the participant's submission *via the
programmatic API* (``benchopt.run_benchmark``, no subprocess), then writes a
tidy results dataframe. The scoring program only parses that dataframe — all
evaluation happens here (see the competition design: "run everything in the
ingestion").

Layout (mirrors the Codabench bundle):
- ``benchmark/``         the standalone benchopt benchmark.
- ``<submission-dir>/``  the participant's ``submission.py`` (put on sys.path
                         so the solvers import their encoder/model).
"""

import os
# scikit-learn array-API dispatch (torch tensors through the linear head)
# needs scipy's array-API support, gated behind this env var and read at
# scipy import time — set it before benchopt/scipy are imported.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

# Metric columns (everything else under ``objective_`` is context).
METRICS = [
    "accuracy", "balanced_accuracy",
    "staging_balanced_accuracy", "onset_f1",
]

SOLVER_BY_TRACK = {"linear_probe": "LinearProbe", "general": "General"}


def tidy_results(df):
    """Melt a benchopt result frame into long per-metric rows."""
    rows = []
    for _, r in df.iterrows():
        for metric in METRICS:
            col = f"objective_{metric}"
            if col not in df.columns or pd.isna(r[col]):
                continue
            task = r.get("objective_task", r.get("dataset_name"))
            rows.append({
                "task": task,
                "data_name": r.get("dataset_name"),
                "track": r.get("objective_track"),
                "task_kind": r.get("objective_task_kind"),
                "solver": r.get("solver_name"),
                "metric": metric,
                "score": float(r[col]),
                "time": float(r.get("time", float("nan"))),
            })
    return pd.DataFrame(rows)


def main(submission_dir, output_dir, benchmark_dir, tracks, datasets):
    sys.path.insert(0, str(submission_dir))
    from benchopt import run_benchmark

    solver_names = [SOLVER_BY_TRACK[t] for t in tracks]

    start = time.time()
    save_file = run_benchmark(
        str(benchmark_dir),
        solver_names=solver_names,
        dataset_names=datasets,                 # None -> all tasks
        objective_filters=[f"EEG[track={t}]" for t in tracks],
        max_runs=1,
        n_repetitions=1,
        plot_result=False,
        display=False,
        html=False,
        show_progress=True,
    )
    duration = time.time() - start

    df = pd.read_parquet(save_file)
    results = tidy_results(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_dir / "results.parquet", index=False)
    results.to_csv(output_dir / "results.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps({"duration": duration})
    )
    print(f"Ingestion done in {duration:.1f}s; {len(results)} result rows.")
    print(results.to_string(index=False))


if __name__ == "__main__":
    here = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(description="EEG competition ingestion")
    parser.add_argument("--submission-dir", default="/app/ingested_program")
    parser.add_argument("--output-dir", default="/app/output")
    parser.add_argument(
        "--benchmark-dir", default=str(here.parent / "benchmark"),
    )
    parser.add_argument(
        "--track", default="both",
        choices=["linear_probe", "general", "both"],
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Dataset names to run (default: all). e.g. Simulated",
    )
    args = parser.parse_args()

    tracks = (["linear_probe", "general"] if args.track == "both"
              else [args.track])

    main(
        Path(args.submission_dir),
        Path(args.output_dir),
        Path(args.benchmark_dir),
        tracks,
        args.datasets,
    )
