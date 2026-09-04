"""Ingestion program, shared by the 4 track competitions.

A submission is a folder with a ``submission.py`` defining a **benchopt
solver** (``class Solver(CompetSolver)``) plus any weight files. Each
competition bundle ships one track's benchmark under ``benchmark/`` next to
this program, together with the ``compet_core`` package. This program:

1. puts the bundle root on ``sys.path`` (so ``import compet_core`` works
   without an install) and exports ``COMPET_SUBMISSION_DIR`` (so the solver
   finds its shipped weights) and ``COMPET_INFERENCE_ONLY=1`` (submissions
   are evaluated **inference-only** — models must arrive fully trained),
2. copies the submission solver(s) into the benchmark's ``solvers/`` so
   benchopt discovers them,
3. runs the benchmark on them via the programmatic API
   (``benchopt.run_benchmark``, no subprocess), and
4. stores the **raw** benchopt results dataframe with ``save_results``.

The scoring program only reads that dataframe (``read_results``) — all
evaluation happens here.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
BUNDLE_ROOT = HERE.parent

# The bundled shared package (no install needed, deps are in the image).
sys.path.insert(0, str(BUNDLE_ROOT))

# Submissions are evaluated inference-only: CompetSolver.run skips ``fit``.
os.environ["COMPET_INFERENCE_ONLY"] = "1"

# scikit-learn array-API dispatch (torch tensors through linear heads) needs
# scipy's array-API support, read at scipy import time.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402


def discover_submission_solvers(submission_dir):
    """Find ``class Solver`` in the submission and return their names."""
    names = []
    for path in sorted(submission_dir.glob("*.py")):
        spec = importlib.util.spec_from_file_location(
            f"_sub_{path.stem}", path
        )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"[ingestion] skip {path.name}: import failed ({e!r})")
            continue
        solver = getattr(module, "Solver", None)
        if solver is not None and getattr(solver, "name", None):
            names.append((path, solver.name))
    return names


def main(submission_dir, output_dir, benchmark_dir, datasets):
    from benchopt import run_benchmark
    from benchopt.results import read_results, save_results

    # Point the solvers at the submission folder (shipped weights).
    os.environ["COMPET_SUBMISSION_DIR"] = str(submission_dir)

    solvers_dir = benchmark_dir / "solvers"
    found = discover_submission_solvers(submission_dir)
    if not found:
        raise SystemExit(
            f"No submission solver (class Solver) found in {submission_dir}."
        )

    # Copy submission solver files into the benchmark so benchopt finds them.
    copied, solver_names = [], []
    for path, name in found:
        dst = solvers_dir / f"_submission_{path.stem}.py"
        shutil.copyfile(path, dst)
        copied.append(dst)
        solver_names.append(name)
    print(f"[ingestion] running submission solvers: {solver_names}")

    try:
        start = time.time()
        save_file = run_benchmark(
            str(benchmark_dir),
            solver_names=solver_names,
            dataset_names=datasets,                 # None -> all datasets
            max_runs=1,
            n_repetitions=1,
            plot_result=False,
            display=False,
            html=False,
            show_progress=True,
        )
        duration = time.time() - start
    finally:
        for dst in copied:
            dst.unlink(missing_ok=True)

    # Round-trip the raw benchopt dataframe (keeps packed artefacts).
    df = read_results(save_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(df, output_dir / "results.parquet", uniquify=False)
    (output_dir / "metadata.json").write_text(
        json.dumps({"duration": duration})
    )
    print(f"[ingestion] done in {duration:.1f}s; {len(df)} result rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competition ingestion")
    parser.add_argument("--submission-dir", default="/app/ingested_program")
    parser.add_argument("--output-dir", default="/app/output")
    parser.add_argument(
        "--benchmark-dir", default=str(BUNDLE_ROOT / "benchmark"),
        help="The track's benchopt benchmark (bundles ship it as benchmark/)",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Dataset names to run (default: all). e.g. Simulated",
    )
    args = parser.parse_args()

    main(
        Path(args.submission_dir),
        Path(args.output_dir),
        Path(args.benchmark_dir).resolve(),
        args.datasets,
    )
