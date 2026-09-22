"""Ingestion program, shared by the 4 track competitions.

A thin wrapper around ``benchopt run``. A submission ships a trained model:
``submission.py`` (a ``CompetSolver``) + weight files. This program copies
the benchmark the phase ships (``<input_data>/benchmark``) to a writable
workdir, drops in the phase's sealed dataset files, then execs benchopt,
selecting the submission **by file path**::

    benchopt run <workdir> --config <input_data>/config.yaml \
        -s <submission_dir>/submission.py \
        -r 1 --no-cache --no-plot --output submission

Selecting by path (benchopt >= 1.10) loads the Solver from that exact file
and keeps its own ``name``, so a submission reusing a bundled baseline's name
(the example kits ship the baselines as ``submission.py``) is run as itself
instead of being shadowed by the benchmark's ``solvers/`` copy of that name.

The phase ``config.yaml`` is a **native benchopt run config** (``dataset``,
``seed``, ``no_timeout``, ...) plus one competition-only key stripped before
the run (benchopt rejects unknown options): ``scoring`` (parsed by
scoring.py — the config is forwarded with the raw results parquet; all
evaluation happens here). The data location is the image's
$BENCHOPT_DATA_HOME (/app/data, where the compute worker mounts the staged
data read-only).
Runs are inference-only unless the phase config's native ``objective:`` key
selects ``<objective>[training=True]``.
$BENCHOPT_DEBUG ensures any solver error aborts the run.
"""

import os
from pathlib import Path

os.environ["BENCHOPT_DEBUG"] = "true"
# scikit-learn array-API dispatch needs scipy's, read at scipy import time.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import yaml  # noqa: E402

# benchopt artefacts + downloaded data, never copied to the workdir.
IGNORE = shutil.ignore_patterns(
    "outputs", "__cache__", "__pycache__", ".pytest_cache", "data"
)


def seed_hf_cache():
    """Ingest staged cache into writable HF_HOME.
    
     This avoids redownloading very common weights while letting
     participants experiment with new ones.
    """
    staged = (Path(os.environ.get("BENCHOPT_DATA_HOME", "/app/data"))
              / "neural_compet" / "hf_cache")
    if not staged.is_dir():
        return
    cache = Path(os.environ.get("HF_HOME")
                 or Path.home() / ".cache" / "huggingface")
    cache.mkdir(parents=True, exist_ok=True)
    # -s: symlink to the read-only staged files; -n: keep anything already
    # downloaded in this container.
    subprocess.run(["cp", "-rsn", f"{staged}/.", str(cache)], check=False)


def setup_workdir(benchmark_dir, input_dir):
    """Writable benchmark copy + the phase's dataset files."""
    workdir = Path(tempfile.mkdtemp(prefix="compet_run_")) / "benchmark"
    shutil.copytree(benchmark_dir, workdir, ignore=IGNORE)
    for path in sorted((input_dir / "datasets").glob("*.py")):
        shutil.copyfile(path, workdir / "datasets" / path.name)
    return workdir


def main(submission_dir, output_dir, benchmark_dir, input_dir):
    # unresolved paths: ``benchmark_dir`` has been through resolve(), which
    # flattens a dangling link into a plain missing path.
    if not benchmark_dir.exists():
        raise SystemExit(
            f"[ingestion] no benchmark at {benchmark_dir}: the phase bundle "
            "ships it alongside config.yaml (on Codabench, upload the "
            "phase's input_data dataset). A bundle copied without "
            "dereferencing its links lands here too."
        )

    # Point the solvers at the submission folder (shipped weights).
    os.environ["COMPET_SUBMISSION_DIR"] = str(submission_dir)
    workdir = setup_workdir(benchmark_dir, input_dir)

    config, run_config = input_dir / "config.yaml", None
    if not config.exists():
        raise SystemExit(
            f"[ingestion] no phase config at {config} — refusing to run "
            "every dataset. Provide an input_data dir with a config.yaml "
            "(on Codabench, upload the phase's input_data dataset)."
        )
    cfg = yaml.safe_load(config.read_text()) or {}
    # benchopt rejects unknown config keys: strip the competition-only
    # one and pass the rest as a genuine `benchopt run` config file.
    cfg = {k: v for k, v in cfg.items() if k != "scoring"}
    if cfg:
        run_config = workdir.parent / "run_config.yml"
        run_config.write_text(yaml.safe_dump(cfg))

    # Select the submission by file path (benchopt >= 1.10): the Solver is
    # loaded from this exact file and keeps its own name, so it cannot be
    # shadowed by a same-named baseline in the benchmark's solvers/.
    submission = submission_dir / "submission.py"
    if not submission.is_file():
        raise SystemExit(
            f"[ingestion] no submission.py in {submission_dir} — a submission "
            "is a folder with a submission.py defining `class Solver`."
        )

    cmd = [
        "benchopt", "run", str(workdir),
        "-r", "1", "--no-cache", "--no-plot", "--no-html", "--no-display",
        "--output", "submission", "-s", str(submission),
        *(["--config", str(run_config)] if run_config else []),
    ]

    seed_hf_cache()
    print(f"[ingestion] evaluating {submission}", flush=True)
    start = time.time()
    subprocess.run(cmd, check=True)

    # Raw results + config for the scoring program, which only parses them.
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = max((workdir / "outputs").glob("submission*.parquet"))
    shutil.copyfile(result_file, output_dir / "results.parquet")
    (output_dir / "metadata.json").write_text(
        json.dumps({"duration": time.time() - start}))
    shutil.copyfile(config, output_dir / "config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competition ingestion")
    parser.add_argument("--submission-dir", default="/app/ingested_program")
    parser.add_argument("--output-dir", default="/app/output")
    parser.add_argument("--input-data", default="/app/input_data",
                        help="Phase dir: benchmark/, config.yaml, datasets/")
    parser.add_argument("--benchmark-dir",
                        help="benchopt benchmark (default: the phase's)")
    args = parser.parse_args()

    input_data = Path(args.input_data)
    main(
        Path(args.submission_dir),
        Path(args.output_dir),
        Path(args.benchmark_dir or input_data / "benchmark").resolve(),
        input_data,
    )
