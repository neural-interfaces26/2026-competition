"""Ingestion program, shared by the 4 track competitions.

A thin wrapper around ``benchopt run``. A submission ships a trained model:
``submission.py`` (a ``CompetSolver``) + weight files. This program copies
the track's benchmark ($COMPET_BENCHMARK_DIR in the image, ``benchmark/`` in
the bundle) to a writable workdir, drops in the submission solver(s) and the
phase's sealed dataset files, then execs::

    benchopt run <workdir> --config <input_data>/config.yaml -s <solver> \
        -r 1 --no-cache --no-plot --output submission

The phase ``config.yaml`` is a **native benchopt run config** (``dataset``,
``seed``, ``no_timeout``, ...) plus two competition-only keys stripped
before the run (benchopt rejects unknown options): ``data_home``
(-> $BENCHOPT_DATA_HOME) and ``scoring`` (parsed by scoring.py — the config
is forwarded with the raw results parquet; all evaluation happens here).
Submissions are evaluated inference-only ($COMPET_INFERENCE_ONLY) and any
solver error aborts the run with its traceback ($BENCHOPT_DEBUG).
"""

import os
import sys
from pathlib import Path

os.environ["COMPET_INFERENCE_ONLY"] = "1"
os.environ["BENCHOPT_DEBUG"] = "true"
# scikit-learn array-API dispatch needs scipy's, read at scipy import time.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import yaml  # noqa: E402

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
# benchopt artefacts + downloaded data, never copied to the workdir.
IGNORE = shutil.ignore_patterns(
    "outputs", "__cache__", "__pycache__", ".pytest_cache", "data")


def setup_workdir(benchmark_dir, input_dir):
    """Writable benchmark copy, compet_core sibling, phase dataset files."""
    workroot = Path(tempfile.mkdtemp(prefix="compet_run_"))
    workdir = workroot / "benchmark"
    shutil.copytree(benchmark_dir, workdir, ignore=IGNORE)
    core = next(p for p in benchmark_dir.parents
                if (p / "compet_core" / "__init__.py").exists())
    shutil.copytree(core / "compet_core", workroot / "compet_core",
                    ignore=IGNORE)
    for path in sorted((input_dir / "datasets").glob("*.py")):
        shutil.copyfile(path, workdir / "datasets" / path.name)
    return workdir


def install_submission_solvers(submission_dir, workdir):
    """Copy the submission's Solver files into the benchmark; return names."""
    sys.path.insert(0, str(workdir))  # submissions import benchmark_utils
    names = []
    for path in sorted(submission_dir.glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"_sub_{path.stem}",
                                                      path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"[ingestion] skip {path.name}: import failed ({e!r})")
            continue
        solver = getattr(module, "Solver", None)
        if solver is not None and getattr(solver, "name", None):
            shutil.copyfile(
                path, workdir / "solvers" / f"_submission_{path.stem}.py")
            names.append(solver.name)
    if not names:
        raise SystemExit(f"No `class Solver` found in {submission_dir}.")
    return names


def main(submission_dir, output_dir, benchmark_dir, input_dir):
    print(f"[ingestion] benchmark: {benchmark_dir}")
    if not benchmark_dir.exists():
        raise SystemExit(
            f"[ingestion] no benchmark at {benchmark_dir}: run inside a "
            "track image (which sets $COMPET_BENCHMARK_DIR) or pass "
            "--benchmark-dir. On Codabench, set the competition's docker "
            "image to the track image."
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
    if cfg.get("data_home"):
        os.environ["BENCHOPT_DATA_HOME"] = str(cfg["data_home"])
    # benchopt rejects unknown config keys: strip the competition-only
    # ones and pass the rest as a genuine `benchopt run` config file.
    cfg = {k: v for k, v in cfg.items() if k not in ("scoring", "data_home")}
    if cfg:
        run_config = workdir.parent / "run_config.yml"
        run_config.write_text(yaml.safe_dump(cfg))

    cmd = [
        "benchopt", "run", str(workdir),
        "-r", "1", "--no-cache", "--no-plot", "--no-html", "--no-display",
        "--output", "submission",
        *(["--config", str(run_config)] if run_config else []),
    ]
    for name in install_submission_solvers(submission_dir, workdir):
        cmd += ["-s", name]

    print(f"[ingestion] {' '.join(cmd)}", flush=True)
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
                        help="Phase dir: config.yaml (+ datasets/*.py)")
    parser.add_argument("--benchmark-dir",
                        default=os.environ.get("COMPET_BENCHMARK_DIR",
                                               BUNDLE_ROOT / "benchmark"),
                        help="The track's benchopt benchmark")
    args = parser.parse_args()

    main(
        Path(args.submission_dir),
        Path(args.output_dir),
        Path(args.benchmark_dir).resolve(),
        Path(args.input_data),
    )
