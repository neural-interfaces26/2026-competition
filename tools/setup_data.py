"""Prepare the competition data.

Thin wrapper over ``benchopt prepare`` for the bundled benchmark: it triggers
each ``Dataset.prepare()`` (MOABB BNCI2014_001 for motor imagery, Sleep-EDF for
sleep staging), downloading into benchopt's configured data folder
(``get_data_path``). Re-runnable: already-downloaded files are skipped.

No-network smoke testing needs nothing here — the zero-dependency ``Simulated``
dataset always works (``benchopt run benchmark/ -d Simulated``).

Usage
-----
    python tools/setup_data.py                 # prepare all real datasets
    python tools/setup_data.py -d MOABB-MI     # prepare a single dataset
"""

import argparse
import subprocess
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"


def main(datasets):
    cmd = [sys.executable, "-m", "benchopt", "prepare", str(BENCHMARK_DIR)]
    for d in datasets or []:
        cmd += ["-d", d]
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare competition data")
    parser.add_argument(
        "-d", "--datasets", nargs="*", default=None,
        help="Dataset names to prepare (default: all real datasets).",
    )
    args = parser.parse_args()
    main(args.datasets)
