"""Prepare the competition data.

Thin wrapper over ``benchopt prepare`` for the track benchmarks: it triggers
each ``Dataset.prepare()``, downloading into benchopt's configured data folder
(``get_data_path``). Re-runnable: already-downloaded files are skipped.

No-network smoke testing needs nothing here — the zero-dependency
``Simulated`` dataset always works (``benchopt run tracks/<t> -d Simulated``).

Usage
-----
    python tools/setup_data.py --track bci_decoding             # all datasets
    python tools/setup_data.py --track bci_decoding -d "BCI[study=tangermann2012]"
"""

import argparse
import subprocess
import sys
from pathlib import Path

TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"


def main(track, datasets):
    cmd = [sys.executable, "-m", "benchopt", "prepare",
           str(TRACKS_DIR / track)]
    for d in datasets or []:
        cmd += ["-d", d]
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare competition data")
    parser.add_argument(
        "--track", required=True,
        choices=sorted(p.name for p in TRACKS_DIR.iterdir() if p.is_dir()),
    )
    parser.add_argument(
        "-d", "--datasets", nargs="*", default=None,
        help="Dataset names to prepare (default: all real datasets).",
    )
    args = parser.parse_args()
    main(args.track, args.datasets)
