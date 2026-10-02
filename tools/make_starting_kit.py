"""Build one track's participant starting kit.

    python tools/make_starting_kit.py --track sleep_onset

Produces ``starting_kit_<track>.zip`` at the repo root::

    README.md            the repo tour
    tracks/README.md     the shared development and packaging workflows
    tracks/<track>/      the benchmark, benchmark_utils dereferenced
    examples/*.zip       each baseline, ready to upload as a submission

The benchmark keeps its ``tracks/<track>/`` path so every command the
participation page documents (``benchopt install tracks/<track>``,
``benchopt run tracks/<track> ...``) works verbatim from the unzipped kit.

A baseline is a solver: ``solvers/<name>.py`` is packaged as ``submission.py``.
Its trained weights, if any, come from the ``outputs/<Solver.name>/`` folder a
training run writes; they travel at the ZIP root, where ``load_model`` reads.
A solver that trains (defines ``save_model``) but has no such folder is shipped
untrained, with a warning. Nothing is committed, so the examples cannot drift
from the benchmark the workers run.
"""

import argparse
import io
import re
import zipfile
from pathlib import Path

# The symlink-following walk and the artefact skip list live with the bundle
# builder; a track's benchmark_utils is a link, and a plain glob would ship a
# benchmark with no shared code. ``_SKIP_PARTS`` also excludes ``outputs`` from
# the shipped tree — trained weights travel in the example ZIPs instead.
from create_bundle import ROOT_DIR, _SKIP_PARTS, _walk

_NAME_RE = re.compile(r'^\s*name\s*=\s*["\'](.+?)["\']', re.M)


def example_archive(solver, weights_dir):
    """One solver packaged as an upload-ready submission ZIP.

    Ships ``submission.py`` plus any weight files a training run left in
    ``weights_dir`` (the solver's ``outputs/<name>/``). Returns
    ``(bytes, trained)`` — ``trained`` is False when no weights were found.
    """
    buffer = io.BytesIO()
    trained = False
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(solver, "submission.py")
        if weights_dir.is_dir():
            for f in sorted(weights_dir.iterdir()):
                if f.is_file() and f.name != "submission.py":
                    bundle.write(f, f.name)
                    trained = True
    return buffer.getvalue(), trained


def build(track):
    src = ROOT_DIR / "tracks" / track
    outputs = src / "outputs"
    out = ROOT_DIR / f"starting_kit_{track}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as kit:
        kit.write(ROOT_DIR / "README.md", "README.md")
        kit.write(ROOT_DIR / "tracks" / "README.md", "tracks/README.md")
        for f in sorted(_walk(src)):
            if not f.is_file() or f.suffix in (".zip", ".pyc"):
                continue
            if f.name.startswith(".") or any(
                    part in _SKIP_PARTS for part in f.parts):
                continue
            kit.write(f, Path("tracks") / track / f.relative_to(src))
        for solver in sorted((src / "solvers").glob("*.py")):
            source = solver.read_text()
            match = _NAME_RE.search(source)
            name = match.group(1) if match else solver.stem
            data, trained = example_archive(solver, outputs / name)
            kit.writestr(f"examples/{solver.stem}.zip", data)
            if "def save_model" in source and not trained:
                print(f"  ! {solver.stem}: no trained weights in "
                      f"outputs/{name}/ — shipping it untrained")
        count = len(kit.namelist())
    print(f"{out.relative_to(ROOT_DIR)}  ({count} files, "
          f"{out.stat().st_size // 1024} kB)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a starting kit")
    parser.add_argument(
        "--track", required=True,
        choices=sorted(p.name for p in (ROOT_DIR / "tracks").iterdir()
                       if p.is_dir()))
    args = parser.parse_args()
    build(args.track)
