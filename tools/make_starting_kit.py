"""Build one track's participant starting kit.

    python tools/make_starting_kit.py --track sleep_onset

Produces ``starting_kit_<track>.zip`` at the repo root::

    README.md            the repo tour
    tracks/<track>/      the benchmark, benchmark_utils dereferenced
    examples/*.zip       each baseline, ready to upload as a submission

The benchmark keeps its ``tracks/<track>/`` path so every command the
participation page documents (``benchopt install tracks/<track>``,
``benchopt run tracks/<track> ...``) works verbatim from the unzipped kit.

A baseline becomes an example by being a solver: ``solvers/<name>.py`` is
packaged as ``submission.py``, with a sibling ``<name>.<ext>`` carried
alongside as its ``weights`` file when one exists. Nothing is committed —
the archives are derived here, so they cannot drift from the benchmark the
workers run. ``push_all.py`` (neural-compet-aws) builds the kit in the same
run that publishes the phase ``input_data``.
"""

import argparse
import io
import zipfile
from pathlib import Path

# The symlink-following walk and the artefact skip list live with the bundle
# builder; a track's benchmark_utils is a link, and a plain glob would ship a
# benchmark with no shared code.
from create_bundle import ROOT_DIR, _SKIP_PARTS, _walk


def example_archive(solver):
    """One solver packaged as an upload-ready submission ZIP."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(solver, "submission.py")
        for weights in sorted(solver.parent.glob(f"{solver.stem}.*")):
            if weights.suffix != ".py":
                bundle.write(weights, f"weights{weights.suffix}")
    return buffer.getvalue()


def build(track):
    src = ROOT_DIR / "tracks" / track
    out = ROOT_DIR / f"starting_kit_{track}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as kit:
        kit.write(ROOT_DIR / "README.md", "README.md")
        for f in sorted(_walk(src)):
            if not f.is_file() or f.suffix in (".zip", ".pyc"):
                continue
            if f.name.startswith(".") or any(
                    part in _SKIP_PARTS for part in f.parts):
                continue
            kit.write(f, Path("tracks") / track / f.relative_to(src))
        for solver in sorted((src / "solvers").glob("*.py")):
            kit.writestr(f"examples/{solver.stem}.zip",
                         example_archive(solver))
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
