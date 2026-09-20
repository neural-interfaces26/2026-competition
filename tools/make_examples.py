"""Build the upload-ready ZIPs for the tracks' worked examples.

    python tools/make_examples.py                    # every track
    python tools/make_examples.py --track sleep_onset

An example lives in ``tracks/<track>/solvers/<NN>_<name>/`` and ships its
sources, not its archive: the ZIP is derived here so it cannot drift from
the files it is built from, and stays out of git.

Everything in the folder is packaged except the README and the archive
itself, with the file defining ``class Solver`` renamed to
``submission.py`` — the layout the ingestion program expects.

A folder shipping only sources is a walkthrough rather than an upload
example (benchopt builds that one's archive from a training run), so it is
skipped. ``tools/make_starting_kit.py`` packs the same archives into the
participants' starting kit.
"""

import argparse
import io
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SKIP = {"README.md"}


def example_folders(track):
    """The track's numbered example directories, in order."""
    solvers = ROOT_DIR / "tracks" / track / "solvers"
    return [f for f in sorted(solvers.glob("[0-9][0-9]_*")) if f.is_dir()]


def _solver_file(folder):
    """The example's solver source (the one defining ``class Solver``)."""
    found = [f for f in sorted(folder.glob("*.py"))
             if "class Solver" in f.read_text(encoding="utf-8")]
    if len(found) != 1:
        raise SystemExit(
            f"{folder} must hold exactly one file defining `class Solver`, "
            f"found {[f.name for f in found]}.")
    return found[0]


def archive_bytes(folder):
    """The example's upload-ready ZIP, or None when it is a walkthrough."""
    solver = _solver_file(folder)
    files = [f for f in sorted(folder.iterdir())
             if f.is_file() and f.suffix != ".zip" and f.name not in SKIP]
    if all(f.suffix == ".py" for f in files):
        return None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(
                path, "submission.py" if path == solver else path.name)
    return buffer.getvalue()


def build(folder):
    payload = archive_bytes(folder)
    name = folder.relative_to(ROOT_DIR)
    if payload is None:
        print(f"{name}  --  walkthrough, no archive")
        return
    archive = folder / f"{folder.name}.zip"
    archive.write_bytes(payload)
    with zipfile.ZipFile(archive) as bundle:
        print(f"{archive.relative_to(ROOT_DIR)}  <-  "
              f"{', '.join(bundle.namelist())}")


def main(track):
    tracks = [track] if track else sorted(
        p.name for p in (ROOT_DIR / "tracks").iterdir() if p.is_dir())
    folders = [f for t in tracks for f in example_folders(t)]
    if not folders:
        raise SystemExit(f"No example folders under tracks/{track or '*'}/")
    for folder in folders:
        build(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the example ZIPs")
    parser.add_argument(
        "--track", choices=sorted(p.name for p in (ROOT_DIR / "tracks")
                                  .iterdir() if p.is_dir()))
    args = parser.parse_args()
    main(args.track)
