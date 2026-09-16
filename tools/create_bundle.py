"""Build a Codabench bundle for one track.

Each of the 4 tracks becomes its own Codabench competition; the bundle ships
the track's benchopt benchmark (as ``benchmark/``), the shared ``compet_core``
package, the shared ingestion/scoring programs, its competition yaml (as
``competition.yaml``) and the sample submission.

Usage
-----
    python tools/create_bundle.py --track bci_decoding
    python tools/create_bundle.py --all
"""

import argparse
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# track dir (tracks/<name>) -> short key, shared by the track's competition
# yaml (codabench/competition_<key>.yaml) and its competition page
# (codabench/pages/competition_<key>.html).
TRACKS = {
    "image_decoding": "image",
    "bci_decoding": "bci",
    "sleep_onset": "sleep",
    "emg_pose": "emg",
}

# Skip benchopt run artefacts / caches / downloaded data when zipping.
_SKIP_PARTS = ("outputs", "__cache__", "__pycache__", ".pytest_cache", "data")


def _add_dir(bundle, src, arc_prefix, exclude=None):
    assert src.exists(), (
        f"{src} does not exist while it should. Make sure you followed the "
        "README instructions before creating the bundle."
    )
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        if f.name.startswith(".") or f.name.endswith(".pyc"):
            continue
        if exclude is not None and exclude(f):
            continue
        if any(part in _SKIP_PARTS for part in f.parts):
            continue
        arcname = Path(arc_prefix) / f.relative_to(src)
        print(arcname)
        bundle.write(f, arcname)


def _competition_page(key):
    """Assemble ``pages/competition.html`` for one track.

    Shared head (scoped styles) + the track-specific body + shared tail (the
    four-track overview and the sponsor / institution logo wall), so the parts
    common to the 4 competitions live in a single file.
    """
    pages = ROOT_DIR / "codabench" / "pages"
    parts = ["_competition_head.html", f"competition_{key}.html",
             "_competition_tail.html"]
    out = []
    for name in parts:
        f = pages / name
        assert f.exists(), f"{f} does not exist while it should."
        out.append(f.read_text(encoding="utf-8"))
    return "".join(out)


def build_bundle(track):
    key = TRACKS[track]
    yaml_name = f"competition_{key}.yaml"
    out = ROOT_DIR / f"bundle_{track}.zip"
    with zipfile.ZipFile(out, mode="w") as bundle:
        # The track's competition config, under the canonical name.
        print(f"competition.yaml  <-  codabench/{yaml_name}")
        bundle.write(ROOT_DIR / "codabench" / yaml_name, "competition.yaml")
        bundle.write(ROOT_DIR / "logo.jpg", "logo.jpg")

        # The track's competition page, assembled from the shared fragments.
        print(f"pages/competition.html  <-  pages/competition_{key}.html")
        bundle.writestr("pages/competition.html", _competition_page(key))

        # The track's benchmark, under the canonical ``benchmark/`` name.
        _add_dir(bundle, ROOT_DIR / "tracks" / track, "benchmark")

        # Shared components: package + programs + pages + sample submission.
        _add_dir(bundle, ROOT_DIR / "compet_core", "compet_core")
        _add_dir(bundle, ROOT_DIR / "codabench" / "ingestion_program",
                 "ingestion_program")
        _add_dir(bundle, ROOT_DIR / "codabench" / "scoring_program",
                 "scoring_program")
        # The competition-page fragments are assembled above, not copied.
        _add_dir(bundle, ROOT_DIR / "codabench" / "pages", "pages",
                 exclude=lambda f: f.name.startswith(("_", "competition_")))
        _add_dir(bundle, ROOT_DIR / "solution" / track, "solution")

        # Dev-phase task data: the track's phase config (+ optional sealed
        # datasets/*.py) as input_data, and the shared placeholder reference.
        _add_dir(bundle, ROOT_DIR / "codabench" / "phases" / "dev" / track,
                 "dev_phase/input_data")
        _add_dir(bundle, ROOT_DIR / "codabench" / "phases" / "reference_data",
                 "dev_phase/reference_data")
    print(f"-> {out.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Codabench bundles")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--track", choices=sorted(TRACKS))
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    for track in (sorted(TRACKS) if args.all else [args.track]):
        build_bundle(track)
