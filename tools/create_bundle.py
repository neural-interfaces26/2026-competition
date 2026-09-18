"""Build a Codabench bundle for one track.

Each of the 4 tracks becomes its own Codabench competition. The bundle ships
its competition yaml (as ``competition.yaml``), the pages, the sample
submission, and the warm-up phase directory as ``input_data`` — config plus
the benchmark it runs, ``benchmark_utils`` included.

Usage
-----
    python tools/create_bundle.py --track bci_decoding
    python tools/create_bundle.py --all
"""

import argparse
import os
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# track dir (tracks/<name>) -> short key, shared by the track's competition
# yaml (codabench/competition_<key>.yaml) and its description page
# (codabench/pages/competition_<key>.md).
TRACKS = {
    "image_decoding": "image",
    "bci_decoding": "bci",
    "sleep_onset": "sleep",
    "emg_pose": "emg",
}

# Skip benchopt run artefacts / caches / downloaded data when zipping.
_SKIP_PARTS = ("outputs", "__cache__", "__pycache__", ".pytest_cache", "data")


def _walk(src):
    """Every file under *src*, following symlinks.

    A track links its ``benchmark_utils`` to the shared one, and ``rglob``
    does not descend into a symlinked directory — it would silently bundle a
    benchmark with no shared code.
    """
    for parent, _dirs, files in os.walk(src, followlinks=True):
        for name in files:
            yield Path(parent) / name


def _add_dir(bundle, src, arc_prefix, exclude=None):
    assert src.exists(), (
        f"{src} does not exist while it should. Make sure you followed the "
        "README instructions before creating the bundle."
    )
    for f in sorted(_walk(src)):
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


def build_bundle(track):
    key = TRACKS[track]
    yaml_name = f"competition_{key}.yaml"
    out = ROOT_DIR / f"bundle_{track}.zip"
    with zipfile.ZipFile(out, mode="w") as bundle:
        # The track's competition config, under the canonical name.
        print(f"competition.yaml  <-  codabench/{yaml_name}")
        bundle.write(ROOT_DIR / "codabench" / yaml_name, "competition.yaml")
        bundle.write(ROOT_DIR / "logo.jpg", "logo.jpg")

        # Shared components: programs + pages + sample submission. The
        # benchmark travels with the phase, below, not at the bundle root.
        _add_dir(bundle, ROOT_DIR / "codabench" / "ingestion_program",
                 "ingestion_program")
        _add_dir(bundle, ROOT_DIR / "codabench" / "scoring_program",
                 "scoring_program")
        # Only this track's description: the other three are other bundles'.
        _add_dir(bundle, ROOT_DIR / "codabench" / "pages", "pages",
                 exclude=lambda f: (f.name.startswith("competition_")
                                    and f.name != f"competition_{key}.md"))
        _add_dir(bundle, ROOT_DIR / "solution" / track, "solution")

        # Warm-up task data: the phase directory as it stands — config,
        # optional sealed datasets/*.py, and the benchmark it runs (a link to
        # tracks/<track>), which the image no longer carries. Plus the shared
        # placeholder reference.
        _add_dir(bundle, ROOT_DIR / "codabench" / "phases" / "warmup" / track,
                 "warmup_phase/input_data")
        _add_dir(bundle, ROOT_DIR / "codabench" / "phases" / "reference_data",
                 "warmup_phase/reference_data")
    print(f"-> {out.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Codabench bundles")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--track", choices=sorted(TRACKS))
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    for track in (sorted(TRACKS) if args.all else [args.track]):
        build_bundle(track)
