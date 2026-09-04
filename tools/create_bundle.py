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

# track dir (tracks/<name>) -> competition yaml (codabench/competition_*.yaml)
TRACKS = {
    "image_decoding": "competition_image.yaml",
    "bci_decoding": "competition_bci.yaml",
    "sleep_onset": "competition_sleep.yaml",
    "emg_pose": "competition_emg.yaml",
}

# Skip benchopt run artefacts / caches when zipping directories.
_SKIP_PARTS = ("outputs", "__cache__", "__pycache__", ".pytest_cache")


def _add_dir(bundle, src, arc_prefix):
    assert src.exists(), (
        f"{src} does not exist while it should. Make sure you followed the "
        "README instructions before creating the bundle."
    )
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        if f.name.startswith(".") or f.name.endswith(".pyc"):
            continue
        if any(part in _SKIP_PARTS for part in f.parts):
            continue
        arcname = Path(arc_prefix) / f.relative_to(src)
        print(arcname)
        bundle.write(f, arcname)


def build_bundle(track):
    yaml_name = TRACKS[track]
    out = ROOT_DIR / f"bundle_{track}.zip"
    with zipfile.ZipFile(out, mode="w") as bundle:
        # The track's competition config, under the canonical name.
        print(f"competition.yaml  <-  codabench/{yaml_name}")
        bundle.write(ROOT_DIR / "codabench" / yaml_name, "competition.yaml")
        bundle.write(ROOT_DIR / "logo.png", "logo.png")

        # The track's benchmark, under the canonical ``benchmark/`` name.
        _add_dir(bundle, ROOT_DIR / "tracks" / track, "benchmark")

        # Shared components: package + programs + pages + sample submission.
        _add_dir(bundle, ROOT_DIR / "compet_core", "compet_core")
        _add_dir(bundle, ROOT_DIR / "codabench" / "ingestion_program",
                 "ingestion_program")
        _add_dir(bundle, ROOT_DIR / "codabench" / "scoring_program",
                 "scoring_program")
        _add_dir(bundle, ROOT_DIR / "codabench" / "pages", "pages")
        _add_dir(bundle, ROOT_DIR / "solution" / track, "solution")
        _add_dir(bundle, ROOT_DIR / "dev_phase", "dev_phase")
    print(f"-> {out.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Codabench bundles")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--track", choices=sorted(TRACKS))
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    for track in (sorted(TRACKS) if args.all else [args.track]):
        build_bundle(track)
