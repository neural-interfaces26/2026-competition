#!/usr/bin/env python3
"""Train, validate, and package an uploadable synthetic debug submission.

The signals and targets are synthetic, but their tensor dimensions match the
current Codabench warm-up contract. The resulting score is meaningless; the
ZIP is intended to exercise the complete Benchopt-to-Codabench workflow
without downloading the public datasets.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


TRACKS = {
    "image_decoding": {
        "objective": "Image-decoding",
        "dataset": (
            "Simulated[n_chans=63,n_times=120,n_images=8,n_outputs=1536,"
            "n_train=12,n_test=8,sfreq=120]"
        ),
        "models": {
            "eegnet": {
                "solver": "EEGNet-CLIP", "artifact": "weights.pt",
                "label": "track01_eegnet_clip_debug",
            },
            "reve": {
                "solver": "REVE", "artifact": "weights.joblib",
                "label": "track01_reve_debug",
            },
        },
    },
    "bci_decoding": {
        "objective": "BCI-decoding",
        "dataset": (
            "Simulated[n_chans=27,n_times=480,n_classes=2,"
            "n_train=16,n_test=8,sfreq=120]"
        ),
        "models": {
            "eegnet": {
                "solver": "EEGNet", "artifact": "weights.pt",
                "label": "track02_eegnet_debug",
            },
            "reve": {
                "solver": "REVE", "artifact": "weights.joblib",
                "label": "track02_reve_debug",
            },
        },
    },
    "sleep_onset": {
        "objective": "Sleep-onset",
        "dataset": (
            "Simulated[n_chans=2,n_times=600,n_train=16,n_test=8,sfreq=120]"
        ),
        "models": {
            "eegnet": {
                "solver": "EEGNet", "artifact": "weights.pt",
                "label": "track03_eegnet_debug",
            },
            "reve": {
                "solver": "REVE", "artifact": "weights.joblib",
                "label": "track03_reve_debug",
            },
        },
    },
    "emg_pose": {
        "objective": "EMG-pose",
        "dataset": (
            "Simulated[n_chans=16,n_joints=20,n_times=10000,"
            "n_train=4,n_test=2,sfreq=2000]"
        ),
        "models": {
            "eegnet": {
                "solver": "EEGNet", "artifact": "weights.pt",
                "label": "track04_eegnet_debug",
            },
        },
    },
}


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def package_track(
    repo: Path,
    benchopt: str,
    track: str,
    destination: Path,
    model: str,
    install: bool,
    gpu: bool,
) -> Path:
    cfg = TRACKS[track]
    model_cfg = cfg["models"].get(model)
    if model_cfg is None:
        raise ValueError(f"{model} is not available for {track}")
    benchmark = repo / "tracks" / track

    if install:
        command = [
            benchopt, "install", str(benchmark),
            "-d", "Simulated", "-s", model_cfg["solver"], "-y",
        ]
        if gpu:
            command.append("--gpu")
        run(command)

    destination.mkdir(parents=True, exist_ok=True)
    final_dir = destination / model_cfg["label"]
    final_zip = destination / f"{model_cfg['label']}.zip"
    if final_dir.exists() or final_zip.exists():
        raise FileExistsError(
            f"{final_dir} or {final_zip} already exists; remove or rename it"
        )

    with tempfile.TemporaryDirectory(prefix=f"ni26-{track}-") as tmp:
        tmp_path = Path(tmp)
        package = tmp_path / "submission"
        package.mkdir()
        env = os.environ.copy()
        env["COMPET_SUBMISSION_DIR"] = str(package)

        common = [
            benchopt, "run", str(benchmark),
            "-d", cfg["dataset"],
            "-r", "1", "--no-cache", "--no-plot", "--no-html",
            "--no-display", "--local",
        ]
        run(
            common + [
                "-o", f"{cfg['objective']}[training=True]",
                "-s", model_cfg["solver"],
                "--output", str(tmp_path / "training.parquet"),
            ],
            env=env,
        )

        submission = package / "submission.py"
        weights = package / model_cfg["artifact"]
        if not submission.is_file() or not weights.is_file():
            raise RuntimeError(
                "Benchopt did not export submission.py and "
                f"{model_cfg['artifact']}"
            )

        # Reload the exported files in inference-only mode with the same real
        # tensor contract. This catches architecture/checkpoint mismatches
        # before the ZIP is produced.
        run(
            common + [
                "-o", f"{cfg['objective']}[training=False]",
                "-s", str(submission),
                "--output", str(tmp_path / "validation.parquet"),
            ],
            env=env,
        )

        shutil.copytree(package, final_dir)
        with zipfile.ZipFile(
            final_zip, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(package.iterdir()):
                if path.is_file():
                    archive.write(path, path.name)

    print(f"Ready to upload: {final_zip}")
    return final_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--track", choices=TRACKS)
    choice.add_argument("--all", action="store_true")
    parser.add_argument(
        "--model", choices=("eegnet", "reve"), default="eegnet",
        help="model to train (default: eegnet; REVE supports EEG tracks)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="destination (default: <repo>/debug_submissions)",
    )
    parser.add_argument(
        "--skip-install", action="store_true",
        help="reuse dependencies already installed in the active environment",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="install Benchopt's GPU requirements instead of CPU requirements",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    destination = (args.output_dir or repo / "debug_submissions").resolve()
    benchopt = shutil.which("benchopt")
    sibling_benchopt = Path(sys.executable).with_name("benchopt")
    if benchopt is None and sibling_benchopt.is_file():
        benchopt = str(sibling_benchopt)
    if benchopt is None:
        raise SystemExit(
            "benchopt is not installed in this environment; install "
            "benchopt>=1.10 first"
        )

    selected = list(TRACKS) if args.all else [args.track]
    selected = [
        track for track in selected if args.model in TRACKS[track]["models"]
    ]
    if not selected:
        raise SystemExit(f"{args.model} is not available for {args.track}")
    for track in selected:
        package_track(
            repo, benchopt, track, destination,
            model=args.model,
            install=not args.skip_install, gpu=args.gpu,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
