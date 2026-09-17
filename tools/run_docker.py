"""Build the image and run ingestion + scoring in it (local test).

Mirrors the platform layout: the phase bundle (benchmark + config) is
mounted as /app/input_data and a sample submission as /app/ingested_program.
The programs come from the image (Codabench delivers its own copies).

Usage
-----
    python tools/run_docker.py --track bci_decoding [--data <host-data-dir>]
"""

import argparse
import shutil
import tempfile
from pathlib import Path

try:
    import docker
except ImportError:
    raise ImportError(
        "The 'docker' package is required to run this script. "
        "Please install it using 'pip install docker'."
    )

REPO = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docker ingestion test")
    parser.add_argument("--track", required=True)
    parser.add_argument("--data", default=None,
                        help="Host data dir to mount as /data")
    args = parser.parse_args()

    client = docker.from_env()
    image = "tommoral/neural-compet:dev"
    print(f"Building {image}...")
    client.images.build(path=str(REPO), dockerfile="tools/Dockerfile",
                        tag=image)

    print("Running ingestion...")
    # The phase dir links to the benchmark, which links to the shared
    # benchmark_utils; a bind mount would carry the links into the container,
    # where they point nowhere. Materialize the bundle first (copytree
    # dereferences), exactly as create_bundle and the staging script do.
    phase = Path(tempfile.mkdtemp(prefix="phase_")) / "input_data"
    shutil.copytree(REPO / "codabench" / "phases" / "warmup" / args.track,
                    phase)
    volumes = [
        f"{phase}:/app/input_data",
        f"{REPO}/solution/{args.track}:/app/ingested_program",
        f"{REPO}/ingestion_res:/app/output",
    ]
    if args.data:
        volumes.append(f"{args.data}:/data")
    logs = client.containers.run(
        image=image, remove=True, name="ingestion", user="root",
        command="python3 /compet/ingestion_program/ingestion.py",
        volumes=volumes,
    )
    print(logs.decode("utf-8"))

    print("Running scoring...")
    logs = client.containers.run(
        image=image, remove=True, name="scoring", user="root",
        command="python3 /compet/scoring_program/scoring.py",
        volumes=[
            f"{REPO}/ingestion_res:/app/input/res",
            f"{REPO}/scoring_res:/app/output",
        ],
    )
    print(logs.decode("utf-8"))
    print("Docker container ran successfully.")
