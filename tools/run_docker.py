"""Build the image and run ingestion + scoring in it (local test).

Mirrors the platform layout: the phase bundle (benchmark + config) is
mounted as /app/input_data and a submission as /app/ingested_program.

Usage
-----
    python tools/run_docker.py --track bci_decoding [--data <host-data-dir>]
    python tools/run_docker.py --track bci_decoding --submission my.zip
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
    parser.add_argument("--submission", default=None,
                        help="Zip to evaluate (default: the track's sample "
                             "in solution/)")
    args = parser.parse_args()

    client = docker.from_env()
    image = "tommoral/neural-compet:dev"
    print(f"Building {image}...")
    client.images.build(path=str(REPO), dockerfile="tools/Dockerfile",
                        tag=image)

    print("Running ingestion...")
    work = Path(tempfile.mkdtemp(prefix="run_docker_"))
    # Materialize symlink in temp dir to run in the container.
    phase = work / "input_data"
    shutil.copytree(REPO / "codabench" / "phases" / "warmup" / args.track,
                    phase)
    if args.submission:
        submission = work / "submission"
        shutil.unpack_archive(args.submission, submission)
    else:
        submission = REPO / "solution" / args.track
    volumes = [
        f"{phase}:/app/input_data",
        f"{submission}:/app/ingested_program",
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
