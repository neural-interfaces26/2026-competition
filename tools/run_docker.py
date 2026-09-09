"""Run the ingestion + scoring programs in Docker for one track (local test).

Mirrors the Codabench bundle layout: the track's benchmark is mounted as
``/app/benchmark`` and ``compet_core`` next to the programs.

Usage
-----
    python tools/run_docker.py --track bci_decoding [--datasets Simulated]
"""

import argparse
from pathlib import Path

try:
    import docker
except ImportError:
    raise ImportError(
        "The 'docker' package is required to run this script. "
        "Please install it using 'pip install docker'."
    )

REPO = Path(__file__).resolve().parent.parent
IMAGE = "tommoral/compet-neural-interfaces:v1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docker ingestion test")
    parser.add_argument("--track", required=True)
    parser.add_argument("--datasets", nargs="*", default=["Simulated"])
    args = parser.parse_args()

    client = docker.from_env()
    print("Building Docker image...")
    client.images.build(path=str(REPO), dockerfile="tools/Dockerfile",
                        tag=IMAGE)

    print("Running ingestion...")
    cmd = "python3 /app/ingestion_program/ingestion.py --datasets " \
        + " ".join(args.datasets)
    logs = client.containers.run(
        image=IMAGE, command=cmd, remove=True, name="ingestion", user="root",
        volumes=[
            f"{REPO}/codabench/ingestion_program:/app/ingestion_program",
            f"{REPO}/compet_core:/app/compet_core",
            f"{REPO}/tracks/{args.track}:/app/benchmark",
            f"{REPO}/ingestion_res:/app/output",
            f"{REPO}/solution/{args.track}:/app/ingested_program",
        ]
    )
    print(logs.decode("utf-8"))

    print("Running scoring...")
    logs = client.containers.run(
        image=IMAGE, command="python3 /app/scoring_program/scoring.py",
        remove=True, name="scoring", user="root",
        volumes=[
            f"{REPO}/codabench/scoring_program:/app/scoring_program",
            f"{REPO}/ingestion_res:/app/input/res",
            f"{REPO}/scoring_res:/app/output",
        ]
    )
    print(logs.decode("utf-8"))
    print("Docker container ran successfully.")
