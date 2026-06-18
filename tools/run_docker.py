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
    client = docker.from_env()
    print("Docker client initialized successfully.")

    print("Building Docker image...")
    client.images.build(path=".", tag="tommoral/template:v1")
    print("Docker image built successfully with tag 'tommoral/template:v1'.")

    print("Running Docker container...")
    logs = client.containers.run(
        image="tommoral/template:v1",
        command="python3 /app/ingestion_program/ingestion.py",
        remove=True,
        name="ingestion",
        user="root",
        volumes=[
            f"{REPO}/ingestion_program:/app/ingestion_program",
            f"{REPO}/dev_phase/input_data:/app/input_data",
            f"{REPO}/ingestion_res:/app/output",
            f"{REPO}/solution:/app/ingested_program",
        ]
    )
    print(logs.decode("utf-8"))
    logs = client.containers.run(
        image="tommoral/template:v1",
        command="python3 /app/scoring_program/scoring.py",
        remove=True,
        name="scoring",
        user="root",
        volumes=[
            f"{REPO}/scoring_program:/app/scoring_program",
            f"{REPO}/dev_phase/reference_data:/app/input/ref",
            f"{REPO}/ingestion_res:/app/input/res",
            f"{REPO}/scoring_res:/app/",
        ]
    )
    print(logs.decode("utf-8"))
    print("Docker container ran successfully.")
