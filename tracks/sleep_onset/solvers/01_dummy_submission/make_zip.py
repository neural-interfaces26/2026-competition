"""Build the upload-ready dummy Track 03 submission."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "dummy-sleep-linear.zip"
FILES = {
    "dummy_submission.py": "submission.py",
    "dummy_weights.pt": "dummy_weights.pt",
    "dummy_config.json": "dummy_config.json",
}

with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as archive:
    for source, destination in FILES.items():
        archive.write(HERE / source, destination)

print(f"Created {OUTPUT}")
