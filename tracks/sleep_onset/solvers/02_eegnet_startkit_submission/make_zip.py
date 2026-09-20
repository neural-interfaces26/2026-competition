"""Build the ready-to-upload EEGNet submission archive."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "eegnet-sleep-onset-startkit.zip"

with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED) as bundle:
    bundle.write(HERE / "eegnet_reg.py", "submission.py")
    bundle.write(HERE / "weights.pt", "weights.pt")

print(f"Ready to upload: {ARCHIVE}")
