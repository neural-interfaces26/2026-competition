"""Locate the shared ``compet_core`` package (no install required).

Importing this module walks up from the benchmark directory to the first
parent holding ``compet_core/`` — the repo root in a checkout, the bundle
root on Codabench — and puts it on ``sys.path``. Every benchmark module
does ``import benchmark_utils`` before importing ``compet_core``, so the
benchmark works from a plain ``git clone`` (no ``pip install -e .``) and
inside the competition bundle alike.
"""

import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "compet_core" / "__init__.py").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break
