"""Make ``trash_sorter`` importable when a sim script runs under the Isaac Sim Python.

Isaac Sim lives in its own venv (see README), so the repo root is added to ``sys.path``
instead of installing this project into it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "outputs"
DATA_DIR = REPO_ROOT / "data"
