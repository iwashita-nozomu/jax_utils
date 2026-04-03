from __future__ import annotations

import os
from pathlib import Path
import sys


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = WORKTREE_ROOT / "python"

if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


# Prevent JAX from preallocating most GPU memory during pytest runs.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
