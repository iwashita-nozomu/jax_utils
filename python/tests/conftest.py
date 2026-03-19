from __future__ import annotations

import os

# Prevent JAX from preallocating most GPU memory during pytest runs.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
