import json
from pathlib import Path

import jax

SOURCE_FILE = Path(__file__).name

print(json.dumps({
	"case": "jax_env",
	"source_file": SOURCE_FILE,
	"event": "version",
	"jax_version": jax.__version__,
}))
print(json.dumps({
	"case": "jax_env",
	"source_file": SOURCE_FILE,
	"event": "devices",
	"devices": [str(device) for device in jax.devices()],
}))

from ..base import *


A :LinearOperator = jax.numpy.array([[1.,2.],[3.,4.]])